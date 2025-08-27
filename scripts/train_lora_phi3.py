from __future__ import annotations
import os, json, argparse
from dataclasses import dataclass
from typing import Dict, List, Any

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
)
from peft import LoraConfig, get_peft_model

# Reusa EXACTAMENTE el mismo prompt template que tu planner
SYSTEM = (
    "You are TAD Agent, an expert Autodesk Revit planner.\n"
    "Return ONE JSON object only. No markdown, no code fences, no prose.\n"
    "JSON schema:{\"version\":\"tad-dsl/0.2\",\"context\":{\"revit_version\":\"2025\",\"units\":\"SI\"},"
    "\"plan\":[{\"action\":\"<action>\",\"args\":{...},\"as\":\"optional-alias\"}]}.\n"
    "Use SI units consistently. If information is missing, add an initial 'get' step.\n"
    "Use only action names from the catalog. Each step MUST have `action` and `args` (object).\n"
    "Your output MUST be STRICT JSON (double quotes, true/false/null, no trailing commas).\n"
    "Never output placeholders like \"<action>\" or empty args. Begin with '{' and end with '}'.\n"
)

# Genera el catálogo de acciones en tiempo de carga (idéntico a /tools)
def build_actions_catalog() -> Dict[str, Any]:
    from shared.tad_dsl.tool_definitions import ACTION_SCHEMAS
    acts = []
    for name, model in ACTION_SCHEMAS.items():
        fields = getattr(model, "model_fields", {})
        acts.append({"action": name, "args": list(fields.keys())})
    return {"available_actions": acts}

def build_messages(tokenizer, user_prompt: str, context: Dict[str, Any]) -> str:
    actions = build_actions_catalog()
    messages = [
        {"role": "system", "content":
            SYSTEM
            + "\nAvailable actions:\n" + json.dumps(actions, ensure_ascii=False)
            + "\nContext:\n" + json.dumps({"client_context": context or {}}, ensure_ascii=False)
        },
        {"role": "user", "content": f"{user_prompt}\nReturn ONLY the JSON object."}
    ]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

@dataclass
class DataCollatorForSFT:
    tokenizer: Any
    pad_to_multiple_of: int = 8

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # pad inputs to same length
        batch = {}
        keys = ["input_ids", "attention_mask", "labels"]
        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            # round up
            rem = max_len % self.pad_to_multiple_of
            if rem != 0:
                max_len += (self.pad_to_multiple_of - rem)

        for k in keys:
            tensors = []
            for f in features:
                x = f[k]
                if len(x) < max_len:
                    pad_id = self.tokenizer.pad_token_id
                    if k == "labels":
                        pad_val = -100
                    else:
                        pad_val = pad_id if k == "input_ids" else 0
                    x = x + [pad_val] * (max_len - len(x))
                tensors.append(torch.tensor(x, dtype=torch.long))
            batch[k] = torch.stack(tensors, dim=0)
        return batch

def build_sample(tokenizer, prompt: str, context: Dict[str, Any], target_obj: Dict[str, Any], eos: str) -> Dict[str, List[int]]:
    # prompt tokens
    prompt_text = build_messages(tokenizer, prompt, context)
    target_text = json.dumps(target_obj, ensure_ascii=False) + eos

    prompt_ids  = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    target_ids  = tokenizer(target_text, add_special_tokens=False)["input_ids"]

    input_ids = prompt_ids + target_ids
    labels    = [-100] * len(prompt_ids) + target_ids  # pérdida SOLO en la respuesta

    attn_mask = [1] * len(input_ids)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attn_mask}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default=os.environ.get("MODEL_ID","microsoft/Phi-3-mini-4k-instruct"))
    ap.add_argument("--train", default="data/sft/train.jsonl")
    ap.add_argument("--eval", default="data/sft/eval.jsonl")
    ap.add_argument("--out", default="out/phi3-tad-lora")
    ap.add_argument("--seq_len", type=int, default=1536)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[info] loading base model:", args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False

    # LoRA config (ajusta a tu gusto)
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # dataset
    def load_jsonl(path):
        ds = load_dataset("json", data_files=path, split="train")
        def _proc(example):
            prompt  = example["prompt"]
            context = example.get("context") or {}
            target  = example["target"]
            sample  = build_sample(tokenizer, prompt, context, target, tokenizer.eos_token)
            # recorta a seq_len (simple)
            for k in ("input_ids","labels","attention_mask"):
                sample[k] = sample[k][:args.seq_len]
            return sample
        return ds.map(_proc, remove_columns=ds.column_names)

    train_ds = load_jsonl(args.train)
    eval_ds  = load_jsonl(args.eval) if os.path.exists(args.eval) else None

    collator = DataCollatorForSFT(tokenizer)

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        bf16=False,
        fp16=True,
        gradient_checkpointing=True,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        evaluation_strategy="steps" if eval_ds is not None else "no",
        eval_steps=500 if eval_ds is not None else None,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"[ok] LoRA adapter saved to: {args.out}")

if __name__ == "__main__":
    main()