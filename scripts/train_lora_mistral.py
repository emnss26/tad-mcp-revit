#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json, argparse, warnings, re
from typing import Dict, List, Any

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

# Catálogo de acciones opcional
try:
    from shared.tad_dsl.tool_definitions import ACTION_SCHEMAS  # type: ignore
    ACTIONS_AVAILABLE = True
except Exception:
    ACTION_SCHEMAS = {}
    ACTIONS_AVAILABLE = False
    warnings.warn("[warn] No pude importar ACTION_SCHEMAS. Continuo sin listado de acciones.")

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

def build_actions_catalog() -> Dict[str, Any]:
    acts = []
    for name, model in ACTION_SCHEMAS.items():
        fields = getattr(model, "model_fields", {})
        acts.append({"action": name, "args": list(fields.keys())})
    return {"available_actions": acts}

def build_messages(tokenizer, user_prompt: str, context: Dict[str, Any] | None) -> str:
    sys_txt = SYSTEM
    if ACTIONS_AVAILABLE:
        sys_txt += "\nAvailable actions:\n" + json.dumps(build_actions_catalog(), ensure_ascii=False)
    if context:
        sys_txt += "\nContext:\n" + json.dumps({"client_context": context}, ensure_ascii=False)
    messages = [
        {"role": "system", "content": sys_txt},
        {"role": "user", "content": f"{user_prompt}\nReturn ONLY the JSON object."},
    ]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

def parse_target(obj: Dict[str, Any]) -> str:
    # Acepta response:str(JSON) o target:dict -> devuelve JSON string
    if "target" in obj and isinstance(obj["target"], dict):
        return json.dumps(obj["target"], ensure_ascii=False)
    if "response" in obj and isinstance(obj["response"], str):
        s = obj["response"].strip()
        # Quita fences tipo ```json ... ```
        s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.IGNORECASE | re.DOTALL).strip()
        # Si es lista de acciones, envolver en encabezado DSL mínimo
        if s.startswith('[') and s.endswith(']'):
            try:
                plan = json.loads(s)
                s = json.dumps(
                    {"version": "tad-dsl/0.2", "context": {"revit_version": "2025", "units": "SI"}, "plan": plan},
                    ensure_ascii=False
                )
            except Exception:
                pass
        return s
    raise ValueError("Cada línea debe contener 'response': str(JSON) o 'target': dict.")

def build_sample(tokenizer, prompt: str, context: Dict[str, Any] | None, target_json: str, eos: str) -> Dict[str, List[int]]:
    prompt_text = build_messages(tokenizer, prompt, context or {})
    target_text = target_json + eos
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"]
    input_ids = prompt_ids + target_ids
    labels    = [-100] * len(prompt_ids) + target_ids  # pérdida SOLO sobre la respuesta
    attn_mask = [1] * len(input_ids)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attn_mask}

class DataCollatorForSFT:
    def __init__(self, tokenizer, pad_to_multiple_of: int = 8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        keys = ["input_ids", "attention_mask", "labels"]
        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            rem = max_len % self.pad_to_multiple_of
            if rem != 0:
                max_len += (self.pad_to_multiple_of - rem)
        batch = {}
        for k in keys:
            padded = []
            for f in features:
                x = f[k]
                if len(x) < max_len:
                    pad_id = self.tokenizer.pad_token_id
                    pad_val = (-100 if k == "labels" else (pad_id if k == "input_ids" else 0))
                    x = x + [pad_val] * (max_len - len(x))
                padded.append(torch.tensor(x, dtype=torch.long))
            batch[k] = torch.stack(padded, dim=0)
        return batch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default=os.environ.get("MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.3"))
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--eval_jsonl", default="")
    ap.add_argument("--out", default="./out/mistral7b-mcp-lora")
    ap.add_argument("--seq_len", type=int, default=768)        # 7B en 16GB: 768 va fino
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--cuda_index", type=int, default=0, help="Índice de GPU a usar (0..N-1).")
    args = ap.parse_args()

    # Selección de GPU explícita
    assert torch.cuda.is_available(), "CUDA no disponible"
    torch.cuda.set_device(args.cuda_index)
    device = torch.device(f"cuda:{args.cuda_index}")
    print(f"[info] usando device {device} ->", torch.cuda.get_device_name(device))

    # Token de HF (no lo hardcodees)
    hf_token = os.environ.get("HUGGING_FACE_TOKEN") or os.environ.get("HF_TOKEN")

    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True, token=hf_token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Rendimiento seguro
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass

    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    mdl = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="eager",   # evita flash-attn en Windows
        token=hf_token,
        # sin device_map para no crear meta tensors
    )
    mdl.to(device)
    mdl.config.use_cache = False

    # LoRA (Mistral/LLaMA-style)
    from peft import LoraConfig, get_peft_model
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        bias="none", task_type="CAUSAL_LM",
    )
    mdl = get_peft_model(mdl, lora_cfg)
    mdl.print_trainable_parameters()

    # Dataset
    def load_jsonl(path):
        ds = load_dataset("json", data_files=path, split="train")
        def _proc(example):
            prompt  = example.get("prompt", "")
            context = example.get("context") or {}
            target_json = parse_target(example)
            sample = build_sample(tok, prompt, context, target_json, tok.eos_token)
            for k in ("input_ids","labels","attention_mask"):
                sample[k] = sample[k][:args.seq_len]
            return sample
        return ds.map(_proc, remove_columns=ds.column_names)

    train_ds = load_jsonl(args.train_jsonl)
    eval_ds  = load_jsonl(args.eval_jsonl) if args.eval_jsonl and os.path.exists(args.eval_jsonl) else None

    collator = DataCollatorForSFT(tok)

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        gradient_checkpointing=True,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        report_to="none",
        fp16=args.fp16,
        bf16=args.bf16,
    )

    # FIX: habilitar grad con checkpointing antes del Trainer
    if targs.gradient_checkpointing:
        mdl.gradient_checkpointing_enable()
        try:
            mdl.enable_input_require_grads()
        except AttributeError:
            for module in mdl.modules():
                if hasattr(module, "input_requires_grad"):
                    module.input_requires_grad = True
    mdl.train()

    trainer = Trainer(
        model=mdl,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )

    trainer.train()
    if eval_ds is not None:
        metrics = trainer.evaluate()
        print("[eval] metrics:", metrics)

    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"[ok] LoRA adapter saved to: {args.out}")

if __name__ == "__main__":
    main()
