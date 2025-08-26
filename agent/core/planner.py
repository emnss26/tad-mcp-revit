from __future__ import annotations
import os, json, re
from typing import Any, Dict, List
import torch

# modelo por defecto (sirve en Windows con tu torch cu129)
MODEL_ID = os.environ.get("MODEL_ID", "microsoft/Phi-3-mini-4k-instruct")

_tokenizer = None
_model = None
_device = "cuda" if torch.cuda.is_available() else "cpu"

def ensure_loaded():
    global _tokenizer, _model
    if _model is not None:
        return
    from transformers import AutoTokenizer, AutoModelForCausalLM
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if _device == "cuda" else torch.float32,
        device_map="auto" if _device == "cuda" else None,
        trust_remote_code=True,
    ).to(_device)

def _available_actions() -> List[Dict[str, Any]]:
    # Leemos desde tu módulo compartido
    from shared.tad_dsl.tool_definitions import ACTION_SCHEMAS
    actions = []
    for name, model in ACTION_SCHEMAS.items():
        fields = getattr(model, "model_fields", {})
        actions.append({"action": name, "args": list(fields.keys())})
    return actions

_SYSTEM = (
"You are TAD Agent, an expert Autodesk Revit planner.\n"
"Return ONE JSON object only. No markdown, no code fences, no prose.\n"
"JSON schema: {\"version\":\"tad-dsl/0.2\",\"context\":{\"revit_version\":\"2025\",\"units\":\"SI\"},"
"\"plan\":[{\"action\":\"<action>\",\"args\":{...},\"as\":\"optional-alias\"}]}.\n"
"Use SI units consistently. If information is missing, add an initial 'get' step.\n"
"Never invent actions. Use only from the provided catalog.\n"
)

def _build_prompt(user_prompt: str, client_context: Dict[str, Any]) -> str:
    actions = _available_actions()
    catalog = {"available_actions": actions}
    ctx = {"client_context": client_context or {}}
    return (
        _SYSTEM +
        "\nAvailable actions:\n" + json.dumps(catalog, ensure_ascii=False) +
        "\nContext:\n" + json.dumps(ctx, ensure_ascii=False) +
        "\nUser request: \"" + user_prompt + "\"\n"
        "Return ONLY the JSON object."
    )

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

def _extract_json(text: str) -> Dict[str, Any]:
    m = _JSON_RE.search(text)
    if not m:
        raise ValueError("No JSON found in model output.")
    return json.loads(m.group(0))

def make_plan(user_prompt: str, client_context: Dict[str, Any] | None = None) -> Dict[str, Any]:
    ensure_loaded()
    prompt = _build_prompt(user_prompt, client_context or {})
    from transformers import TextIteratorStreamer
    inputs = _tokenizer(prompt, return_tensors="pt").to(_device)

    gen = _model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False,           # greedy para ser determinista
        temperature=0.0,
        eos_token_id=_tokenizer.eos_token_id,
    )

    out = _tokenizer.decode(gen[0], skip_special_tokens=True)
    # El modelo repite el prompt; nos quedamos con lo que viene después:
    completion = out[len(prompt):].strip()

    # A veces los modelos ponen ```json ... ```
    completion = completion.replace("```json", "").replace("```", "").strip()

    plan = _extract_json(completion)
    if not isinstance(plan, dict) or "plan" not in plan:
        raise ValueError("Invalid plan structure.")
    if not isinstance(plan["plan"], list) or not plan["plan"]:
        raise ValueError("Empty plan.")
    return plan