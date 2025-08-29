from __future__ import annotations
import os, sys, json, re
from pathlib import Path
from typing import Any, Dict, List

import torch
import psutil

# ───────────────── env / paths ─────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except Exception:
    pass

# Evitar fragmentación (algunas builds lo ignoran)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Schemas reales (acciones)
from shared.tad_dsl.tool_definitions import ACTION_SCHEMAS

# ───────────────── config ─────────────────
MODEL_ID    = os.environ.get("MODEL_ID", "microsoft/Phi-3-mini-4k-instruct")
ADAPTER_DIR = os.environ.get("ADAPTER_DIR", "").strip()  # p.ej.: out/phi3-mcp-lora
MAX_NEW     = int(os.environ.get("TAD_MAX_NEW", "96"))
DEBUG       = os.environ.get("TAD_DEBUG", "") == "1"

_tokenizer = None
_model = None

# ───────────── memoria por dispositivo ─────────
def _build_max_memory() -> Dict[Any, str]:
    mm: Dict[Any, str] = {}
    if not torch.cuda.is_available():
        return mm
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total_gib = props.total_memory // (1024**3)
        env_key = f"TAD_GPU{i}_GIB"
        if env_key in os.environ:
            budget = int(os.environ[env_key])
        else:
            budget = 4 if total_gib <= 8 else max(6, int(total_gib * 0.80) - 1)
        mm[i] = f"{budget}GiB"
    cpu_free_gib = int(psutil.virtual_memory().available / (1024**3)) - 2
    if cpu_free_gib > 2:
        mm["cpu"] = f"{cpu_free_gib}GiB"
    return mm

# ───────────── cargar modelo/tokenizer ─────────
def ensure_loaded():
    global _tokenizer, _model
    if _model is not None:
        return

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    base_id = MODEL_ID
    offload_dir = REPO_ROOT / "offload"
    offload_dir.mkdir(parents=True, exist_ok=True)

    _tokenizer = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    kwargs = dict(
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    if torch.cuda.is_available():
        kwargs.update(
            device_map="auto",
            max_memory=_build_max_memory(),
            offload_folder=str(offload_dir),
        )

    base = AutoModelForCausalLM.from_pretrained(base_id, **kwargs)

    # Aplica LoRA si existe
    if ADAPTER_DIR and os.path.isdir(ADAPTER_DIR):
        try:
            base = PeftModel.from_pretrained(base, ADAPTER_DIR)
            print("[planner] adapter:", ADAPTER_DIR)
        except Exception as e:
            print("[planner] WARNING no pude cargar adapter:", e)

    base.eval()
    _model = base

    # Usa el chat_template del adapter si existe
    try:
        ct = Path(ADAPTER_DIR, "chat_template.jinja")
        if ct.is_file():
            _tokenizer.chat_template = ct.read_text(encoding="utf-8")
    except Exception:
        pass

    devmap = getattr(_model, "hf_device_map", None)
    if devmap:
        print("[planner] hf_device_map:", devmap)

    # Warm-up (corto) para evitar la 1ª llamada lenta
    try:
        prompt = _build_prompt("ping", {})
        ii = _tokenizer(prompt, return_tensors="pt")
        tgt = _inputs_device_for(_model)
        ii = {k: v.to(tgt) for k, v in ii.items()}
        with torch.inference_mode():
            _ = _model.generate(
                **ii,
                max_new_tokens=4,
                do_sample=True, temperature=0.25, top_p=0.9, repetition_penalty=1.05,
                use_cache=False,
                eos_token_id=_tokenizer.eos_token_id,
                pad_token_id=_tokenizer.pad_token_id
            )
    except Exception:
        pass

# Expuesto para /health
def model_info() -> Dict[str, Any]:
    ensure_loaded()
    return {
        "model_base": MODEL_ID,
        "adapter_dir": ADAPTER_DIR or None,
        "device_map": getattr(_model, "hf_device_map", None),
        "max_new_tokens": MAX_NEW,
    }

# ───────────── catálogo / prompt ───────────────
def _available_actions() -> List[Dict[str, Any]]:
    filters = {s.strip() for s in os.environ.get("TAD_ACTIONS_FILTER", "").split(",") if s.strip()}
    acts: List[Dict[str, Any]] = []
    for name, model in ACTION_SCHEMAS.items():
        if filters and name not in filters:
            continue
        fields = getattr(model, "model_fields", {})
        acts.append({"action": name, "args": list(fields.keys())})
    return acts

def _action_names_only() -> List[str]:
    filters = {s.strip() for s in os.environ.get("TAD_ACTIONS_FILTER", "").split(",") if s.strip()}
    names = list(ACTION_SCHEMAS.keys())
    if filters:
        names = [n for n in names if n in filters]
    return sorted(names)

_SYSTEM = (
    "You are TAD Agent, an expert Autodesk Revit planner.\n"
    "Return ONE JSON object only. No markdown, no code fences, no prose.\n"
    "JSON schema:{\"version\":\"tad-dsl/0.2\",\"context\":{\"revit_version\":\"2025\",\"units\":\"SI\"},"
    "\"plan\":[{\"action\":\"<action>\",\"args\":{...}}]}.\n"
    "\n"
    "CRITICAL RULES:\n"
    "1) Choose an action whose NAME directly matches the user's MAIN intent.\n"
    "   Examples of mapping:\n"
    "   - 'muro', 'wall' -> use a 'walls.*' action (typically walls.create_linear)\n"
    "   - 'nivel', 'level' -> use 'levels.create'\n"
    "   - 'vista de plano', 'plan view' -> use 'views.create_plan'\n"
    "2) NEVER create or rename levels unless the prompt EXPLICITLY asks for a level.\n"
    "3) Each step MUST have `action` and `args` (object). No placeholders like '<action>'.\n"
    "4) Use ONLY action names from the provided catalog. Use SI units.\n"
    "5) If the user provides coordinates or dimensions, pass them as numeric args in meters.\n"
    "\n"
    "Mini-examples (follow these exactly):\n"
    "USER: 'Dibuja un muro de 5 m en el nivel N1, tipo \"Muro Genérico 200\" de (0,0) a (5,0).'\n"
    "OUTPUT: {\"version\":\"tad-dsl/0.2\",\"context\":{\"revit_version\":\"2025\",\"units\":\"SI\"},\"plan\":[\n"
    "  {\"action\":\"walls.create_linear\",\"args\":{\"start\":[0,0,0],\"end\":[5,0,0],\"level\":\"N1\",\"type\":\"Muro Genérico 200\",\"height\":3.0}}\n"
    "]}\n"
    "\n"
    "USER: 'Crea un nivel llamado N1 a 3.50 m.'\n"
    "OUTPUT: {\"version\":\"tad-dsl/0.2\",\"context\":{\"revit_version\":\"2025\",\"units\":\"SI\"},\"plan\":[\n"
    "  {\"action\":\"levels.create\",\"args\":{\"name\":\"N1\",\"elevation\":3.5}}\n"
    "]}\n"
    "\n"
    "Return ONLY the JSON object."
)

def _build_prompt(user_prompt: str, client_context: Dict[str, Any]) -> str:
    names_only = _action_names_only()
    messages = [
        {"role": "system", "content":
            _SYSTEM
            + "\nAllowed action names:\n"
            + json.dumps({"allowed_actions": names_only}, ensure_ascii=False)
            + "\nAvailable actions (with args):\n"
            + json.dumps({"available_actions": _available_actions()}, ensure_ascii=False)
            + "\nContext:\n"
            + json.dumps({"client_context": client_context or {}}, ensure_ascii=False)
        },
        {"role": "user", "content": f"{user_prompt}\nReturn ONLY the JSON object."}
    ]
    return _tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

# ───────────── extracción / normalización ──────
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

def _find_balanced_json(s: str) -> str | None:
    start = s.find("{")
    if start == -1:
        return None
    depth, i = 0, start
    while i < len(s):
        c = s[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[start:i+1]
        i += 1
    return None

def _json_coerce(s: str) -> Dict[str, Any]:
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    s2 = re.sub(r",\s*([}\]])", r"\1", s)
    s2 = re.sub(r"(?<!\\)'", '"', s2)
    s2 = s2.replace("None", "null").replace("True", "true").replace("False", "false")
    try:
        return json.loads(s2)
    except Exception:
        pass
    try:
        import ast
        s3 = s.replace("null", "None").replace("true", "True").replace("false", "False")
        obj = ast.literal_eval(s3)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    raise ValueError("No JSON found in model output.")

def _extract_json(text: str) -> Dict[str, Any]:
    jb = _find_balanced_json(text)
    if jb is not None:
        return _json_coerce(jb)
    m = _JSON_RE.search(text)
    if m:
        return _json_coerce(m.group(0))
    raise ValueError("No JSON found in model output.")

def _normalize_plan(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, list):
        obj = {"plan": obj}
    if not isinstance(obj, dict):
        raise ValueError("Plan must be a JSON object.")

    if "plan" not in obj:
        for k in ("steps", "actions", "tasks", "procedure", "commands", "sequence"):
            if k in obj:
                obj["plan"] = obj.pop(k); break
    if "plan" in obj and isinstance(obj["plan"], dict):
        obj["plan"] = [obj["plan"]]
    if "plan" not in obj or not isinstance(obj["plan"], list):
        raise ValueError("Missing 'plan' list.")

    obj.setdefault("version", "tad-dsl/0.2")
    ctx = obj.get("context")
    if not isinstance(ctx, dict):
        ctx = {}
    ctx.setdefault("revit_version", "2025")
    ctx.setdefault("units", "SI")
    obj["context"] = ctx

    cleaned: List[Dict[str, Any]] = []
    for step in obj["plan"]:
        if not isinstance(step, dict):
            continue
        action = step.get("action") or step.get("tool") or step.get("name")
        args = step.get("args") or step.get("parameters") or step.get("params") or {}
        alias = step.get("as") or step.get("id") or step.get("alias")
        if not isinstance(args, dict):
            try:
                args = dict(args)
            except Exception:
                args = {}
        if action:
            item = {"action": str(action), "args": args}
            if alias:
                item["as"] = str(alias)
            cleaned.append(item)
    if not cleaned:
        raise ValueError("Empty plan.")
    obj["plan"] = cleaned
    return obj

# ───────────── validación ligera (sin “parches”) ─────────
def _validate_against_schema(plan: Dict[str, Any]) -> Dict[str, Any]:
    steps = []
    for step in plan.get("plan", []):
        if not isinstance(step, dict):
            continue
        action = step.get("action", "")
        if not action:
            continue
        args = step.get("args") or {}
        if action in ACTION_SCHEMAS and isinstance(args, dict):
            fields = set(getattr(ACTION_SCHEMAS[action], "model_fields", {}).keys())
            args = {k: v for k, v in args.items() if k in fields}
        step["args"] = args if isinstance(args, dict) else {}
        steps.append(step)
    if not steps:
        raise ValueError("Empty/invalid plan after validation.")
    plan["plan"] = steps
    return plan

# ───────────── device de inputs ────────────────
def _inputs_device_for(model) -> torch.device:
    devmap = getattr(model, "hf_device_map", None)
    if devmap:
        key = "model.embed_tokens"
        if key in devmap:
            tgt = devmap[key]
            return torch.device(f"cuda:{tgt}") if isinstance(tgt, int) else torch.device(tgt)
        gpu_ids = [v for v in devmap.values() if isinstance(v, int)]
        if gpu_ids:
            return torch.device(f"cuda:{min(gpu_ids)}")
        for v in devmap.values():
            if isinstance(v, str) and v.startswith("cuda"):
                return torch.device(v)
        return torch.device("cpu")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ───────────── planificación ───────────────────
def make_plan(user_prompt: str, client_context: Dict[str, Any] | None = None) -> Dict[str, Any]:
    ensure_loaded()
    prompt = _build_prompt(user_prompt, client_context or {})

    inputs = _tokenizer(prompt, return_tensors="pt")
    target = _inputs_device_for(_model)
    inputs = {k: v.to(target) for k, v in inputs.items()}

    pad_id = _tokenizer.pad_token_id or _tokenizer.eos_token_id

    with torch.inference_mode():
        gen = _model.generate(
            **inputs,
            max_new_tokens=MAX_NEW,
            do_sample=True,                 # <- sampling suave para no “copiar” el ejemplo
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.05,
            use_cache=False,
            eos_token_id=_tokenizer.eos_token_id,
            pad_token_id=pad_id,
        )

    out = _tokenizer.decode(gen[0], skip_special_tokens=True)
    completion = out[out.rfind("assistant") + len("assistant"):].strip() if "assistant" in out else out
    completion = completion.replace("```json", "").replace("```", "").strip()

    if DEBUG:
        print("\n[planner][RAW COMPLETION]\n", completion[:1000], "\n")

    try:
        raw = _extract_json(completion)
    except Exception:
        raw = _extract_json(out)

    plan = _normalize_plan(raw)
    plan = _validate_against_schema(plan)  # sin “parches”
    return plan
