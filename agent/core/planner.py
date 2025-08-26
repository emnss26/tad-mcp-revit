from __future__ import annotations
import os, sys, json, re
from pathlib import Path
from typing import Any, Dict, List

import torch
import psutil

# ───────────────────────────────────────────────────────────────
# Carga .env si está disponible (para que TAD_GPU*_GIB y otros apliquen)
# ───────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
try:
    from dotenv import load_dotenv  # pip install python-dotenv
    load_dotenv(REPO_ROOT / ".env")
except Exception:
    pass

# ───────────────────────────────────────────────────────────────
# Evitar fragmentación del allocator CUDA (si tu build lo soporta)
# ───────────────────────────────────────────────────────────────
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")

# ───────────────────────────────────────────────────────────────
# Pathing para importar shared/… sin problemas
# ───────────────────────────────────────────────────────────────
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ───────────────────────────────────────────────────────────────
# Modelo
# ───────────────────────────────────────────────────────────────
MODEL_ID = os.environ.get("MODEL_ID", "microsoft/Phi-3-mini-4k-instruct")
MAX_NEW = int(os.environ.get("TAD_MAX_NEW", "96"))  # menos KV-cache que 160

_tokenizer = None
_model = None

def _build_max_memory() -> Dict[Any, str]:
    """Arma límites por dispositivo favoreciendo GPUs con más VRAM."""
    max_memory: Dict[Any, str] = {}
    if not torch.cuda.is_available():
        return max_memory

    num_gpus = torch.cuda.device_count()
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        total_gib = props.total_memory // (1024**3)

        # Si hay override explícito, úsalo.
        env_key = f"TAD_GPU{i}_GIB"
        if env_key in os.environ:
            budget_gib = int(os.environ[env_key])
        else:
            # Heurística: en GPUs de 8 GB, 4–5 GB; en >8 GB, ~80% - 1 GB.
            if total_gib <= 8:
                budget_gib = 4
            else:
                budget_gib = max(6, int(total_gib * 0.80) - 1)
        max_memory[i] = f"{budget_gib}GiB"

    # Offload a CPU si hay RAM libre
    cpu_free_gib = int(psutil.virtual_memory().available / (1024**3)) - 2
    if cpu_free_gib > 2:
        max_memory["cpu"] = f"{cpu_free_gib}GiB"

    return max_memory

# ───────────────────────────────────────────────────────────────
# Carga del modelo (multi-GPU con device_map="auto")
# ───────────────────────────────────────────────────────────────
def ensure_loaded():
    """Carga perezosa del modelo/tokenizer, con sharding en múltiples GPUs si hay VRAM suficiente."""
    global _tokenizer, _model
    if _model is not None:
        return

    from transformers import AutoTokenizer, AutoModelForCausalLM

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    has_cuda = torch.cuda.is_available()
    if has_cuda and torch.cuda.device_count() >= 1:
        max_memory = _build_max_memory()

        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
            max_memory=max_memory,
            offload_folder=str(REPO_ROOT / "offload"),
            attn_implementation="eager",
        )
    else:
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )

    _model.eval()
    devmap = getattr(_model, "hf_device_map", None)
    if devmap:
        print("[planner] hf_device_map:", devmap)

# ───────────────────────────────────────────────────────────────
# Catálogo + Prompting
# ───────────────────────────────────────────────────────────────
def _available_actions() -> List[Dict[str, Any]]:
    from shared.tad_dsl.tool_definitions import ACTION_SCHEMAS
    actions: List[Dict[str, Any]] = []
    for name, model in ACTION_SCHEMAS.items():
        fields = getattr(model, "model_fields", {})
        actions.append({"action": name, "args": list(fields.keys())})
    return actions

_SYSTEM = (
    "You are TAD Agent, an expert Autodesk Revit planner.\n"
    "Return ONE JSON object only. No markdown, no code fences, no prose.\n"
    "JSON schema:{\"version\":\"tad-dsl/0.2\",\"context\":{\"revit_version\":\"2025\",\"units\":\"SI\"},"
    "\"plan\":[{\"action\":\"<action>\",\"args\":{...},\"as\":\"optional-alias\"}]}.\n"
    "Use SI units consistently. If information is missing, add an initial 'get' step.\n"
    "Never invent actions. Use only from the provided catalog.\n"
)

def _build_prompt(user_prompt: str, client_context: Dict[str, Any]) -> str:
    catalog = {"available_actions": _available_actions()}
    ctx = {"client_context": client_context or {}}
    return (
        _SYSTEM
        + "\nAvailable actions:\n"
        + json.dumps(catalog, ensure_ascii=False)
        + "\nContext:\n"
        + json.dumps(ctx, ensure_ascii=False)
        + f'\nUser request: "{user_prompt}"\n'
        + "Return ONLY the JSON object."
    )

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

def _extract_json(text: str) -> Dict[str, Any]:
    m = _JSON_RE.search(text)
    if not m:
        raise ValueError("No JSON found in model output.")
    return json.loads(m.group(0))

def _inputs_device_for(model) -> torch.device:
    devmap = getattr(model, "hf_device_map", None)
    if devmap:
        key = "model.embed_tokens"
        if key in devmap:
            tgt = devmap[key]
            if isinstance(tgt, int):
                return torch.device(f"cuda:{tgt}")
            if isinstance(tgt, str):
                return torch.device(tgt)
        gpu_ids = [v for v in devmap.values() if isinstance(v, int)]
        if gpu_ids:
            return torch.device(f"cuda:{min(gpu_ids)}")
        for v in devmap.values():
            if isinstance(v, str) and v.startswith("cuda"):
                return torch.device(v)
        return torch.device("cpu")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ───────────────────────────────────────────────────────────────
# Planificación
# ───────────────────────────────────────────────────────────────
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
            max_new_tokens=MAX_NEW,  # <= 96 por defecto
            do_sample=False,
            temperature=0.0,
            use_cache=False,
            eos_token_id=_tokenizer.eos_token_id,
            pad_token_id=pad_id,
        )

    out = _tokenizer.decode(gen[0], skip_special_tokens=True)

    def _clean(t: str) -> str:
        return t.replace("```json", "").replace("```", "").strip()

    try:
        plan = _extract_json(_clean(out))
    except Exception:
        completion = _clean(out[len(prompt):].strip())
        plan = _extract_json(completion)

    if not isinstance(plan, dict) or "plan" not in plan:
        raise ValueError("Invalid plan structure.")
    if not isinstance(plan["plan"], list) or not plan["plan"]:
        raise ValueError("Empty plan.")
    return plan