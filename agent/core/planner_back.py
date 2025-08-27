from __future__ import annotations
import os, sys, json, re
from pathlib import Path
from typing import Any, Dict, List, Tuple
import difflib

import torch
import psutil
from pydantic import ValidationError

# ───────────────── env / paths ─────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except Exception:
    pass

# Evitar fragmentación (si tu build lo soporta; PyTorch avisa si está deprecado)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Usamos los schemas REALES para conocer campos válidos
from shared.tad_dsl.tool_definitions import ACTION_SCHEMAS  # dict[str, PydanticModel]

MODEL_ID = os.environ.get("MODEL_ID", "microsoft/Phi-3-mini-4k-instruct")
MAX_NEW  = int(os.environ.get("TAD_MAX_NEW", "96"))
DEBUG    = os.environ.get("TAD_DEBUG", "") == "1"

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
            # 8GB → 4 GiB (conservador). >8GB → ~80% - 1 GiB (mín 6).
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
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    if torch.cuda.is_available() and torch.cuda.device_count() >= 1:
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
            max_memory=_build_max_memory(),
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

def _action_names() -> List[str]:
    filters = {s.strip() for s in os.environ.get("TAD_ACTIONS_FILTER", "").split(",") if s.strip()}
    names = list(ACTION_SCHEMAS.keys())
    if filters:
        names = [n for n in names if n in filters]
    return sorted(names)

# Export público (útil para otros módulos)
def get_available_actions() -> List[Dict[str, Any]]:
    return _available_actions()

_SYSTEM = (
    "You are TAD Agent, an expert Autodesk Revit planner.\n"
    "Return ONE JSON object only. No markdown, no code fences, no prose.\n"
    "JSON schema:{\"version\":\"tad-dsl/0.2\",\"context\":{\"revit_version\":\"2025\",\"units\":\"SI\"},"
    "\"plan\":[{\"action\":\"<action>\",\"args\":{...},\"as\":\"optional-alias\"}]}.\n"
    "Use SI units consistently. If information is missing, add an initial 'get' step.\n"
    "Use only action names from the catalog. Each step MUST have `action` and `args` (object).\n"
    "Your output MUST be STRICT JSON (double quotes, true/false/null, no trailing commas).\n"
    "Never output placeholders like \"<action>\" or empty args. Begin with '{' and end with '}'.\n"
)

def _build_prompt(user_prompt: str, client_context: Dict[str, Any]) -> str:
    messages = [
        {"role": "system", "content": _SYSTEM +
            "\nAvailable actions:\n" + json.dumps({"available_actions": _available_actions()}, ensure_ascii=False) +
            "\nContext:\n" + json.dumps({"client_context": client_context or {}}, ensure_ascii=False)
        },
        {"role": "user", "content": f'{user_prompt}\nReturn ONLY the JSON object.'}
    ]
    return _tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

# ───────────── extracción / normalización ──────
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

def _find_balanced_json(s: str) -> str | None:
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(s)):
        c = s[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[start:i+1]
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
                obj["plan"] = obj.pop(k)
                break
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

# Export público (útil para /validate)
def normalize_plan(obj: Any) -> Dict[str, Any]:
    return _normalize_plan(obj)

# ───────────── helpers de intención/parseo ─────
_SP_FLOAT = re.compile(r"(-?\d+(?:[.,]\d+)?)")

def _to_float_m(s: str) -> float:
    s = s.strip().lower().replace(",", ".")
    return float(s)

def _intent_from_prompt(prompt: str) -> str:
    p = prompt.lower()
    if any(w in p for w in ("puerta", "door")):
        return "door"
    if any(w in p for w in ("muro", "wall")):
        return "wall"
    if any(w in p for w in ("vista de planta", "plan view", "planta")):
        return "view_plan"
    if "nivel" in p or "level" in p:
        return "level"
    return "unknown"

def _choose_action(intent: str, names: List[str]) -> str | None:
    # Preferencias exactas (si existen en el catálogo)
    prefer_exact = {
        "level":     ["levels.create"],
        "wall":      ["walls.create_linear"],
        "view_plan": ["views.create_plan"],
        "door":      ["doors.place_in_wall"],
    }
    for p in prefer_exact.get(intent, []):
        if p in names:
            return p

    # Heurística por substring si no hay exacta
    keys = {
        "level":     ["level", "levels.create"],
        "wall":      ["wall", "walls.create"],
        "view_plan": ["views.create_plan", "plan"],
        "door":      ["door", "doors.place"],
    }.get(intent, [])
    for k in keys:
        for n in names:
            if k in n.lower():
                return n

    # Fondo de barril
    return names[0] if names else None

def _fill_level_args(prompt: str, action: str) -> Dict[str, Any]:
    # nombre explícito "llamado ..." o genérico "nivel X ..."
    name = None
    m_name = re.search(r"nivel\s+(?:llamado|con\s+nombre)\s+([A-Za-z0-9._-]+)", prompt, flags=re.I)
    if not m_name:
        m_name = re.search(r"\bnivel\s+([A-Za-z0-9._-]+)\b", prompt, flags=re.I)
    if m_name:
        token = m_name.group(1)
        if token.lower() not in ("a", "en"):  # evita capturar preps
            name = token

    # elevación: "a 3.50 m", etc.
    m_elev = re.search(r"a\s+(" + _SP_FLOAT.pattern + r")\s*m", prompt, flags=re.I)
    elev = _to_float_m(m_elev.group(1)) if m_elev else 0.0

    fields = list(getattr(ACTION_SCHEMAS[action], "model_fields", {}).keys())
    out: Dict[str, Any] = {}
    # nombre
    for cand in ("name", "level_name", "id", "label"):
        if cand in fields and name is not None:
            out[cand] = name
            break
    # elevación
    for cand in ("elevation_m", "elevation", "height_m", "z_m"):
        if cand in fields:
            out[cand] = elev
            break
    return out

def _fill_view_plan_args(prompt: str, action: str) -> Dict[str, Any]:
    fields = list(getattr(ACTION_SCHEMAS[action], "model_fields", {}).keys())
    out: Dict[str, Any] = {}
    if "level_name" in fields:
        m = re.search(r"nivel\s+([A-Za-z0-9._-]+)", prompt, flags=re.I)
        if m:
            out["level_name"] = m.group(1)
    return out

# ───────────── canonicalización de args ────────
_SYNONYMS: Dict[str, Dict[str, List[str]]] = {
    "levels.create": {
        "level_name":  ["name", "label"],
        "name":        ["level_name", "label"],
        "elevation_m": ["elevation", "z_m", "height_m"],
    },
    # "walls.create_linear": { ... }  # ejemplo futuro si lo necesitas
}

def _canonicalize_step(step: Dict[str, Any]) -> Dict[str, Any]:
    action = step.get("action")
    if action not in ACTION_SCHEMAS:
        if step.get("as") == "optional-alias":
            step.pop("as", None)
        return step

    schema = ACTION_SCHEMAS[action]
    fields = list(getattr(schema, "model_fields", {}).keys())

    args_in = step.get("args", {}) or {}
    if not isinstance(args_in, dict):
        try:
            args_in = dict(args_in)
        except Exception:
            args_in = {}

    # 1) aplica sinónimos
    syn = _SYNONYMS.get(action, {})
    mapped: Dict[str, Any] = {}
    used_keys = set()
    for dest, alts in syn.items():
        if dest in args_in:
            mapped[dest] = args_in[dest]; used_keys.add(dest)
            continue
        for k in alts:
            if k in args_in:
                mapped[dest] = args_in[k]; used_keys.add(k)
                break

    # 2) copia campos ya correctos
    for k, v in args_in.items():
        if k in fields:
            mapped.setdefault(k, v)
            used_keys.add(k)

    # 3) filtra a solo campos válidos del schema
    cleaned = {k: v for k, v in mapped.items() if k in fields}

    # 4) limpia alias placeholder
    if step.get("as") == "optional-alias":
        step.pop("as", None)

    step["args"] = cleaned
    return step

def _canonicalize_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    steps = plan.get("plan", [])
    if isinstance(steps, list):
        plan["plan"] = [_canonicalize_step(s) if isinstance(s, dict) else s for s in steps]
    return plan

# export público
def canonicalize_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    return _canonicalize_plan(plan)

# ───────────── reparador (elige acción/args) ───
def _repair_plan(plan: Dict[str, Any], prompt: str) -> Dict[str, Any]:
    available = _action_names()
    changed = False
    intent = _intent_from_prompt(prompt)

    expected = _choose_action(intent, available) if intent != "unknown" else None

    for step in plan.get("plan", []):
        # 0) si la acción actual no concuerda con la intención, re-encaminar
        current = (step.get("action") or "").lower()
        if intent != "unknown" and expected:
            ok = (
                (intent == "level"     and "level" in current) or
                (intent == "wall"      and "wall"  in current) or
                (intent == "view_plan" and "views.create_plan" in current) or
                (intent == "door"      and "door"  in current)
            )
            if not ok or step.get("action") not in available:
                step["action"] = expected
                changed = True

        # 1) si acción vacía o inválida, intenta escoger
        action = step.get("action", "")
        if not action or action == "<action>" or action not in available:
            best = _choose_action(intent, available)
            if best:
                step["action"] = best
                changed = True

        # 2) autocompletado MINIMAL para casos conocidos (solo si args vienen vacíos)
        args = step.get("args", {})
        if not isinstance(args, dict):
            args = {}

        if not args:
            if intent == "level" and step["action"] in available:
                guessed = _fill_level_args(prompt, step["action"])
                if guessed:
                    args.update(guessed)
                    changed = True
            elif intent == "view_plan" and step["action"] in available:
                guessed = _fill_view_plan_args(prompt, step["action"])
                if guessed:
                    args.update(guessed)
                    changed = True

        step["args"] = args

    if changed and DEBUG:
        print("[planner][REPAIRED PLAN] ", json.dumps(plan, ensure_ascii=False))

    return plan

# export público
def repair_plan(plan: Dict[str, Any], prompt: str) -> Dict[str, Any]:
    return _repair_plan(plan, prompt)

# ───────────── validator (público) ─────────────
def validate_plan(plan: Dict[str, Any]) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Valida cada paso con su Pydantic model en ACTION_SCHEMAS.
    Devuelve (ok, errores). Cada error: {"index": i, "action": action, "errors": [...]}.
    """
    errors: List[Dict[str, Any]] = []
    steps = plan.get("plan", [])
    if not isinstance(steps, list):
        return False, [{"index": None, "action": None, "errors": ["'plan' must be a list."]}]
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            errors.append({"index": i, "action": None, "errors": ["step must be an object"]})
            continue
        action = step.get("action")
        args = step.get("args", {}) or {}
        if not action or action not in ACTION_SCHEMAS:
            errors.append({"index": i, "action": action, "errors": [f"Unknown or missing action: {action!r}"]})
            continue
        Model = ACTION_SCHEMAS[action]
        try:
            Model(**args)  # valida
        except ValidationError as ve:
            errors.append({"index": i, "action": action, "errors": json.loads(ve.json())})
    return (len(errors) == 0), errors

# ───────────── device de inputs ────────────────
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
            do_sample=False,
            temperature=0.0,
            use_cache=False,
            eos_token_id=_tokenizer.eos_token_id,
            pad_token_id=pad_id,
        )

    out = _tokenizer.decode(gen[0], skip_special_tokens=True)
    # Intenta quedarte con la última “salida” del asistente en chat_template
    completion = out[out.rfind("assistant") + len("assistant"):].strip() if "assistant" in out else out
    completion = completion.replace("```json", "").replace("```", "").strip()

    if DEBUG:
        print("\n[planner][RAW COMPLETION]\n", completion[:1000], "\n")

    try:
        raw = _extract_json(completion)
    except Exception:
        raw = _extract_json(out)

    plan = _normalize_plan(raw)
    plan = _repair_plan(plan, user_prompt)     # corrige acción y args mínimos (solo level/view_plan)
    plan = _canonicalize_plan(plan)            # aplica sinónimos + filtra a schema
    return plan
