# scripts/check_tools.py
from __future__ import annotations
import sys, json
from pathlib import Path

# ── bootstrap de path al repo root ─────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]  # .../TAD-Revit-Agent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Carga .env si existe (opcional)
try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except Exception:
    pass

# ── importa schemas reales ─────────────────────────────────────
from shared.tad_dsl.tool_definitions import ACTION_SCHEMAS

def build_catalog():
    out = []
    for name, model in ACTION_SCHEMAS.items():
        fields = getattr(model, "model_fields", {})  # pydantic v2
        req, opt = [], []
        for arg_name, finfo in fields.items():
            # pydantic v2: FieldInfo.is_required(); fallback por si acaso
            try:
                is_req = finfo.is_required()
            except Exception:
                is_req = getattr(finfo, "default", Ellipsis) is Ellipsis
            (req if is_req else opt).append(arg_name)

        out.append({
            "action": name,
            "args": {"required": sorted(req), "optional": sorted(opt)}
        })

    out.sort(key=lambda x: x["action"])
    return {"ok": True, "count": len(out), "actions": out}

if __name__ == "__main__":
    print(json.dumps(build_catalog(), indent=2, ensure_ascii=False))
