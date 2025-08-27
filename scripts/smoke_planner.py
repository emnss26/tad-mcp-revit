from __future__ import annotations
import os, json, argparse, time, sys
from pathlib import Path
from typing import Dict, Any, List

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

import requests
from shared.tad_dsl.tool_definitions import ACTION_SCHEMAS

def is_required(field_info) -> bool:
    try:
        return field_info.is_required()  # pydantic v2
    except Exception:
        return getattr(field_info, "default", Ellipsis) is Ellipsis

def validate_plan(plan: Dict[str, Any]) -> List[str]:
    issues: List[str] = []
    if not isinstance(plan, dict):
        return ["Top-level debe ser objeto."]
    steps = plan.get("plan")
    if not isinstance(steps, list) or not steps:
        return ["Falta 'plan' o está vacío."]

    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            issues.append(f"Step {i}: debe ser objeto.")
            continue
        action = step.get("action")
        if action not in ACTION_SCHEMAS:
            issues.append(f"Step {i}: acción desconocida '{action}'.")
            continue
        args = step.get("args")
        if not isinstance(args, dict):
            issues.append(f"Step {i}: args debe ser objeto.")
            continue

        model = ACTION_SCHEMAS[action]
        fields = getattr(model, "model_fields", {})

        # requeridos
        for arg_name, finfo in fields.items():
            if is_required(finfo) and arg_name not in args:
                issues.append(f"Step {i}: falta arg requerido '{arg_name}' para '{action}'.")

        # desconocidos
        for k in args.keys():
            if k not in fields:
                issues.append(f"Step {i}: arg desconocido '{k}' en '{action}'.")
    return issues

def load_prompts(path: str | None) -> List[Dict[str, Any]]:
    if path and os.path.exists(path):
        lines = []
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    obj = json.loads(ln)
                    if "prompt" in obj:
                        lines.append(obj)
                except Exception:
                    pass
        return lines

    # fallback: prompts demo
    return [
        {"prompt": "Crea un nivel llamado L1 a 0.00 m"},
        {"prompt": "Crea un nivel L2 a 3.50 m"},
        {"prompt": "Crea un muro de 5m entre (0,0) y (5,0) en el nivel L1 tipo Basic Wall"},
        {"prompt": "Crea una vista de planta para el nivel L1"},
        {"prompt": "Coloca una puerta tipo Single-Flush en el muro más cercano al punto (2,0) en L1"},
    ]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=os.environ.get("API_BASE_URL", "http://127.0.0.1:8008"))
    ap.add_argument("--prompts", default="scripts/prompts_smoke.jsonl", help="JSONL con objetos {'prompt': ...}")
    ap.add_argument("--out", default="smoke_results.jsonl")
    args = ap.parse_args()

    prompts = load_prompts(args.prompts)
    if not prompts:
        print("No hay prompts; saliendo.")
        sys.exit(1)

    ok = 0
    total = 0
    t0 = time.time()

    with open(args.out, "w", encoding="utf-8") as fout:
        for p in prompts:
            total += 1
            body = {"prompt": p["prompt"], "context": p.get("context", {}), "dry_run": True}
            try:
                r = requests.post(f"{args.base_url}/plan", json=body, timeout=600)
                r.raise_for_status()
                data = r.json()
                plan = data.get("plan") if isinstance(data, dict) else None
                if not plan:
                    row = {"prompt": p["prompt"], "ok": False, "error": "Respuesta sin 'plan'."}
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                    print(f"[FAIL] {p['prompt']} → sin 'plan'")
                    continue

                issues = validate_plan(plan)
                row = {
                    "prompt": p["prompt"],
                    "ok": len(issues) == 0,
                    "issues": issues,
                    "plan": plan
                }
                if row["ok"]:
                    ok += 1
                    print(f"[OK]   {p['prompt']}")
                else:
                    print(f"[FAIL] {p['prompt']} ({len(issues)} issues)")
                    for it in issues:
                        print("  -", it)

                fout.write(json.dumps(row, ensure_ascii=False) + "\n")

            except Exception as e:
                row = {"prompt": p["prompt"], "ok": False, "error": str(e)}
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"[ERROR] {p['prompt']} → {e}")

    dt = time.time() - t0
    print(f"\nPasaron {ok}/{total} prompts. Tiempo: {dt:.1f}s. Resultados → {args.out}")

if __name__ == "__main__":
    main()