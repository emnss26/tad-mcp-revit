from __future__ import annotations
import os, sys, json, argparse
from pathlib import Path
from typing import Any, Dict, List

# bootstrap repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.tad_dsl.validators import normalize_plan, canonicalize_plan, validate_plan
from shared.tad_dsl.tool_definitions import ACTION_SCHEMAS

# --- mapeos de normalización (edita a tu gusto) ---
CATEGORY_MAP = {
    # ES -> EN
    "Muros": "Walls",
    "Ventanas": "Windows",
    "Puertas": "Doors",
    "Tuberías": "Pipes",
    "Ductos": "Ducts",
    "Topografía": "Topography",
}

DISCIPLINE_MAP = {
    "Arquitectura": "Architectural",
    "Estructuras": "Structural",
    "Mecánica": "Mechanical",
    "Eléctrica": "Electrical",
    "Plomería": "Plumbing",
}

DETAIL_LEVEL_MAP = {
    "Fino": "Fine",
    "Medio": "Medium",
    "Grueso": "Coarse",
}

def _normalize_inplace(obj: Any):
    # Recorre todo el dict y normaliza strings específicas
    if isinstance(obj, dict):
        for k, v in list(obj.items()):
            if k in ("category", "category_filter") and isinstance(v, str):
                obj[k] = CATEGORY_MAP.get(v, v)
            if k == "discipline" and isinstance(v, str):
                obj[k] = DISCIPLINE_MAP.get(v, v)
            if k == "level" and isinstance(v, str):
                # OJO: esto es el detail level en views.set_detail_level, no el "Nivel"
                obj[k] = DETAIL_LEVEL_MAP.get(v, v)
            _normalize_inplace(v)
    elif isinstance(obj, list):
        for x in obj:
            _normalize_inplace(x)

def _load_any(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read().strip()
        if not txt:
            return []
        if txt[0] == "[":
            return json.loads(txt)
        # JSONL
        out = []
        for ln in txt.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            out.append(json.loads(ln))
        return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Ruta a JSON (array) o JSONL (planes)")
    ap.add_argument("--report", default="dataset_report.json", help="Reporte de issues")
    ap.add_argument("--normalize", action="store_true", help="Aplicar normalización ES->EN")
    ap.add_argument("--write-fixed", default="", help="Si se indica, escribe dataset corregido aquí (JSON array)")
    args = ap.parse_args()

    data = _load_any(args.inp)
    fixed = []
    report = []
    ok_count = 0

    for i, item in enumerate(data):
        try:
            plan = normalize_plan(item)
            if args.normalize:
                _normalize_inplace(plan)
                plan = canonicalize_plan(plan)
            is_ok, errs = validate_plan(plan)
            fixed.append(plan)
            report.append({"index": i, "ok": is_ok, "errors": errs})
            if is_ok:
                ok_count += 1
        except Exception as e:
            report.append({"index": i, "ok": False, "errors": [str(e)]})

    with open(args.report, "w", encoding="utf-8") as fout:
        json.dump({"total": len(data), "ok": ok_count, "items": report}, fout, ensure_ascii=False, indent=2)

    if args.write_fixed:
        with open(args.write_fixed, "w", encoding="utf-8") as fout:
            json.dump(fixed, fout, ensure_ascii=False, indent=2)

    print(f"[validate_dataset] OK {ok_count}/{len(data)}. Reporte → {args.report}")
    if args.write_fixed:
        print(f"[validate_dataset] Dataset corregido → {args.write_fixed}")

if __name__ == "__main__":
    main()