from __future__ import annotations
import os, sys, json, argparse, math
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.core.planner import normalize_plan, canonicalize_plan

def _load_any(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read().strip()
        if not txt:
            return []
        # Soporta JSON (array) o JSONL
        if txt[0] == "[":
            return json.loads(txt)
        out = []
        for ln in txt.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            out.append(json.loads(ln))
        return out

def _dist_m(a: Dict[str, float], b: Dict[str, float]) -> float:
    ax, ay, az = a.get("x", 0.0) or 0.0, a.get("y", 0.0) or 0.0, a.get("z", 0.0) or 0.0
    bx, by, bz = b.get("x", 0.0) or 0.0, b.get("y", 0.0) or 0.0, b.get("z", 0.0) or 0.0
    dx, dy, dz = (ax - bx), (ay - by), (az - bz)
    return float(math.sqrt(dx * dx + dy * dy + dz * dz))

def _fmt_num(x: Any, nd: int = 2) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)

def _prompt_from_step(step: Dict[str, Any]) -> str:
    act = (step.get("action") or "").lower()
    args = step.get("args", {}) or {}

    if act == "levels.create":
        nm = args.get("name") or args.get("level_name")
        elev = args.get("elevation") or args.get("elevation_m")
        if nm is not None and elev is not None:
            return f"Crea un nivel llamado {nm} a {_fmt_num(elev)} m"

    if act == "walls.create_linear":
        lvl = args.get("level"); typ = args.get("type")
        start = args.get("start"); end = args.get("end")
        h = args.get("height")
        if start and end and lvl and typ and h is not None:
            L = _dist_m(start, end)
            return (f"Crea un muro lineal tipo \"{typ}\" entre "
                    f"({start.get('x',0)},{start.get('y',0)},{start.get('z',0)}) y "
                    f"({end.get('x',0)},{end.get('y',0)},{end.get('z',0)}) "
                    f"en {lvl} con altura {_fmt_num(h)} m (longitud {_fmt_num(L)} m)")

    if act == "floors.create_rectangular":
        lvl = args.get("level"); typ = args.get("type")
        o = args.get("origin"); w = args.get("width"); d = args.get("depth")
        if lvl and typ and o and w is not None and d is not None:
            return (f"Crea un piso rectangular tipo \"{typ}\" en {lvl}, "
                    f"origen ({o.get('x',0)},{o.get('y',0)},{o.get('z',0)}), "
                    f"ancho {_fmt_num(w)} m, profundidad {_fmt_num(d)} m")

    if act == "views.create_section":
        s = args.get("start"); e = args.get("end"); name = args.get("name","")
        if s and e:
            base = (f"Crea una sección desde ({s.get('x',0)},{s.get('y',0)},{s.get('z',0)}) "
                    f"hasta ({e.get('x',0)},{e.get('y',0)},{e.get('z',0)})")
            return base + (f" llamada {name}" if name else "")

    if act == "schedules.create_by_category":
        cat = args.get("category"); nm = args.get("name"); fields = args.get("fields",[])
        if cat and nm and fields:
            return f"Crea un schedule \"{nm}\" para la categoría \"{cat}\" con campos {fields}"

    # Fallback genérico
    return f"Ejecuta la acción {step.get('action')} con {args}"

def _plan_to_pair(plan: Dict[str, Any]) -> Dict[str, Any]:
    plan = canonicalize_plan(normalize_plan(plan))
    steps = plan.get("plan", [])
    prompt = _prompt_from_step(steps[0]) if steps else "Ejecuta este plan en Revit"
    return {"prompt": prompt, "response": plan}

def main():
    ap = argparse.ArgumentParser(description="Convierte planes TAD-DSL en pares prompt→response para SFT.")
    ap.add_argument("--input", "-i", required=True, help="Ruta a JSON (array) o JSONL con planes.")
    ap.add_argument("--output", "-o", required=True, help="Salida JSONL con pares {prompt, response}.")
    args = ap.parse_args()

    data = _load_any(args.input)
    with open(args.output, "w", encoding="utf-8") as fout:
        for item in data:
            pair = _plan_to_pair(item)
            fout.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"[make_pairs_from_plans] Escribí {len(data)} pares → {args.output}")

if __name__ == "__main__":
    main()