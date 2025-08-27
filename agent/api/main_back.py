from __future__ import annotations
import os
import argparse
import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agent.core.planner import ensure_loaded, make_plan, MODEL_ID
from shared.tad_dsl.tool_definitions import ACTION_SCHEMAS

app = FastAPI(title="TAD MCP – Agent", version="0.1.0")

class PlanRequest(BaseModel):
    prompt: str
    context: dict = {}
    dry_run: bool = False

def _filtered_actions():
    """
    Devuelve el catálogo de acciones (respetando TAD_ACTIONS_FILTER)
    con 'required' y 'optional' según los model_fields de Pydantic v2.
    """
    filters = {s.strip() for s in os.environ.get("TAD_ACTIONS_FILTER", "").split(",") if s.strip()}
    out = []
    for name, model in ACTION_SCHEMAS.items():
        if filters and name not in filters:
            continue
        fields = getattr(model, "model_fields", {})
        req, opt = [], []
        for arg_name, finfo in fields.items():
            is_req = False
            try:
                # pydantic v2
                is_req = finfo.is_required()
            except Exception:
                # fallback defensivo
                is_req = getattr(finfo, "default", Ellipsis) is Ellipsis
            (req if is_req else opt).append(arg_name)
        out.append({
            "action": name,
            "args": {
                "required": sorted(req),
                "optional": sorted(opt),
            }
        })
    out.sort(key=lambda x: x["action"])
    return out

@app.on_event("startup")
async def _startup():
    ensure_loaded()

@app.get("/health")
def health():
    return {"ok": True, "model": MODEL_ID}

@app.get("/tools")
def tools():
    actions = _filtered_actions()
    return {"ok": True, "count": len(actions), "actions": actions}

@app.post("/plan")
def plan(req: PlanRequest):
    try:
        plan = make_plan(req.prompt, client_context=req.context)
        # normaliza base
        plan.setdefault("version", "tad-dsl/0.2")
        plan.setdefault("context", {"revit_version": "2025", "units": "SI"})
        plan["context"]["dry_run"] = bool(req.dry_run)
        return {"ok": True, "plan": plan}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8008")))
    args = parser.parse_args()

    import uvicorn
    uvicorn.run("agent.api.main:app", host=args.host, port=args.port, reload=False)
