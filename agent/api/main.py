from __future__ import annotations
import os
import argparse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent.core.planner import ensure_loaded, make_plan, MODEL_ID

app = FastAPI(title="TAD MCP – Agent", version="0.1.0")

class PlanRequest(BaseModel):
    prompt: str
    context: dict = {}
    dry_run: bool = False

@app.on_event("startup")
async def _startup():
    # carga perezosa del modelo al iniciar el server
    ensure_loaded()

@app.get("/health")
def health():
    return {"ok": True, "model": MODEL_ID}

@app.post("/plan")
def plan(req: PlanRequest):
    try:
        plan = make_plan(req.prompt, client_context=req.context)
        # normaliza campos base
        plan.setdefault("version", "tad-dsl/0.2")
        plan.setdefault("context", {"revit_version": "2025", "units": "SI"})
        plan["context"]["dry_run"] = bool(req.dry_run)
        return {"ok": True, "plan": plan}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # puedes lanzar el server directamente: python agent/api/main.py --port 8008
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8008")))
    args = parser.parse_args()

    import uvicorn
    uvicorn.run("agent.api.main:app", host=args.host, port=args.port, reload=False)