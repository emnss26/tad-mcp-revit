from __future__ import annotations
from typing import Dict, Any, List
from .tool_definitions import ACTION_SCHEMAS  # <-- solo lee, no exportes nada que reimporte tools

def catalog_summary() -> dict:
    available: List[Dict[str, Any]] = []
    for action, model in ACTION_SCHEMAS.items():
        try:
            fields = getattr(model, "model_fields", {})
            available.append({"action": action, "args": list(fields.keys())})
        except Exception:
            available.append({"action": action, "args": []})
    return {"available_actions": available}