from __future__ import annotations
from typing import Any, Dict, List, Tuple

from .tool_definitions import ACTION_SCHEMAS

# --- FUNCIONES DE VALIDACIÓN ---

def _is_required(field_info) -> bool:
    """Verifica si un campo de un modelo Pydantic es requerido."""
    try:
        return field_info.is_required()  # Pydantic v2
    except Exception:
        return getattr(field_info, "default", Ellipsis) is Ellipsis

def validate_plan(plan: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Valida un plan contra el esquema de acciones. Devuelve (is_ok, [errors])."""
    issues: List[str] = []
    if not isinstance(plan, dict):
        return False, ["El nivel superior del plan debe ser un objeto JSON."]
    
    steps = plan.get("plan")
    if not isinstance(steps, list) or not steps:
        return False, ["La clave 'plan' es requerida y debe ser una lista no vacía de pasos."]

    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            issues.append(f"Paso {i}: Cada paso debe ser un objeto JSON.")
            continue
        
        action = step.get("action")
        if action not in ACTION_SCHEMAS:
            issues.append(f"Paso {i}: La acción '{action}' es desconocida o no está en el catálogo.")
            continue
        
        args = step.get("args")
        if not isinstance(args, dict):
            # Aceptamos args nulos si la acción no tiene args requeridos
            model_fields = getattr(ACTION_SCHEMAS[action], "model_fields", {})
            has_required_args = any(_is_required(finfo) for finfo in model_fields.values())
            if has_required_args:
                 issues.append(f"Paso {i}: La clave 'args' debe ser un objeto JSON.")
            continue

        model = ACTION_SCHEMAS[action]
        fields = getattr(model, "model_fields", {})

        # Validar argumentos requeridos
        for arg_name, finfo in fields.items():
            if _is_required(finfo) and arg_name not in args:
                issues.append(f"Paso {i}: Falta el argumento requerido '{arg_name}' para la acción '{action}'.")

        # Validar argumentos desconocidos
        for k in args.keys():
            if k not in fields:
                issues.append(f"Paso {i}: Argumento desconocido '{k}' encontrado en la acción '{action}'.")
                
    return len(issues) == 0, issues

# --- FUNCIONES DE NORMALIZACIÓN Y CANONICALIZACIÓN ---

def normalize_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Asegura que el plan tenga la estructura base correcta (versión, plan, etc.)."""
    if "plan" not in plan:
        # Asume que el objeto entero es el plan si la clave 'plan' falta
        return {"version": "tad-dsl/0.2", "plan": [plan]}
    if "version" not in plan:
        plan["version"] = "tad-dsl/0.2"
    return plan

def canonicalize_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Pone el plan en un formato estándar y consistente (ej. ordena claves). Aún no implementado."""
    # TODO: Implementar lógica si se necesita, como ordenar las claves de los diccionarios
    # para hacer comparaciones más sencillas. Por ahora, solo lo devuelve.
    return plan