from __future__ import annotations
from typing import Dict, Any

# importa los módulos por disciplina
from .core   import ACTION_SCHEMAS as CORE
from .tools.arch   import ACTION_SCHEMAS as ARCH
from .tools.struct import ACTION_SCHEMAS as STRUCT
from .tools.mep    import ACTION_SCHEMAS as MEP

# merge estable: la clave es el nombre público de la acción (contrato con Bridge)
ACTION_SCHEMAS: Dict[str, Any] = {}
ACTION_SCHEMAS.update(CORE)
ACTION_SCHEMAS.update(ARCH)
ACTION_SCHEMAS.update(STRUCT)
ACTION_SCHEMAS.update(MEP)

# smoke test opcional
if __name__ == "__main__":
    print(f"[tools] acciones registradas: {len(ACTION_SCHEMAS)}")