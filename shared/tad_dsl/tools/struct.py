from __future__ import annotations
from typing import List, Optional, Union, Literal, Dict, Any
from pydantic import BaseModel, Field

Number = float

# ---- STRUCTURE ---------------------------------------------------------------
class ArgsStructCreateColumn(BaseModel):
    type: str
    location: List[float] = Field(..., min_items=3, max_items=3)
    level: str
    base_offset: Optional[float] = 0.0
    top_level: Optional[str] = None
    top_offset: Optional[float] = 0.0

class ArgsStructCreateBeam(BaseModel):
    type: str
    start: List[float] = Field(..., min_items=3, max_items=3)
    end:   List[float] = Field(..., min_items=3, max_items=3)
    level: str

class ArgsStructCreateBrace(BaseModel):
    type: str
    start: List[float] = Field(..., min_items=3, max_items=3)
    end:   List[float] = Field(..., min_items=3, max_items=3)
    level: str

class ArgsStructCreateTruss(BaseModel):
    type: str
    line: List[float] = Field(..., min_items=6, max_items=6, description="startXYZ + endXYZ")
    level: str

class ArgsStructCreateFoundation(BaseModel):
    type: str
    location: List[float] = Field(..., min_items=3, max_items=3)
    level: str

class ArgsRebarPlace(BaseModel):
    host: Optional[Union[str, List[int]]] = "@prev"
    rebar_type: str
    bar_diameter_mm: Optional[float] = 16
    count: Optional[int] = 4
    spacing_mm: Optional[float] = 200

# ---- STAIRS / RAILINGS -------------------------------------------------------
class ArgsStairsCreateByRun(BaseModel):
    base_level: str
    top_level: str
    run_width: float
    location_line: List[List[float]]  # polyline (m)

class ArgsRailingsCreateByPath(BaseModel):
    type: Optional[str] = None
    path: List[List[float]]
    host_stairs: Optional[Union[str, int]] = None

# ---- PATTERNS / ARRAYS -------------------------------------------------------
class ArgsPatternsCreateOnFace(BaseModel):
    host_element_id: int
    face_index: int
    family_type_to_place: str
    u_spacing: Number
    v_spacing: Number
    orientation: Literal["U","V","Auto"] = "Auto"

class ArgsArraysCreateLinear(BaseModel):
    element_id: Union[int, str]
    count: int
    offset: List[Number]  # [dx,dy,dz] (m)
    group: bool = True
    keep_original: bool = True

class ArgsArraysCreateRadial(BaseModel):
    element_id: Union[int, str]
    center_point: List[Number]
    count: int
    angle_deg: Number
    keep_original: bool = True
    group: bool = True

# ---- REGISTRO ----------------------------------------------------------------
ACTION_SCHEMAS: Dict[str, Any] = {
    # Structure
    "structure.create_column":     ArgsStructCreateColumn,
    "structure.create_beam":       ArgsStructCreateBeam,
    "structure.create_brace":      ArgsStructCreateBrace,
    "structure.create_truss":      ArgsStructCreateTruss,
    "structure.create_foundation": ArgsStructCreateFoundation,
    "rebar.place":                 ArgsRebarPlace,

    # Stairs / Railings
    "stairs.create_by_run":    ArgsStairsCreateByRun,
    "railings.create_by_path": ArgsRailingsCreateByPath,

    # Patterns / Arrays
    "patterns.create_on_face": ArgsPatternsCreateOnFace,
    "arrays.create_linear":    ArgsArraysCreateLinear,
    "arrays.create_radial":    ArgsArraysCreateRadial,
}