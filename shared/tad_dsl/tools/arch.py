from __future__ import annotations
from typing import List, Optional, Union, Literal, Dict, Any
from pydantic import BaseModel, Field

Number = float

# ---- LEVELS / GRIDS -----------------------------------------------------------
class ArgsLevelsCreate(BaseModel):
    name: str
    elevation: float  # metros

class ArgsGridsCreateLinear(BaseModel):
    name: str | None = None
    start: list[float] = Field(..., min_items=3, max_items=3)
    end:   list[float] = Field(..., min_items=3, max_items=3)

class ArgsGridsCreateAxis(BaseModel):
    name: str
    start: List[float] = Field(..., min_items=3, max_items=3)
    end:   List[float] = Field(..., min_items=3, max_items=3)

class ArgsGridsCreateRadial(BaseModel):
    center_point: List[Number]
    number_of_radial_lines: int
    number_of_circular_lines: int = 0
    radial_spacing_m: Number = 0.0
    circular_spacing_deg: Number
    naming_prefix: str = "A"
    start_index: int = 1

# ---- WALLS / FLOORS / CEILINGS / ROOFS ---------------------------------------
class ArgsWallsCreateLinear(BaseModel):
    start: List[float] = Field(..., min_items=3, max_items=3, description="m")
    end:   List[float] = Field(..., min_items=3, max_items=3, description="m")
    height: float
    level: str
    type: str
    structural: bool = False

class ArgsFloorsCreateRectangular(BaseModel):
    origin: List[float] = Field(..., min_items=3, max_items=3)
    width: float
    depth: float
    level: str
    type: str

class ArgsCeilingsCreateRectangular(BaseModel):
    origin: List[float] = Field(..., min_items=3, max_items=3)
    width: float
    depth: float
    level: str
    type: str
    height_offset: Optional[float] = 2.7

class ArgsRoofsCreateFootprint(BaseModel):
    level: str
    type: str
    slope: Optional[float] = 30.0  # deg
    perimeter: List[List[float]] = Field(..., min_items=3, description="lista de puntos (m) cerrada")

# ---- WINDOWS / DOORS / FAMILIES ----------------------------------------------
class ArgsWindowsPlaceInWall(BaseModel):
    host: Optional[Union[str, int]] = "@prev"
    type: str
    location: List[float] = Field(..., min_items=3, max_items=3)
    sill_height: Optional[float] = 0.9

class ArgsDoorsPlaceInWall(BaseModel):
    host: Optional[Union[str, int]] = "@prev"
    type: str
    location: List[float] = Field(..., min_items=3, max_items=3)

class ArgsFamiliesPlaceInstance(BaseModel):
    type: str
    location: List[float] = Field(..., min_items=3, max_items=3)
    level: str
    host: Optional[Union[str, int]] = None

# ---- ROOMS / ANNOTATIONS / DIMENSIONS / LINES / REGIONS -----------------------
class ArgsRoomsCreateAtPoint(BaseModel):
    level: str
    location: List[float] = Field(..., min_items=3, max_items=3)

class ArgsAnnotationsTagElements(BaseModel):
    source: Optional[Union[str, List[int]]] = "@prev"
    tag_type: Optional[str] = None
    view: Optional[str] = "Active"

class ArgsAnnotationsTagAll(BaseModel):
    category: str
    view: Optional[str] = "Active"

class ArgsAnnotationsTagRooms(BaseModel):
    level: str

class ArgsDimensionsCreateLinear(BaseModel):
    points: List[List[float]] = Field(..., min_items=2)
    view: Optional[str] = "Active"

class ArgsLinesCreateModelLine(BaseModel):
    start: List[float] = Field(..., min_items=3, max_items=3)
    end:   List[float] = Field(..., min_items=3, max_items=3)
    level: str

class ArgsRegionsCreateFilled(BaseModel):
    profile: List[List[float]] = Field(..., min_items=3, description="cerrada")
    view: Optional[str] = "Active"
    fill_type: Optional[str] = "Solid Black"

# ---- OPENINGS ----------------------------------------------------------------
class ArgsOpeningsCreateShaft(BaseModel):
    profile_points: List[List[Number]]
    base_level: str
    top_level: str

class ArgsOpeningsCreateByFace(BaseModel):
    host_element_id: int
    face_index: int
    profile_points: List[List[Number]]

class ArgsOpeningsCreateForMEP(BaseModel):
    mep_element_ids: Union[str, List[int]] = "@prev"
    sleeve_type: Optional[str] = None
    offset_mm: Number = 0
    host_preference: Literal["Auto","Floors","Walls","Roofs"] = "Auto"

# ---- CURTAIN WALL -------------------------------------------------------------
class ArgsCurtainAddGrid(BaseModel):
    wall: Union[str, List[int]] = "@prev"
    direction: Literal["U","V"]
    offset: float
    justify: Optional[Literal["Center","Start","End"]] = "Center"

class ArgsCurtainAddMullion(BaseModel):
    grid_line_id: Union[str, int]
    mullion_type: Optional[str] = None

class ArgsCurtainReplacePanelType(BaseModel):
    panel_ids: Union[str, List[int]] = "@prev"
    to_type_name: str

# ---- VIEWS --------------------------------------------------------------------
class ArgsViewsSetScale(BaseModel):
    view: Optional[str] = "Active"
    scale: int = Field(..., ge=1)

class ArgsViewsApplyTemplate(BaseModel):
    view: str
    template_name: str

class ArgsViewsRename(BaseModel):
    from_: str = Field(..., alias="from")
    to: str

class ArgsViewsDuplicate(BaseModel):
    source: str
    mode: Literal["duplicate","with_detailing","dependent"] = "duplicate"
    new_name: Optional[str] = None

class ArgsViewsCreatePlan(BaseModel):
    level_name: str
    discipline: Optional[Literal["Architectural","Structural","Mechanical","Electrical","Plumbing"]] = "Architectural"

class ArgsViewsCreateSection(BaseModel):
    start: List[float] = Field(..., min_items=3, max_items=3)
    end:   List[float] = Field(..., min_items=3, max_items=3)
    name: Optional[str] = None

class ArgsViewsCreateElevation(BaseModel):
    origin: List[float] = Field(..., min_items=3, max_items=3)
    direction: Literal["N","S","E","W"] = "N"
    name: Optional[str] = None

class ArgsViewsSetPhaseFilter(BaseModel):
    view: Optional[str] = "Active"
    phase: Optional[str] = None
    phase_filter: Optional[str] = None

class ArgsViewsSetDetailLevel(BaseModel):
    view: Optional[str] = "Active"
    level: Literal["Coarse","Medium","Fine"]

class ArgsViewsSetDiscipline(BaseModel):
    view: Optional[str] = "Active"
    discipline: Literal["Architecture","Structure","Mechanical","Electrical","Coordination"]

class ArgsViewsSetUnderlay(BaseModel):
    view: Optional[str] = "Active"
    bottom: Optional[str] = None
    top: Optional[str] = None
    orientation: Optional[Literal["Plan","RCP"]] = None

class ArgsViewsSetScopeBox(BaseModel):
    view: Optional[str] = "Active"
    scope_box_name: Optional[str] = None

class ArgsViewsCropRegion(BaseModel):
    view: Optional[str] = "Active"
    enabled: Optional[bool] = True
    offset: Optional[List[float]] = Field(default=None, min_items=4, max_items=4, description="l,t,r,b en m")

class ArgsViewsRenameByRegex(BaseModel):
    pattern: str
    replace: str
    scope: Literal["Views","Sheets","Both"] = "Views"

# ---- REGISTRO ----------------------------------------------------------------
ACTION_SCHEMAS: Dict[str, Any] = {
    # Levels / Grids
    "levels.create":       ArgsLevelsCreate,
    "grids.create_linear": ArgsGridsCreateLinear,
    "grids.create_axis":   ArgsGridsCreateAxis,
    "grids.create_radial": ArgsGridsCreateRadial,

    # Walls / Floors / Ceilings / Roofs
    "walls.create_linear":          ArgsWallsCreateLinear,
    "floors.create_rectangular":    ArgsFloorsCreateRectangular,
    "ceilings.create_rectangular":  ArgsCeilingsCreateRectangular,
    "roofs.create_footprint":       ArgsRoofsCreateFootprint,

    # Windows / Doors / Families
    "windows.place_in_wall":    ArgsWindowsPlaceInWall,
    "doors.place_in_wall":      ArgsDoorsPlaceInWall,
    "families.place_instance":  ArgsFamiliesPlaceInstance,

    # Rooms / Annotations / Dims / Lines / Regions
    "rooms.create_at_point":     ArgsRoomsCreateAtPoint,
    "annotations.tag_elements":  ArgsAnnotationsTagElements,
    "annotations.tag_all":       ArgsAnnotationsTagAll,
    "annotations.tag_rooms":     ArgsAnnotationsTagRooms,
    "dimensions.create_linear":  ArgsDimensionsCreateLinear,
    "lines.create_model_line":   ArgsLinesCreateModelLine,
    "regions.create_filled":     ArgsRegionsCreateFilled,

    # Openings
    "openings.create_shaft":   ArgsOpeningsCreateShaft,
    "openings.create_by_face": ArgsOpeningsCreateByFace,
    "openings.create_for_mep": ArgsOpeningsCreateForMEP,

    # Curtain Wall
    "curtain.add_grid":            ArgsCurtainAddGrid,
    "curtain.add_mullion":         ArgsCurtainAddMullion,
    "curtain.replace_panel_type":  ArgsCurtainReplacePanelType,

    # Views
    "views.set_scale":         ArgsViewsSetScale,
    "views.apply_template":    ArgsViewsApplyTemplate,
    "views.rename":            ArgsViewsRename,
    "views.duplicate":         ArgsViewsDuplicate,
    "views.create_plan":       ArgsViewsCreatePlan,
    "views.create_section":    ArgsViewsCreateSection,
    "views.create_elevation":  ArgsViewsCreateElevation,
    "views.set_phase_filter":  ArgsViewsSetPhaseFilter,
    "views.set_detail_level":  ArgsViewsSetDetailLevel,
    "views.set_discipline":    ArgsViewsSetDiscipline,
    "views.set_underlay":      ArgsViewsSetUnderlay,
    "views.set_scope_box":     ArgsViewsSetScopeBox,
    "views.crop_region":       ArgsViewsCropRegion,
    "views.rename_by_regex":   ArgsViewsRenameByRegex,
}