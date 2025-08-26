from __future__ import annotations
from typing import List, Optional, Union, Literal, Dict, Any
from pydantic import BaseModel, Field, field_validator, FieldValidationInfo

Number = float

# ---- GET / UI / EXPORT --------------------------------------------------------
class ArgsGetParameterValue(BaseModel):
    source: Union[str, List[int]] = "@prev"
    name: Optional[str] = None
    builtin: Optional[str] = None
    as_number: bool = False

    @field_validator("name", "builtin", mode="before")
    def _strip_empty(cls, v):
        if isinstance(v, str) and not v.strip():
            return None
        return v

    @field_validator("builtin")
    def _require_name_or_builtin(cls, v, info: FieldValidationInfo):
        name_val = info.data.get("name")
        if (v in (None, "")) and (name_val in (None, "")):
            raise ValueError("Debes proporcionar 'name' o 'builtin'")
        return v

class ArgsGetElementLocation(BaseModel):
    source: Union[str, List[int]] = "@prev"
    coordinate_system: Literal["Project","Internal"] = "Project"

class ArgsGetBoundingBox(BaseModel):
    source: Union[str, List[int]] = "@prev"
    view: Optional[str] = None
    units: Literal["Model","Meters","Millimeters","Feet"] = "Model"

class ArgsUiPromptForSelection(BaseModel):
    category_filter: Optional[Union[str, List[str]]] = None
    multi: bool = False
    prompt: str = "Selecciona elemento(s) en Revit"

class ArgsUiShowMessage(BaseModel):
    title: str = "Información"
    message: str
    buttons: Literal["OK","YesNo","YesNoCancel"] = "OK"
    severity: Literal["info","warning","error"] = "info"

class ArgsUiPromptForTextInput(BaseModel):
    title: str = "Entrada requerida"
    message: str = "Escribe un valor:"
    default: Optional[str] = None
    allow_empty: bool = False

class ArgsExportElementsToCsv(BaseModel):
    source: Union[str, List[int]] = "@prev"
    fields: List[str]
    path: str
    delimiter: str = ","
    include_headers: bool = True

class ArgsExportElementsToExcel(BaseModel):
    source: Union[str, List[int]] = "@prev"
    fields: List[str]
    path: str  # .xlsx
    sheet_name: str = "Data"
    include_headers: bool = True

# ---- ELEMENTS (genéricos) -----------------------------------------------------
class ArgsSetParameter(BaseModel):
    source: Optional[Union[str, List[int]]] = "@prev"
    name: str
    value: Union[str, float, int, bool]

class ArgsElementsCopy(BaseModel):
    source: Optional[Union[str, List[int]]] = "@prev"
    offset: List[float] = Field(..., min_items=3, max_items=3)
    count: Optional[int] = 1

class ArgsElementsRotate(BaseModel):
    source: Optional[Union[str, List[int]]] = "@prev"
    origin: List[float] = Field(..., min_items=3, max_items=3)
    axis: Literal["X","Y","Z"] = "Z"
    angle_deg: float

class ArgsElementsMirror(BaseModel):
    source: Optional[Union[str, List[int]]] = "@prev"
    plane_point: List[float] = Field(..., min_items=3, max_items=3)
    plane_normal: List[float] = Field(..., min_items=3, max_items=3)

class ArgsElementsJoin(BaseModel):
    a: Union[str, List[int]]
    b: Union[str, List[int]]

class ArgsElementsCut(BaseModel):
    a: Union[str, List[int]]
    b: Union[str, List[int]]

class ArgsElementsTrim(BaseModel):
    a: Union[str, List[int]]
    b: Union[str, List[int]]

class ArgsElementsAlign(BaseModel):
    element_to_align_id: Union[int, str]
    target_element_id: int
    method: Literal["Faces","Centerline","Edges"] = "Centerline"
    alignment: Literal["Move","Rotate","Both"] = "Move"

class ArgsElementsAttachTopBase(BaseModel):
    wall_ids: Union[str, List[int]] = "@prev"
    target_element_id: int
    attachment_type: Literal["Top","Base"] = "Top"

class ArgsElementsEditProfile(BaseModel):
    element_id: int
    new_profile_points: List[List[Number]]
    coordinate_system: Literal["Project","Internal"] = "Project"

# ---- GRAPHICS / VISIBILITY ----------------------------------------------------
class ArgsGraphicsOverrideColor(BaseModel):
    source: Optional[Union[str, List[int]]] = "@prev"
    rgb: List[int] = Field(..., min_items=3, max_items=3)
    view: Optional[str] = "Active"

class ArgsGraphicsSetCategoryVisibility(BaseModel):
    category: str
    view: Optional[str] = "Active"
    visible: Optional[bool] = True

# ---- PHASES -------------------------------------------------------------------
class ArgsPhasesCreate(BaseModel):
    name: str
    order_after: Optional[str] = None

class ArgsElementsSetPhase(BaseModel):
    source: Optional[Union[str, List[int]]] = "@prev"
    created_phase: Optional[str] = None
    demolished_phase: Optional[str] = None

# ---- LINKS / COORDS -----------------------------------------------------------
class ArgsLinksReload(BaseModel):
    names: Optional[List[str]] = None
    all: Optional[bool] = False

class ArgsLinksUnload(BaseModel):
    names: Optional[List[str]] = None
    all: Optional[bool] = False

class ArgsLinksBind(BaseModel):
    name: str
    as_attached: Optional[bool] = False

class ArgsCoordinatesAcquireShared(BaseModel):
    from_link_name: str

# ---- WORKSHARING / WORKSETS ---------------------------------------------------
class ArgsWorksetsCreate(BaseModel):
    names: List[str]

class ArgsElementsSetWorkset(BaseModel):
    source: Union[str, List[int]] = "@prev"
    workset_name: str

class ArgsWorksharingSynchronize(BaseModel):
    compact: Optional[bool] = False
    relinquish_all: Optional[bool] = True

# ---- SCHEDULES / SHEETS / LEGENDS --------------------------------------------
class ArgsSchedulesCreateByCategory(BaseModel):
    category: str
    name: str
    fields: List[str] = Field(..., min_items=1)

class ArgsSchedulesAddCalculatedValue(BaseModel):
    schedule_name: str
    field_name: str
    formula: str
    result_type: Optional[str] = None

class ArgsSchedulesUpdateFilterSort(BaseModel):
    schedule_name: str
    filters: Optional[List[Dict[str, str]]] = None
    sort_by: Optional[List[Dict[str, str]]] = None

class ArgsSchedulesPlaceOnSheet(BaseModel):
    schedule_name: str
    sheet_number: str
    location: List[float] = Field(min_items=2, max_items=2)

class ArgsSheetsCreateWithViews(BaseModel):
    number: str
    name: str
    titleblock_type: str
    views: List[str] = Field(..., min_items=1)

class ArgsSheetsPlaceView(BaseModel):
    sheet_number: str
    view_name: str
    location: List[float] = Field(min_items=2, max_items=2, description="x,y en mm en el plano")

class ArgsLegendsCreate(BaseModel):
    name: str
    scale: Optional[int] = 100

# ---- BATCH / EXCEL ------------------------------------------------------------
class ArgsFamiliesBatchUpdateFromExcel(BaseModel):
    path: str
    sheet: Optional[str] = None
    id_field: Literal["UniqueId","ElementId","FamilyName"] = "FamilyName"
    mapping: Dict[str, str] = Field(default_factory=dict, description="param_name -> column_name")
    create_missing: Optional[bool] = False

# ---- REGISTRO ----------------------------------------------------------------
ACTION_SCHEMAS: Dict[str, Any] = {
    # GET / UI / EXPORT
    "get.parameter_value": ArgsGetParameterValue,
    "get.element_location": ArgsGetElementLocation,
    "get.bounding_box":     ArgsGetBoundingBox,
    "ui.prompt_for_selection": ArgsUiPromptForSelection,
    "ui.show_message":         ArgsUiShowMessage,
    "ui.prompt_for_text_input":ArgsUiPromptForTextInput,
    "export.elements_to_csv":  ArgsExportElementsToCsv,
    "export.elements_to_excel":ArgsExportElementsToExcel,

    # ELEMENTS (genéricos)
    "elements.set_parameter": ArgsSetParameter,
    "elements.copy":          ArgsElementsCopy,
    "elements.rotate":        ArgsElementsRotate,
    "elements.mirror":        ArgsElementsMirror,
    "elements.join":          ArgsElementsJoin,
    "elements.cut":           ArgsElementsCut,
    "elements.trim":          ArgsElementsTrim,
    "elements.align":         ArgsElementsAlign,
    "elements.attach_top_base": ArgsElementsAttachTopBase,
    "elements.edit_profile":    ArgsElementsEditProfile,

    # GRAPHICS / VISIBILITY
    "graphics.override_color":           ArgsGraphicsOverrideColor,
    "graphics.set_category_visibility":  ArgsGraphicsSetCategoryVisibility,

    # PHASES
    "phases.create":      ArgsPhasesCreate,
    "elements.set_phase": ArgsElementsSetPhase,

    # LINKS / COORDS
    "links.reload":               ArgsLinksReload,
    "links.unload":               ArgsLinksUnload,
    "links.bind":                 ArgsLinksBind,
    "coordinates.acquire_shared": ArgsCoordinatesAcquireShared,

    # WORKSHARING
    "worksets.create":             ArgsWorksetsCreate,
    "elements.set_workset":        ArgsElementsSetWorkset,
    "worksharing.synchronize":     ArgsWorksharingSynchronize,

    # SCHEDULES / SHEETS / LEGENDS
    "schedules.create_by_category": ArgsSchedulesCreateByCategory,
    "schedules.add_calculated_value": ArgsSchedulesAddCalculatedValue,
    "schedules.update_filter_sort": ArgsSchedulesUpdateFilterSort,
    "schedules.place_on_sheet":     ArgsSchedulesPlaceOnSheet,
    "sheets.create_with_views":     ArgsSheetsCreateWithViews,
    "sheets.place_view":            ArgsSheetsPlaceView,
    "legends.create":               ArgsLegendsCreate,

    # BATCH
    "families.batch_update_from_excel": ArgsFamiliesBatchUpdateFromExcel,
}