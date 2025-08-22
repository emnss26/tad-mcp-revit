from __future__ import annotations
from typing import List, Optional, Union, Literal, Dict, Any
from pydantic import BaseModel, Field

# ---- DUCTS -------------------------------------------------------------------
class ArgsMEPPlaceDuct(BaseModel):
    type: str
    start: List[float] = Field(..., min_items=3, max_items=3)
    end:   List[float] = Field(..., min_items=3, max_items=3)
    level: str
    diameter_mm: Optional[float] = 300

class ArgsMEPPlaceFlexDuct(BaseModel):
    type: str
    path: List[List[float]] = Field(..., min_items=2)
    level: str
    diameter_mm: Optional[float] = 200

class ArgsMEPPlaceDuctFitting(BaseModel):
    type: str
    at: List[float] = Field(..., min_items=3, max_items=3)
    level: str

class ArgsMEPPlaceDuctAccessory(BaseModel):
    type: str
    at: List[float] = Field(..., min_items=3, max_items=3)
    level: str

class ArgsMEPPlaceAirTerminal(BaseModel):
    type: str
    location: List[float] = Field(..., min_items=3, max_items=3)
    level: str

# ---- PIPES -------------------------------------------------------------------
class ArgsMEPPlacePipe(BaseModel):
    type: str
    start: List[float] = Field(..., min_items=3, max_items=3)
    end:   List[float] = Field(..., min_items=3, max_items=3)
    level: str
    diameter_mm: Optional[float] = 50

class ArgsMEPPlacePipeFitting(BaseModel):
    type: str
    at: List[float] = Field(..., min_items=3, max_items=3)
    level: str

class ArgsMEPPlacePipeAccessory(BaseModel):
    type: str
    location: List[float] = Field(..., min_items=3, max_items=3)
    level: str

# ---- FIRE / CONDUIT / CABLETRAY ----------------------------------------------
class ArgsMEPPlaceSprinkler(BaseModel):
    type: str
    location: List[float] = Field(..., min_items=3, max_items=3)
    level: str

class ArgsMEPPlaceConduit(BaseModel):
    type: str
    start: List[float] = Field(..., min_items=3, max_items=3)
    end:   List[float] = Field(..., min_items=3, max_items=3)
    level: str
    diameter_mm: Optional[float] = 25

class ArgsMEPPlaceCableTray(BaseModel):
    type: str
    start: List[float] = Field(..., min_items=3, max_items=3)
    end:   List[float] = Field(..., min_items=3, max_items=3)
    level: str

# ---- CONNECTION --------------------------------------------------------------
class ArgsMEPConnectElements(BaseModel):
    a: Union[str, List[int]]
    b: Union[str, List[int]]

# ---- ELECTRICAL / LIGHTING (opcionales si los usas) --------------------------
class ArgsElectricalPlaceDevice(BaseModel):
    type: str
    location: List[float] = Field(min_items=3, max_items=3)
    level: str

class ArgsLightingPlaceFixture(BaseModel):
    type: str
    location: List[float] = Field(min_items=3, max_items=3)
    level: str

# ---- REGISTRO ----------------------------------------------------------------
ACTION_SCHEMAS: Dict[str, Any] = {
    # Ducts
    "mep.place_duct":          ArgsMEPPlaceDuct,
    "mep.place_flex_duct":     ArgsMEPPlaceFlexDuct,
    "mep.place_duct_fitting":  ArgsMEPPlaceDuctFitting,
    "mep.place_duct_accessory":ArgsMEPPlaceDuctAccessory,
    "mep.place_air_terminal":  ArgsMEPPlaceAirTerminal,

    # Pipes
    "mep.place_pipe":          ArgsMEPPlacePipe,
    "mep.place_pipe_fitting":  ArgsMEPPlacePipeFitting,
    "mep.place_pipe_accessory":ArgsMEPPlacePipeAccessory,

    # Fire / Conduit / Cabletray
    "mep.place_sprinkler":     ArgsMEPPlaceSprinkler,
    "mep.place_conduit":       ArgsMEPPlaceConduit,
    "mep.place_cabletray":     ArgsMEPPlaceCableTray,

    # Connection
    "mep.connect_elements":    ArgsMEPConnectElements,

    # (opcionales)
    "electrical.place_device": ArgsElectricalPlaceDevice,
    "lighting.place_fixture":  ArgsLightingPlaceFixture,
}