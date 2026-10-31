"""Material models and the preset database."""

from softsim.materials.library import list_presets, load_material, save_material
from softsim.materials.models import (
    Adhesion,
    ArrudaBoyce,
    Friction,
    Gent,
    KelvinVoigt,
    LinearElastic,
    MagnetoActive,
    Material,
    MooneyRivlin,
    NeoHookean,
    Ogden,
    PronySeries,
    RayleighDamping,
    StableNeoHookean,
    StVenantKirchhoff,
    Yeoh,
)
from softsim.materials.sampling import sample

__all__ = [
    "Adhesion",
    "ArrudaBoyce",
    "Friction",
    "Gent",
    "KelvinVoigt",
    "LinearElastic",
    "MagnetoActive",
    "Material",
    "MooneyRivlin",
    "NeoHookean",
    "Ogden",
    "PronySeries",
    "RayleighDamping",
    "StableNeoHookean",
    "StVenantKirchhoff",
    "Yeoh",
    "list_presets",
    "load_material",
    "sample",
    "save_material",
]
