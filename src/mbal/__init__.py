"""mbal - generalized tank material balance, aquifer modelling and pressure history matching."""

from .aquifers import CTAquifer, FetkovichAquifer, HVEAquifer, NoAquifer, RadialAquiferProperties
from .config import Case, ModelFactory, build_simulator, load_case
from .history_match import HistoryMatcher
from .material_balance import (
    BlackOilMaterialBalance,
    GasMaterialBalance,
    GeneralizedGasMaterialBalance,
    GeneralizedOilMaterialBalance,
    GeneralizedTwoPhaseMaterialBalance,
)
from .simulator import PressureSolverSettings, Simulator

__version__ = "0.3.0"

__all__ = [
    "Simulator", "PressureSolverSettings", "Case", "ModelFactory", "load_case", "build_simulator",
    "HistoryMatcher", "NoAquifer", "FetkovichAquifer", "CTAquifer", "HVEAquifer", "RadialAquiferProperties",
    "GasMaterialBalance", "BlackOilMaterialBalance", "GeneralizedOilMaterialBalance",
    "GeneralizedGasMaterialBalance", "GeneralizedTwoPhaseMaterialBalance",
]
