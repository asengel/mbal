class MaterialBalanceError(Exception):
    """Base exception for material-balance calculations."""


class InputValidationError(MaterialBalanceError):
    """Raised when case data are incomplete or physically inconsistent."""


class PressureSolveError(MaterialBalanceError):
    """Raised when a pressure root cannot be bracketed or closed."""


class PVTError(MaterialBalanceError):
    """Raised for invalid or unsupported PVT data."""
