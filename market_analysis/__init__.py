"""Market analysis package exports with lazy loading to avoid heavy imports at package import time."""

from typing import TYPE_CHECKING

__all__ = ["PositionRiskManager", "RiskOrderExecutor"]

# Provide type hints without triggering heavy imports at runtime
if TYPE_CHECKING:  # pragma: no cover
    from .position_risk_manager import PositionRiskManager as PositionRiskManager
    from .order_executor import RiskOrderExecutor as RiskOrderExecutor


def __getattr__(name: str):
    """Lazily import heavy modules on attribute access to keep package import cheap.

    This prevents build-time tools or accidental imports from triggering network
    or API client initialization during packaging/installation.
    """
    if name == "PositionRiskManager":
        from .position_risk_manager import PositionRiskManager  # local import
        return PositionRiskManager
    if name == "RiskOrderExecutor":
        from .order_executor import RiskOrderExecutor  # local import
        return RiskOrderExecutor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
