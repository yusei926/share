"""Register the Furniture-GR00T policy with LeRobot."""

from .configuration_furniture_groot import FurnitureGrootConfig
from .processor_furniture_groot import FurnitureGrootTemporalProgressStep

__all__ = ["FurnitureGrootConfig", "FurnitureGrootTemporalProgressStep"]
