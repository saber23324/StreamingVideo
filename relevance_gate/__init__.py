"""Independent audio/visual correspondence experiments.

Importing this package does not patch or configure the legacy AudioRouter.
"""

from .models import ImageBindCosineGate, PooledAVGate, PrototypeAVGate
from .types import RelevanceGateOutput

__all__ = ["ImageBindCosineGate", "PooledAVGate", "PrototypeAVGate", "RelevanceGateOutput"]
