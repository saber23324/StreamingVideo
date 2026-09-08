"""Audio-conditioned visual bottleneck modules.

Audio is used only to construct visual attention queries.  No audio tensor from
this package is passed to the language model.
"""

from .adbt import AudioConditionedBottleneck, AudioQueryGenerator
from .beats_encoder import BEATsAudioEncoder

__all__ = ["AudioConditionedBottleneck", "AudioQueryGenerator", "BEATsAudioEncoder"]
