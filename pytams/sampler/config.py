from dataclasses import dataclass
from pytams.config import MergePolicy


@dataclass(frozen=True)
class SamplerConfig:
    """Sampler configuration."""

    __section__ = "sampler"
    __merge_policy__ = MergePolicy.IMMUTABLE

    strategy: str = "ams"
    deterministic: bool = False
