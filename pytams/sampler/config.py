from dataclasses import dataclass
from dataclasses import field
from pytams.core import MergePolicy


@dataclass(frozen=True)
class SamplerConfig:
    """Sampler configuration."""

    __section__ = "sampler"
    __merge_policy__ = MergePolicy.IMMUTABLE

    strategy: str = field(
        default="ams",
        metadata={
            "doc": "Sampling strategy to use, either 'ams' or 'montecarlo'.",
        },
    )

    deterministic: bool = field(
        default=False, metadata={"doc": "Use deterministic sampling, seeding all RNGs. fmodel must also do so."}
    )
