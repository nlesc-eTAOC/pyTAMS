from pathlib import Path
import toml
from pytams.core import Config
from pytams.sampler.system_config import SystemConfig


def test_config():
    params_dict = {
        "sampler": {"strategy": "ams"},
        "runtime": {"walltime": 20.0},
        "ams": {"ntrajectories": 20, "nsplititer": 20},
        "trajectory": {"step_size": 0.01},
    }
    with Path("input.toml").open("w") as f:
        toml.dump(params_dict, f)
    config = Config(params_dict)
    sys_cfg = config.load(SystemConfig)
    # sampler_cfg = config.load(SamplerConfig)
    # print(sampler_cfg)
