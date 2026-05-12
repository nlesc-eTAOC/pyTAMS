import pytest
from pytams.core import Config
from pytams.sampler.system_config import SystemConfig


def test_instantiate_sysconfig():
    """Test instantiation of SystemConfig class."""
    params_dict = {
        "sampler": {"strategy": "ams"},
        "runtime": {"walltime": 20.0},
        "ams": {"ntrajectories": 20, "nsplititer": 20},
        "trajectory": {"step_size": 0.01},
    }
    config = Config(params_dict)
    _ = config.load(SystemConfig)


def test_amscfg_nosplit():
    """Test validate the AMSConfig class."""
    params_dict = {
        "sampler": {"strategy": "ams"},
        "ams": {"ntrajectories": 20},
    }
    config = Config(params_dict)
    syscfg = config.load(SystemConfig)

    with pytest.raises(ValueError):
        syscfg.strategy.validate()


def test_amscfg_nontraj():
    """Test validate the AMSConfig class."""
    params_dict = {
        "sampler": {"strategy": "ams"},
        "ams": {"nsplititer": 20},
    }
    config = Config(params_dict)
    syscfg = config.load(SystemConfig)

    with pytest.raises(ValueError):
        syscfg.strategy.validate()


def test_amscfg_tams():
    """Test validate the AMSConfig class."""
    params_dict = {
        "sampler": {"strategy": "ams"},
        "ams": {"ntrajectories": 20, "nsplititer": 20},
    }
    config = Config(params_dict)
    syscfg = config.load(SystemConfig)

    with pytest.raises(ValueError):
        syscfg.strategy.validate()


def test_amscfg_ams():
    """Test validate the AMSConfig class."""
    params_dict = {
        "sampler": {"strategy": "ams"},
        "ams": {"ntrajectories": 20, "nsplititer": 20, "variant": "ams"},
    }
    config = Config(params_dict)
    syscfg = config.load(SystemConfig)

    with pytest.raises(ValueError):
        syscfg.strategy.validate()


def test_amscfg_wrongvariant():
    """Test validate the AMSConfig class."""
    params_dict = {
        "sampler": {"strategy": "ams"},
        "ams": {"ntrajectories": 20, "nsplititer": 20, "variant": "unknown"},
    }
    config = Config(params_dict)
    syscfg = config.load(SystemConfig)

    with pytest.raises(ValueError):
        syscfg.strategy.validate()


def test_mccfg_nontraj():
    """Test validate the MCConfig class."""
    params_dict = {
        "sampler": {"strategy": "montecarlo"},
    }
    config = Config(params_dict)
    syscfg = config.load(SystemConfig)

    with pytest.raises(ValueError):
        syscfg.strategy.validate()
