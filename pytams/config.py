"""A configuration class to expose limited configuration."""

from collections.abc import Mapping
from typing import Any


class Config:
    """Lightweight structured access to configuration."""

    def __init__(self, data: Mapping[str, Any]) -> None:
        self._data = data

    def section(self, name: str) -> Mapping[str, Any]:
        """Return a configuration section."""
        return self._data.get(name, {})

    def require(self, section: str, key: str) -> Any:
        """Get a required parameter."""
        try:
            return self._data[section][key]
        except KeyError as exc:
            err_msg = f"Missing required config: [{section}].{key}"
            raise ValueError(err_msg) from exc

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """Get an optional parameter."""
        return self._data.get(section, {}).get(key, default)
