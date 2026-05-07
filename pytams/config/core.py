"""A configuration class to expose limited configuration."""

from __future__ import annotations
from collections.abc import Mapping
from dataclasses import MISSING
from dataclasses import Field
from dataclasses import asdict
from dataclasses import fields
from dataclasses import is_dataclass
from typing import Any
from typing import TypeVar
from typing import cast
from typing import get_type_hints

T = TypeVar("T")


class MergePolicy:
    """A merge policy for configuration dataclasses."""

    IMMUTABLE = "immutable"
    REPLACE = "replace"


class Config:
    """Lightweight structured access to configuration.

    pyREVS input parameters are mostly handled through dataclasses
    to ensure default values are available and types are enforced:
    TOML file -> Config -> Typed dataclass.

    The typed dataclass are the final object used to instanciate
    pyREVS objects (Sampler, Database, Trajectories, ...)

    The Config class provides a simple interface to access configuration
    parameters in a more structured way. To add a new section to
    the TOML input file, create a new dataclass with a `__section__`
    class attribute set to the name of the section.

    """

    def __init__(self, data: Mapping[str, Any]) -> None:
        self._data = data

    def section(self, name: str) -> Config:
        """Return a configuration section."""
        value = self._data.get(name, {})
        if not isinstance(value, Mapping):
            err_msg = f"Config section {name} is not a mapping !"
            raise TypeError(err_msg)
        return Config(value)

    def section_dict(self, name: str) -> dict[str, Any]:
        """Return a section as dict."""
        value = self._data.get(name, {})
        if not isinstance(value, Mapping):
            err_msg = f"Config section {name} is not a mapping !"
            raise TypeError(err_msg)
        return dict(value)

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

    def as_dict(self) -> dict[str, Any]:
        """Return the config Mapping as a dict."""
        return dict(self._data)

    def load(self, cls: type[T]) -> T:
        """Load a dataclass of the provided type from this config section."""
        if not is_dataclass(cls):
            err_msg = f"{cls} is not a dataclass"
            raise TypeError(err_msg)

        # Use the custom __load__ method if available
        # This casting is an easy fix for mypy
        # but defining a Protocol would be more elegant
        if hasattr(cls, "__load__"):
            return cast("T", cast("Any", cls).__load__(self))

        cfg = self._resolve_section(cls)
        kwargs = self._build_kwargs(cfg, cls)

        return cls(**kwargs)

    def _resolve_section(self, cls: type[T]) -> Config:
        """Return the appropriate Config view for a given dataclass.

        If the dataclass defines a ``__section__`` attribute, this method
        extracts that subsection from the underlying data and wraps it
        into a new ``Config`` instance. Otherwise, the current instance
        is returned.

        Args:
            cls: The dataclass type being loaded.

        Returns:
            A ``Config`` instance scoped to the relevant section.

        Raises:
            TypeError: If the resolved section is not a mapping.
        """
        section_name = getattr(cls, "__section__", None)

        if section_name is None:
            return self

        sub = self._data.get(section_name, {})

        if not isinstance(sub, Mapping):
            err_msg = f"Config section {section_name} is not a mapping !"
            raise TypeError(err_msg)

        return Config(sub)

    def _build_kwargs(self, cfg: Config, cls: type[Any]) -> dict[str, Any]:
        """Construct keyword arguments for dataclass instantiation.

        This method iterates over the dataclass fields, resolves their
        values from the configuration data (or defaults), and applies
        recursive loading for nested dataclasses.

        Args:
            cfg: The Config instance scoped to the relevant section.
            cls: The dataclass type to instantiate.

        Returns:
            A dictionary of field names to resolved values.
        """
        type_hints = get_type_hints(cls)
        data = cfg.as_dict()

        kwargs: dict[str, Any] = {}

        for f in fields(cls):
            value = self._get_field_value(cfg, data, f, type_hints[f.name])
            kwargs[f.name] = value

        return kwargs

    def _get_field_value(
        self,
        cfg: Config,
        data: Mapping[str, Any],
        f: Field[Any],
        ftype: Any,
    ) -> Any:
        """Resolve the value of a single dataclass field.

        The resolution order is:
            1. Value from configuration data
            2. Field default
            3. Field default factory

        If none are available, a missing value placeholder is returned.

        Nested dataclasses are resolved recursively.

        Args:
            cfg: The Config instance for nested resolution.
            data: The raw mapping for the current section.
            f: The dataclass field definition.
            ftype: The resolved type hint for the field.

        Returns:
            The resolved field value.
        """
        key = f.name

        if key in data:
            value = data[key]
        elif f.default is not MISSING:
            value = f.default
        elif f.default_factory is not MISSING:
            value = f.default_factory()
        else:
            err_msg = f"Missing required field: {key}"
            raise ValueError(err_msg)

        return self._resolve_nested(cfg, value, ftype)

    def _resolve_nested(self, cfg: Config, value: Any, ftype: Any) -> Any:
        """Resolve nested dataclass values.

        If the field type is a dataclass, this method will:
            - Load it from its declared section if it defines ``__section__``
            - Otherwise, attempt to load it from an inline mapping

        Non-dataclass values are returned unchanged.

        Args:
            cfg: The Config instance used for loading.
            value: The raw value extracted from configuration data.
            ftype: The type hint associated with the field.

        Returns:
            The resolved value, possibly transformed into a dataclass instance.
        """
        if isinstance(ftype, type) and is_dataclass(ftype):
            section_name = getattr(ftype, "__section__", None)

            if section_name is not None:
                return cfg.load(ftype)

            if isinstance(value, Mapping):
                return Config(value).load(ftype)

        return value


def collect_sections(*configs: Any) -> dict[str, Any]:
    """Build a TOML-ready dict from config dataclasses.

    Each dataclass must define __section__.
    """
    result: dict[str, Any] = {}

    for cfg in configs:
        if cfg is None:
            continue

        section = getattr(type(cfg), "__section__", None)
        if section is None:
            err_msg = f"{type(cfg).__name__} has no __section__"
            raise ValueError(err_msg)

        result[section] = asdict(cfg)

    return result


def merge_config(old: T, new: T) -> T:
    """Merge two configuration dataclass objects."""
    if old is None:
        return new
    if new is None:
        return old

    policy = getattr(type(old), "__merge_policy__", MergePolicy.IMMUTABLE)

    if policy == MergePolicy.IMMUTABLE:
        if old != new:
            err_msg = f"{type(old).__name__} is immutable"
            raise ValueError(err_msg)
        return old

    if policy == MergePolicy.REPLACE:
        return new

    err_msg = f"Unknown merge policy for {type(old).__name__}"
    raise ValueError(err_msg)


def print_config_help(cls: type) -> None:
    """Print a help for a pyREVS config dataclass."""
    section = getattr(cls, "__section__", cls.__name__.lower())
    docstring = cls.__doc__ or ""

    print(f"\n[{section}]", docstring.strip())  # noqa: T201

    for f in fields(cls):
        typ = "[" + str(f.type) + "]"
        doc = f.metadata.get("doc", "")
        default = f.default
        print(f"{f.name:<20} {typ:<12} default={default}")  # noqa: T201
        print(f"    {doc}")  # noqa: T201
