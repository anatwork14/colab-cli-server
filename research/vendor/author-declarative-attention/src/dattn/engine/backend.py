"""`DattnBackend`, built from the pinned base backend class.

`_make_backend` subclasses the pinned base and overrides only `get_name()` and
`get_builder_cls()`, so `supports_mm_prefix()`, `get_supported_head_sizes()`,
`get_supported_kernel_block_sizes()` and `get_preferred_block_size()` are all inherited
verbatim. That inheritance is load-bearing for Gemma: `TritonAttentionBackend` is the
only non-Flex backend that returns True from `supports_mm_prefix()`, and a subclass that
broke it would fail backend selection outright (DESIGN.md section 2.2).

`DattnBackend` is a module-level name because `register_backend` takes a class PATH, and
`resolve_obj_by_qualname("dattn.engine.backend.DattnBackend")` has to find it. It does
not exist until `install_backend` runs, and reading it before then is a boot error with
the reason rather than an `AttributeError` three frames deep.
"""

from __future__ import annotations

from typing import Any

from dattn.contracts import BackendProfile, DattnBootError
from dattn.engine.builder import make_builder_cls

__all__ = ["BACKEND_NAME", "_make_backend", "installed_base", "install_backend"]

BACKEND_NAME = "CUSTOM"

_INSTALLED: dict[str, Any] = {"backend": None, "base": None, "profile": None}


def _make_backend(base: type, profile: BackendProfile | None = None) -> type:
    """Return a subclass of `base` overriding only `get_name` and `get_builder_cls`."""
    builder_cls = make_builder_cls(base.get_builder_cls(), profile)  # type: ignore[attr-defined]

    class DattnBackend(base):  # type: ignore[misc, valid-type]
        dattn_base_backend: type = base
        dattn_profile: BackendProfile | None = profile

        @staticmethod
        def get_name() -> str:
            return BACKEND_NAME

        @staticmethod
        def get_builder_cls() -> type:
            return builder_cls

    DattnBackend.__name__ = "DattnBackend"
    DattnBackend.__qualname__ = "DattnBackend"
    DattnBackend.__module__ = __name__
    return DattnBackend


def install_backend(base: type, profile: BackendProfile | None = None) -> type:
    """Build `DattnBackend` from `base` and publish it as a module-level name."""
    cls = _make_backend(base, profile)
    _INSTALLED["backend"] = cls
    _INSTALLED["base"] = base
    _INSTALLED["profile"] = profile
    globals()["DattnBackend"] = cls
    return cls


def installed_base() -> type | None:
    return _INSTALLED["base"]


def __getattr__(name: str) -> Any:
    if name == "DattnBackend":
        raise DattnBootError(
            "dattn.engine.backend.DattnBackend does not exist yet. The vllm.general_plugins "
            "entry point builds it from the profile's base backend at plugin load. If you see "
            "this, dattn.engine.plugin.install_plugin never ran, or it ran without a profile: "
            "set DATTN_MODEL (and optionally DATTN_CAPABILITY, DATTN_USE), or DATTN_BASE_BACKEND, "
            "in the engine's environment (DESIGN.md section 2.2)."
        )
    raise AttributeError(name)
