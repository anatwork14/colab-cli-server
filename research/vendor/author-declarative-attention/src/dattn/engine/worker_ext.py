"""The `worker_extension_cls` that makes the five RPC names reachable (integration).

`collective_rpc(name, ...)` with a **string** resolves `getattr(worker, name)`
(`N:v1/serial_utils.py:500-506`), and `dattn_probe` is not a method of vLLM's `Worker`, so
every one of the five names raised `NotImplementedError: Method 'dattn_probe' is not
implemented` the first time anything called them for real. Passing the function object
instead does not help either: a callable is invoked as `partial(fn, worker)(*args)`, so
the worker arrives as the FIRST positional argument, and every function in
`dattn.engine.api` takes it LAST, as a keyword, because that is what lets the same
functions serve an in-process test with no worker at all.

vLLM's own answer to this is `--worker-extension-cls`: the class is resolved by qualified
name and injected into the worker class's bases, after which its methods are exactly what
`collective_rpc` can call (`N:v1/worker/worker_base.py:285-309`). It asserts that no name
here already exists on the worker, which is why every method is prefixed `dattn_`.

The profile carries it in `engine_args`, so `dattn serve`, `launch.engine_kwargs` and the
bench driver all get it from the same place and the DA and vanilla arms are identical in
it.
"""

from __future__ import annotations

from typing import Any

from dattn.engine import api

__all__ = ["DattnWorkerExtension"]


class DattnWorkerExtension:
    """One thin method per `api.RPC_NAMES` entry. No logic lives here."""

    def dattn_probe(self) -> dict[str, Any]:
        return api.dattn_probe(self)

    def dattn_record(self, enabled: bool, window_id: str = "") -> dict[str, Any]:
        return api.dattn_record(enabled, window_id=window_id, worker=self)

    def dattn_drain(self) -> list[Any]:
        return api.dattn_drain(self)

    def dattn_audit(self) -> dict[str, Any]:
        return api.dattn_audit(self)

    def dattn_assert(self) -> dict[str, Any]:
        return api.dattn_assert(self)

    def dattn_boot(self) -> dict[str, Any]:
        return api.dattn_boot(self)
