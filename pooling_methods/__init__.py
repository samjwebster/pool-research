"""Dynamic pooling method registry.

Any Python file added to this package that defines a ``BasePoolingMethod``
subclass is imported and registered automatically.
"""

import importlib
import inspect
import pkgutil

from .base import BasePoolingMethod

POOLING_METHODS = {}


def _load_methods():
    for module_info in pkgutil.iter_modules(__path__):
        if module_info.name == "base":
            continue
        module = importlib.import_module(f"{__name__}.{module_info.name}")
        for name, candidate in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(candidate, BasePoolingMethod)
                and candidate is not BasePoolingMethod
                and candidate.__module__ == module.__name__
            ):
                POOLING_METHODS[name] = candidate


from ._baselines import GAP, GAP_GMP, GeM, GMP, LAE, LSE, NWE  # noqa: E402,F401
_load_methods()

for _name, _method in POOLING_METHODS.items():
    globals()[_name] = _method

__all__ = ["BasePoolingMethod", "POOLING_METHODS", *sorted(POOLING_METHODS)]
