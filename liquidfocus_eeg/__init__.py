"""Compatibility imports for scripts written before the LiquidFocus rename.

New code should import from ``liquidfocus``. Legacy module and class names
resolve to the same implementation; no parameters or model state are copied.
"""

from importlib import import_module as _import_module
import pkgutil as _pkgutil
import sys as _sys

import liquidfocus as _current


def _legacy_class_aliases(module):
    for name, value in list(vars(module).items()):
        if isinstance(value, type) and "LiquidFocus" in name and "LiquidFocusEEG" not in name:
            setattr(module, name.replace("LiquidFocus", "LiquidFocusEEG"), value)


_legacy_class_aliases(_current)
for _entry in _pkgutil.walk_packages(_current.__path__, _current.__name__ + "."):
    _target = _import_module(_entry.name)
    _legacy_class_aliases(_target)
    _suffix = _entry.name[len(_current.__name__):]
    _sys.modules[__name__ + _suffix] = _target
    if _suffix.count(".") == 1:
        globals()[_suffix[1:]] = _target

__all__ = list(_current.__all__) + ["EvidenceDecoupledLiquidFocusEEG"]
for _name in __all__:
    globals()[_name] = getattr(_current, _name)
