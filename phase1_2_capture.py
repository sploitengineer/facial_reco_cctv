"""Compatibility entry point; implementation: examples.phase1_2_capture."""
import importlib
import sys

_module = importlib.import_module('examples.phase1_2_capture')
if __name__ == '__main__':
    raise SystemExit(_module.main())
else:
    sys.modules[__name__] = _module
