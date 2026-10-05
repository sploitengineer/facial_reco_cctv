"""Compatibility entry point; implementation: examples.phase3_recognition."""
import importlib
import sys

_module = importlib.import_module('examples.phase3_recognition')
if __name__ == '__main__':
    raise SystemExit(_module.main())
else:
    sys.modules[__name__] = _module
