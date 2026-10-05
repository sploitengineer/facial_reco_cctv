"""Compatibility entry point; implementation: examples.phase4_integration."""
import importlib
import sys

_module = importlib.import_module('examples.phase4_integration')
if __name__ == '__main__':
    raise SystemExit(_module.main())
else:
    sys.modules[__name__] = _module
