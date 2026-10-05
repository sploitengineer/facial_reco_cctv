"""Compatibility entry point; implementation: examples.combined_prototype."""
import importlib
import sys

_module = importlib.import_module('examples.combined_prototype')
if __name__ == '__main__':
    raise SystemExit(_module.main())
else:
    sys.modules[__name__] = _module
