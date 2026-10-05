"""Compatibility entry point; implementation: sentrycare.evaluation."""
import importlib
import sys

_module = importlib.import_module('sentrycare.evaluation')
if __name__ == '__main__':
    raise SystemExit(_module.main())
else:
    sys.modules[__name__] = _module
