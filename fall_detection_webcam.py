"""Compatibility entry point; implementation: examples.fall_detection_webcam."""
import importlib
import sys

_module = importlib.import_module('examples.fall_detection_webcam')
if __name__ == '__main__':
    raise SystemExit(_module.main())
else:
    sys.modules[__name__] = _module
