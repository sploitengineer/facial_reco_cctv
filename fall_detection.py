"""Compatibility entry point; implementation: sentrycare.fall_detection."""
import importlib
import sys

_module = importlib.import_module('sentrycare.fall_detection')
sys.modules[__name__] = _module
