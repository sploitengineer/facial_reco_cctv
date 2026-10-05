"""Compatibility entry point; implementation: sentrycare.face_identity."""
import importlib
import sys

_module = importlib.import_module('sentrycare.face_identity')
sys.modules[__name__] = _module
