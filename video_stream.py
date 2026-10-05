"""Compatibility entry point; implementation: sentrycare.video_stream."""
import importlib
import sys

_module = importlib.import_module('sentrycare.video_stream')
sys.modules[__name__] = _module
