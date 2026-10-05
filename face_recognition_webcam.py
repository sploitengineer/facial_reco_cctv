"""Compatibility entry point for the standalone face demo."""
import runpy

if __name__ == '__main__':
    runpy.run_module('examples.face_recognition_webcam', run_name='__main__')
