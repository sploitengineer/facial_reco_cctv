"""Video acquisition with bounded RTSP reconnects and deterministic cleanup."""

import logging
import os
import time

logger = logging.getLogger("SentryCare")


def frames(source, max_reconnect=5, retry_delay=2.0, capture_factory=None, sleep=time.sleep):
    if max_reconnect < 0 or retry_delay < 0:
        raise ValueError("Reconnect count and retry delay cannot be negative")
    if capture_factory is None:
        import cv2
        def capture_factory(value):
            if isinstance(value, int) and os.name == "nt":
                return cv2.VideoCapture(value, cv2.CAP_DSHOW)
            if isinstance(value, str) and value.lower().startswith(("rtsp://", "rtsps://")):
                return cv2.VideoCapture(value, cv2.CAP_FFMPEG, [
                    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000,
                    cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000,
                ])
            return cv2.VideoCapture(value)
    rtsp = isinstance(source, str) and source.lower().startswith(("rtsp://", "rtsps://"))
    cap, attempts, reconnected = None, 0, False
    try:
        while True:
            if cap is None:
                cap = capture_factory(source)
                logger.info("Opening %s source", "RTSP" if rtsp else "local video")
            opened = cap.isOpened()
            if not opened and not rtsp:
                raise RuntimeError("Could not open local camera or video source")
            ok, frame = cap.read() if opened else (False, None)
            if ok:
                attempts = 0
                yield frame, reconnected
                reconnected = False
                continue
            cap.release()
            cap = None
            if not rtsp:
                return
            if attempts >= max_reconnect:
                raise RuntimeError(f"Stream unavailable after {max_reconnect} reconnect attempts")
            attempts += 1
            reconnected = True
            logger.warning("RTSP stream unavailable; reconnect %d/%d", attempts, max_reconnect)
            sleep(retry_delay)
    finally:
        if cap is not None:
            cap.release()
