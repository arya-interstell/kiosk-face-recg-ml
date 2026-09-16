"""YuNet face detection.

Wraps cv2.FaceDetectorYN. Detection runs on a downscaled copy of the frame and
results are mapped back to full-frame coordinates, so callers always work in
capture-resolution pixels.
"""

from dataclasses import dataclass

import cv2
import numpy as np

import config

# Landmark row order produced by YuNet.
RIGHT_EYE, LEFT_EYE, NOSE, RIGHT_MOUTH, LEFT_MOUTH = range(5)


@dataclass
class Face:
    """One detected face, in full-frame pixel coordinates."""

    x: float
    y: float
    w: float
    h: float
    score: float
    landmarks: np.ndarray  # (5, 2) float32: right eye, left eye, nose, r mouth, l mouth

    @property
    def area(self) -> float:
        return self.w * self.h

    @property
    def center(self):
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)

    def area_ratio(self, frame_w: int, frame_h: int) -> float:
        return self.area / float(frame_w * frame_h)


class FaceDetector:
    """Stateless per-frame detector.

    The model is 232 KB and runs in ~1-2 ms at 320px on a desktop CPU, which is
    why we can afford to run it on every single frame rather than tracking
    between detections.
    """

    def __init__(self, model_path=None, detect_width=None):
        self.model_path = model_path or config.FACE_MODEL_PATH
        self.detect_width = detect_width or config.DETECT_WIDTH
        self._detector = None
        self._input_size = None

    def _ensure(self, size):
        if self._detector is None:
            self._detector = cv2.FaceDetectorYN.create(
                model=self.model_path,
                config="",
                input_size=size,
                score_threshold=config.DETECT_SCORE_THRESHOLD,
                nms_threshold=config.DETECT_NMS_THRESHOLD,
                top_k=config.DETECT_TOP_K,
            )
            self._input_size = size
        elif size != self._input_size:
            self._detector.setInputSize(size)
            self._input_size = size

    def detect(self, frame):
        """Return every face in the frame, largest first."""
        h, w = frame.shape[:2]

        if w > self.detect_width:
            scale = self.detect_width / float(w)
            small = cv2.resize(frame, (self.detect_width, int(round(h * scale))),
                               interpolation=cv2.INTER_LINEAR)
        else:
            scale = 1.0
            small = frame

        sh, sw = small.shape[:2]
        self._ensure((sw, sh))

        _, raw = self._detector.detect(small)
        if raw is None or len(raw) == 0:
            return []

        inv = 1.0 / scale
        faces = []
        for row in raw:
            landmarks = row[4:14].reshape(5, 2).astype(np.float32) * inv
            faces.append(Face(
                x=float(row[0]) * inv,
                y=float(row[1]) * inv,
                w=float(row[2]) * inv,
                h=float(row[3]) * inv,
                score=float(row[14]),
                landmarks=landmarks,
            ))

        faces.sort(key=lambda f: f.area, reverse=True)
        return faces

    def largest(self, frame):
        """Return the biggest face in the frame, or None.

        The kiosk only ever talks to one person, and the person standing at it
        is the person whose face fills the most pixels. Picking the largest face
        rather than tracking every face in a busy concourse keeps the per-frame
        cost flat no matter how crowded the background is.
        """
        faces = self.detect(frame)
        return faces[0] if faces else None
