"""Face embedding with SFace.

SFace turns an aligned face crop into 128 numbers. Two crops of the same person
give vectors pointing in a similar direction; two different people do not. That
is the whole idea - the "identity" of a face is a direction in 128-dimensional
space, and comparing people is a dot product.

Alignment matters more than people expect. SFace is trained on faces warped so
the eyes and mouth sit at fixed positions, and OpenCV's alignCrop does that
warp using exactly the five landmarks YuNet already gives us. The detector and
the recogniser were designed as a pair.
"""

import cv2
import numpy as np

import config


def face_to_row(face):
    """Pack our Face back into the flat row alignCrop expects.

    OpenCV's recogniser wants the detector's raw output format: bbox, then the
    five landmarks, then the score.
    """
    return np.array(
        [face.x, face.y, face.w, face.h, *face.landmarks.flatten(), face.score],
        dtype=np.float32,
    )


def sharpness(image) -> float:
    """Variance of the Laplacian - a standard, cheap blur measure.

    A motion-blurred face still detects fine but embeds badly, and a blurred
    template poisons every future comparison against that person.
    """
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(grey, cv2.CV_64F).var())


def crop_quality(face, measurement, crop) -> float:
    """Score a candidate crop, or return 0 if it is unusable.

    Higher is better. Used to keep only the best few frames of a visit rather
    than whichever ones happened to arrive first.
    """
    if min(face.w, face.h) < config.RECOGNITION_MIN_FACE_PX:
        return 0.0

    yaw = abs(measurement.yaw_deg or 0.0)
    pitch = abs(measurement.pitch_deg or 0.0)
    if yaw > config.RECOGNITION_MAX_YAW_DEG or pitch > config.RECOGNITION_MAX_PITCH_DEG:
        return 0.0

    sharp = sharpness(crop)
    if sharp < config.RECOGNITION_MIN_SHARPNESS:
        return 0.0

    # Reward frontality and sharpness; size only breaks ties.
    frontality = 1.0 - (yaw / config.RECOGNITION_MAX_YAW_DEG) * 0.5 \
                     - (pitch / config.RECOGNITION_MAX_PITCH_DEG) * 0.3
    return frontality * min(sharp, 500.0) * min(face.w, face.h)


class FaceRecognizer:
    """Wraps cv2.FaceRecognizerSF. Loads the 38 MB model lazily."""

    def __init__(self, model_path=None):
        self.model_path = model_path or config.FACE_RECOGNITION_MODEL_PATH
        self._rec = None

    @property
    def rec(self):
        if self._rec is None:
            self._rec = cv2.FaceRecognizerSF.create(self.model_path, "")
        return self._rec

    def align(self, frame, face):
        """Warp a face to the canonical 112x112 SFace expects."""
        return self.rec.alignCrop(frame, face_to_row(face))

    def embed(self, aligned_crop) -> np.ndarray:
        """128-d unit vector for an aligned crop.

        Normalising here means every later comparison is a plain dot product,
        and nothing downstream has to remember to normalise.
        """
        raw = self.rec.feature(aligned_crop).flatten().astype(np.float32)
        norm = np.linalg.norm(raw)
        return raw / norm if norm > 0 else raw


def combine(embeddings) -> np.ndarray:
    """Fuse several embeddings of one visit into a single template.

    The median is used rather than the mean: one bad frame - a blink, a hand
    across the face - drags a mean noticeably but barely moves a median.
    """
    stacked = np.vstack(embeddings)
    template = np.median(stacked, axis=0).astype(np.float32)
    norm = np.linalg.norm(template)
    return template / norm if norm > 0 else template


def similarity(a, b) -> float:
    """Cosine similarity of two unit vectors, in [-1, 1]."""
    return float(np.dot(a, b))
