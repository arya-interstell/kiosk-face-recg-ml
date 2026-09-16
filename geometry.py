"""Distance and head-pose estimation from the five YuNet landmarks.

Distance comes from inter-pupillary distance rather than bounding box size.
The box grows and shrinks with hairstyle, head tilt and detector jitter; the
gap between two pupils is a rigid piece of skull, so it makes a far steadier
ruler at kiosk range.
"""

import math

import cv2
import numpy as np

import config
from detector import LEFT_EYE, NOSE, RIGHT_EYE, LEFT_MOUTH, RIGHT_MOUTH

# How far the eye plane sits behind the nose tip, in millimetres, per the model
# below. The IPD ruler measures to the eyes; subtracting this reports the
# distance to the nearest point of the face, which is what "how close is the
# user" means for a kiosk.
EYE_DEPTH_MM = 27.0

# Generic adult face in millimetres, origin at the nose tip.
# Axes match image convention: +x right, +y down, +z away from the camera.
# The eyes and mouth sit behind the nose tip, hence positive z.
# The eye separation is taken from config.IPD_MM rather than hard-coded: the
# same anatomy underpins both the pose solve and the distance ruler, and if the
# two ever disagreed every distance reading would carry a silent scale error.
_EYE_X = config.IPD_MM / 2.0

_MODEL_POINTS = np.array([
    [-_EYE_X, -34.0, EYE_DEPTH_MM],   # right eye (appears left in the image)
    [ _EYE_X, -34.0, EYE_DEPTH_MM],   # left eye
    [    0.0,   0.0,         0.0],    # nose tip
    [ -28.0,   32.0,        24.0],    # right mouth corner
    [  28.0,   32.0,        24.0],    # left mouth corner
], dtype=np.float64)


def inter_pupillary_px(landmarks) -> float:
    """Pixel distance between the two detected eye centres."""
    dx = float(landmarks[LEFT_EYE][0] - landmarks[RIGHT_EYE][0])
    dy = float(landmarks[LEFT_EYE][1] - landmarks[RIGHT_EYE][1])
    return math.hypot(dx, dy)


def estimate_distance_cm(landmarks, focal_px, yaw_deg=None):
    """Distance from camera to the front of the face, in cm, or None.

    Pinhole model: a real-world span of IPD_MM projects to ipd_px pixels at
    distance d, so d = IPD_MM * focal_px / ipd_px. That lands on the eye plane,
    so EYE_DEPTH_MM is removed to give the distance to the nose tip.

    A head turned away foreshortens the eye-to-eye span, which would read as
    "further away". When we already know the yaw we divide it back out; the
    correction is clamped because it blows up as the face approaches profile.
    """
    if not focal_px:
        return None

    ipd = inter_pupillary_px(landmarks)
    if ipd < 1.0:
        return None

    if yaw_deg is not None:
        cos_yaw = math.cos(math.radians(max(-60.0, min(60.0, yaw_deg))))
        ipd = ipd / max(cos_yaw, 0.5)

    eye_plane_mm = config.IPD_MM * focal_px / ipd
    return max(0.0, eye_plane_mm - EYE_DEPTH_MM) / 10.0


def camera_matrix(frame_w, frame_h, focal_px=None):
    """Intrinsics for solvePnP.

    Uses the calibrated focal length when available. Falling back to the frame
    width is the usual rough guess and is good enough for a coarse yaw/pitch
    gate, but a calibrated camera gives noticeably steadier angles.
    """
    f = focal_px or float(frame_w)
    return np.array([
        [f, 0.0, frame_w / 2.0],
        [0.0, f, frame_h / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def estimate_head_pose(landmarks, frame_w, frame_h, focal_px=None):
    """Return (yaw, pitch, roll) in degrees, or None if the solve fails.

    All three are zero when the face points straight down the optical axis.
    Yaw is positive when the subject turns to their left (image right).
    """
    image_points = np.array([
        landmarks[RIGHT_EYE],
        landmarks[LEFT_EYE],
        landmarks[NOSE],
        landmarks[RIGHT_MOUTH],
        landmarks[LEFT_MOUTH],
    ], dtype=np.float64)

    ok, rvec, _ = cv2.solvePnP(
        _MODEL_POINTS,
        image_points,
        camera_matrix(frame_w, frame_h, focal_px),
        np.zeros((4, 1)),
        flags=cv2.SOLVEPNP_SQPNP,
    )
    if not ok:
        return None

    rmat, _ = cv2.Rodrigues(rvec)
    # Rows of the model->camera rotation give the Euler angles directly.
    sy = math.hypot(rmat[0, 0], rmat[1, 0])
    if sy < 1e-6:
        pitch = math.atan2(-rmat[1, 2], rmat[1, 1])
        yaw = math.atan2(-rmat[2, 0], sy)
        roll = 0.0
    else:
        pitch = math.atan2(rmat[2, 1], rmat[2, 2])
        yaw = math.atan2(-rmat[2, 0], sy)
        roll = math.atan2(rmat[1, 0], rmat[0, 0])

    return (
        math.degrees(yaw),
        _wrap180(math.degrees(pitch)),
        _wrap180(math.degrees(roll)),
    )


def _wrap180(angle):
    """Map an angle onto (-180, 180].

    solvePnP returns the head as rotated ~180 degrees about x relative to our
    y-down model, so raw pitch and roll come back near +/-180 for a face
    looking straight ahead. Unwrapping puts a frontal face back at zero.
    """
    angle = (angle + 180.0) % 360.0 - 180.0
    if angle > 90.0:
        angle -= 180.0
    elif angle < -90.0:
        angle += 180.0
    return angle


class EMA:
    """Exponential moving average that tolerates gaps in the signal."""

    def __init__(self, alpha=None):
        self.alpha = config.EMA_ALPHA if alpha is None else alpha
        self.value = None

    def update(self, sample):
        if sample is None:
            return self.value
        if self.value is None:
            self.value = float(sample)
        else:
            self.value += self.alpha * (float(sample) - self.value)
        return self.value

    def reset(self):
        self.value = None
