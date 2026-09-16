"""
Central configuration for the airport kiosk presence service.

Every tunable lives here. Nothing else in the codebase hard-codes a threshold.

Two kinds of values live in this file:

  * TUNING values  - product decisions (how long is "staring", how close is
                     "at the kiosk"). Edit these freely.
  * CALIBRATION    - a physical property of the camera that produced the
                     frames. Do not guess these. calibration.json holds the
                     server-side default; a client that knows its own camera
                     should send `focal_length_px` in its hello instead.
"""

import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------
# Frames arrive over the WebSocket as encoded images (JPEG is the expected
# format). The service never opens a camera itself - it has none.

# Hard ceiling on a single inbound message. Also handed to the websockets
# server as max_size, so an oversized frame is rejected at the protocol layer
# rather than after we have already buffered it. A 1280x720 JPEG at quality 80
# is ~120 KB, so this is a generous 60x headroom, not a target.
MAX_FRAME_BYTES = 8 * 1024 * 1024

# Frames wider than this are downscaled immediately after decode. Detection
# already runs at DETECT_WIDTH, but recognition crops and the pose solve work
# on the full frame, and nothing at kiosk range benefits from more than this.
INGEST_MAX_WIDTH = 1280

# Mirror inbound frames horizontally. Normally the sender is already sending
# correctly-oriented frames, so this defaults off; a client can override it
# per-connection in its hello. Does not affect the maths either way.
DEFAULT_FLIP_HORIZONTAL = False

# Refuse new connections past this. Each session holds its own detector and
# identity gallery, so this is the memory knob.
MAX_SESSIONS = 8

# Threads available for decode + detect. OpenCV releases the GIL inside both,
# so these genuinely run in parallel. Roughly one per expected active session.
WORKER_THREADS = 8

# Close a session that has sent nothing at all for this long. A kiosk with a
# wedged camera should free its slot rather than hold it forever.
SESSION_IDLE_TIMEOUT_S = 60.0

# Largest step the dwell timer will take from one frame to the next in
# client-clock mode. Replaying recorded footage at 10x sends small steps and is
# unaffected; this only stops a client advancing its own engagement timer by
# claiming a frame came from the future.
MAX_CLIENT_CLOCK_STEP_S = 1.0


# ---------------------------------------------------------------------------
# Face detector (YuNet)
# ---------------------------------------------------------------------------

FACE_MODEL_PATH = os.path.join(BASE_DIR, "models", "face_detection_yunet_2023mar.onnx")

# Detection is done on a copy of the frame scaled down to this width. YuNet is
# trained around 320px inputs and this is the single biggest latency lever:
# 320 -> ~1-2 ms/frame on a desktop CPU. Raise only if you need to detect
# faces further away than the kiosk's engagement range.
DETECT_WIDTH = 320

# Minimum detector confidence for a box to be considered a face at all.
DETECT_SCORE_THRESHOLD = 0.75

# Non-maximum suppression IoU threshold.
DETECT_NMS_THRESHOLD = 0.3

# Upper bound on boxes kept before NMS.
DETECT_TOP_K = 50


# ---------------------------------------------------------------------------
# Calibration  (measured, not guessed)
# ---------------------------------------------------------------------------

# Mean adult inter-pupillary distance in millimetres. 63mm is the accepted
# adult average; it is the anatomical ruler the distance estimate is built on.
# Children and outliers will read slightly further away than they are.
IPD_MM = 63.0

# Focal length in PIXELS of the camera that produced the frames, at the
# resolution they are sent in. None = uncalibrated; the service reports no
# centimetres and falls back to the face-area gate alone.
#
# This is a property of the SENDER, not of this server. calibration.json
# supplies the default below; a client streaming from a different camera should
# send its own `focal_length_px` at handshake, which overrides it for that
# session only. Sending frames at a different resolution than the one the
# calibration was measured at scales the focal length proportionally - the
# service applies that correction itself when the client declares its width.
FOCAL_LENGTH_PX = None

CALIBRATION_FILE = os.path.join(BASE_DIR, "calibration.json")


# ---------------------------------------------------------------------------
# Engagement gates - "is this person AT the kiosk?"
# ---------------------------------------------------------------------------
# Each gate has an ENTER threshold and a looser EXIT threshold. The gap between
# them is hysteresis: it stops a person hovering exactly on the boundary from
# flickering the avatar on and off several times a second.

# Face bounding box area as a fraction of total frame area, required to even be
# considered. This is the "minimum threshold area" gate and it runs before any
# distance maths, so an uncalibrated sender still behaves sensibly.
MIN_FACE_AREA_RATIO = 0.035

# Once engaged, the face may shrink to this fraction of the enter threshold
# before we call it a disengagement.
FACE_AREA_EXIT_FACTOR = 0.75

# Engagement distance band, in centimetres. The near bound rejects a face
# pressed against the lens (someone cleaning the screen, a bag brushing past).
MAX_ENGAGE_DISTANCE_CM = 120.0
MIN_ENGAGE_DISTANCE_CM = 25.0

# Added to MAX_ENGAGE_DISTANCE_CM when deciding to DROP an engaged person.
DISTANCE_EXIT_MARGIN_CM = 20.0


# ---------------------------------------------------------------------------
# Attention gates - "are they facing the kiosk?"
# ---------------------------------------------------------------------------
# Head pose from solvePnP on the five face landmarks. Yaw is left/right, pitch
# is up/down, both in degrees, both zero when looking straight at the camera.
# This is head direction, not eye gaze - the right proxy at kiosk range.

MAX_YAW_DEG = 22.0
MAX_PITCH_DEG = 18.0

# Added to both limits when deciding to drop an already-engaged person, so a
# glance at the luggage does not end the conversation.
ANGLE_EXIT_MARGIN_DEG = 8.0

# Set False to ignore head pose entirely and gate on presence + size only.
REQUIRE_FRONTAL_POSE = True


# ---------------------------------------------------------------------------
# Dwell timing - "for how long?"
# ---------------------------------------------------------------------------

# How long a person must continuously satisfy every gate before we declare them
# engaged and wake the avatar.
ENGAGE_DWELL_SECONDS = 2.0

# While counting up to ENGAGE_DWELL_SECONDS, tolerate this long a break in
# detection without resetting the timer. Covers blinks, a head turn, and people
# walking between the user and the camera.
CANDIDATE_GRACE_SECONDS = 0.4

# Same idea once engaged, but more forgiving: the user is mid-conversation and
# we do not want a passer-by to hang up on them.
ENGAGED_GRACE_SECONDS = 1.2

# After a disengagement, ignore new candidates for this long. Stops the avatar
# re-greeting someone who is walking away and glances back.
DISENGAGE_COOLDOWN_SECONDS = 1.0


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------

# Exponential moving average factor for distance and head angles.
# 1.0 = no smoothing (jittery), 0.1 = very smooth (laggy).
EMA_ALPHA = 0.35


# ---------------------------------------------------------------------------
# Recognition - "have we seen this person before?"
# ---------------------------------------------------------------------------
# Recognition runs ONCE per visitor, not per frame: crops are collected while
# the dwell timer counts down, and the identification happens at the moment
# engagement fires. Cost is ~10 ms per customer, not per frame.

RECOGNITION_ENABLED = True

FACE_RECOGNITION_MODEL_PATH = os.path.join(
    BASE_DIR, "models", "face_recognition_sface_2021dec.onnx")

# How many good crops to gather per visit before averaging them into one
# template. Voting across frames cancels most single-frame noise; the dwell
# supplies far more frames than this to choose from, so we can afford to be
# picky.
RECOGNITION_SAMPLES = 10
RECOGNITION_MIN_SAMPLES = 3

# A crop is only worth embedding if the face is near-frontal and sharp. These
# are deliberately stricter than the engagement gates: a face good enough to
# talk to is not necessarily good enough to identify.
RECOGNITION_MAX_YAW_DEG = 15.0
RECOGNITION_MAX_PITCH_DEG = 12.0
RECOGNITION_MIN_FACE_PX = 80          # shorter bbox side, in pixels
RECOGNITION_MIN_SHARPNESS = 25.0      # variance of Laplacian on the crop

# --- Matching ---------------------------------------------------------------
# Cosine similarity between 128-d embeddings, in [-1, 1].
#
# OpenCV's documented 1:1 threshold for SFace is 0.363. That is NOT safe here:
# we ask "is this ANY of the N people we know?", so every new traveller is
# compared against the whole gallery and the chance of a false match grows with
# N. Hence a stricter bar, plus a margin test below.
#
# TUNE THESE ON REAL FOOTAGE before going live - see README.
MATCH_THRESHOLD = 0.45

# The best candidate must beat the runner-up by this much. If the top two are
# neck and neck that is ambiguity, not a match, and we enrol a new identity
# instead. Errors are asymmetric: a spurious new ID costs one row, a false
# match greets a stranger as somebody else.
MATCH_MARGIN = 0.05

# Faces vary with light, glasses and angle, so each person keeps several
# templates and matches against their best one.
MAX_TEMPLATES_PER_IDENTITY = 3

# A fresh template is only added if it differs from the stored ones by at least
# this much - otherwise we would store three copies of the same view.
TEMPLATE_NOVELTY_THRESHOLD = 0.75

# --- Visits and retention ---------------------------------------------------

# Someone lingering at the kiosk is ONE visit, not forty. A re-appearance
# within this window updates the existing visit rather than appending.
REVISIT_WINDOW_SECONDS = 300.0

# Templates older than this are purged on load and on write. This is both the
# privacy retention policy and the accuracy control: a small gallery is what
# keeps false matches rare. Raising it degrades both.
IDENTITY_RETENTION_HOURS = 48.0

# JSON is the source of truth, shared between sessions. Writes are atomic and
# lock-protected; see identity_store.py for the limits of that on network
# filesystems.
IDENTITY_STORE_PATH = os.environ.get(
    "KIOSK_IDENTITY_STORE", os.path.join(BASE_DIR, "identities.json"))

# How often the running service applies the retention policy.
PURGE_INTERVAL_S = 600.0


# ---------------------------------------------------------------------------
# WebSocket service
# ---------------------------------------------------------------------------

# 0.0.0.0 because this is deployed on a server and the kiosks are elsewhere.
# It is the deployment's job to put this behind TLS and a firewall - see README.
SERVICE_HOST = os.environ.get("KIOSK_HOST", "0.0.0.0")
SERVICE_PORT = int(os.environ.get("KIOSK_PORT", "8765"))

# Shared secret required from every connecting client, via an
# `Authorization: Bearer <token>` header or a `token` field in its hello.
# Unset = no authentication, and the service says so loudly at startup.
AUTH_TOKEN = os.environ.get("KIOSK_AUTH_TOKEN") or None

# Push a `state` telemetry message at this rate regardless of transitions, so a
# newly connected subscriber knows the current situation without waiting for an
# edge. Set to 0 to emit transitions only.
STATE_BROADCAST_HZ = 5.0

# Outbound queue depth per client. Telemetry is dropped when it fills;
# transition events never are.
CLIENT_QUEUE_SIZE = 32

# Keepalive. A kiosk whose network drops should free its session promptly
# rather than leave the server holding a half-open socket.
PING_INTERVAL_S = 20.0
PING_TIMEOUT_S = 20.0


# ---------------------------------------------------------------------------
# Debug
# ---------------------------------------------------------------------------

# Log a latency summary every N frames, per session. 0 disables.
LATENCY_LOG_EVERY = 300


# ---------------------------------------------------------------------------
# Calibration override - keep at the bottom of the file
# ---------------------------------------------------------------------------

CALIBRATION = {}


def load_calibration():
    """Overlay calibration.json onto this module, if it exists.

    Kept separate from the tuning values above because it is measured
    per-device: two kiosks with different cameras share config.py but never
    share a focal length. This is only the default for clients that do not
    declare their own.
    """
    global FOCAL_LENGTH_PX, CALIBRATION
    if not os.path.exists(CALIBRATION_FILE):
        return False
    with open(CALIBRATION_FILE) as fh:
        data = json.load(fh)
    CALIBRATION = data
    value = data.get("focal_length_px")
    if value:
        FOCAL_LENGTH_PX = float(value)
        return True
    return False


IS_CALIBRATED = load_calibration()

# Width the default focal length was measured at. Needed to rescale it when a
# client sends frames at a different resolution without declaring its own.
CALIBRATION_WIDTH = CALIBRATION.get("frame_width")
