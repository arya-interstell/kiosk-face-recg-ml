"""Frame -> measurement -> engagement state, with timings.

Kept free of any capture or transport concerns. One instance owns the whole
state of one video stream - dwell timer, smoothing filters, visit evidence -
so the service can run a separate instance per connected kiosk without them
interfering with each other.
"""

import time
from dataclasses import asdict, dataclass
from typing import List, Optional

import config
from detector import FaceDetector
from engagement import EngagementTracker, Event, Measurement, State
from geometry import EMA, estimate_distance_cm, estimate_head_pose, inter_pupillary_px
from session import VisitRecognizer


@dataclass
class FrameResult:
    timestamp: float
    state: str
    events: List[str]
    dwell_seconds: float
    progress: float
    measurement: Measurement
    detect_ms: float
    total_ms: float
    fps: float
    identity: object = None      # MatchResult, only on the frame engagement fires
    recognition_ms: float = 0.0
    focal_px: float = None       # the focal length this frame was measured with


class KioskPipeline:
    def __init__(self, detector: Optional[FaceDetector] = None, visits=None,
                 focal_px: Optional[float] = None):
        self.detector = detector or FaceDetector()
        self.tracker = EngagementTracker()
        # Per-instance, not global: two kiosks streaming into one server have
        # two different cameras, and a focal length belongs to a camera.
        self.focal_px = config.FOCAL_LENGTH_PX if focal_px is None else focal_px
        self.visits = visits
        if self.visits is None and config.RECOGNITION_ENABLED:
            self.visits = VisitRecognizer()
        self._distance = EMA()
        self._yaw = EMA()
        self._pitch = EMA()
        self._roll = EMA()
        self._fps = EMA(alpha=0.1)
        self._last_frame_ts = None

    def process(self, frame, now: Optional[float] = None) -> FrameResult:
        """Advance one frame. The normal path."""
        now = time.monotonic() if now is None else now
        t0 = time.perf_counter()
        h, w = frame.shape[:2]

        face = self.detector.largest(frame)
        detect_ms = (time.perf_counter() - t0) * 1000.0
        return self._advance(face, frame, w, h, now, t0, detect_ms, from_frame=True)

    def idle(self, now: Optional[float] = None) -> FrameResult:
        """Advance with no frame available at all.

        A stream that stops delivering is not the same as a stream showing an
        empty room, but engagement must treat them alike: someone who was mid
        conversation when their kiosk's camera died has still left, and the
        avatar has to be told. Without this the state machine simply freezes
        on its last frame, because every timeout in it is evaluated on arrival.
        """
        now = time.monotonic() if now is None else now
        return self._advance(None, None, 0, 0, now, time.perf_counter(), 0.0,
                             from_frame=False)

    def _advance(self, face, frame, w, h, now, t0, detect_ms, from_frame):
        m = Measurement()
        if face is not None:
            pose = estimate_head_pose(face.landmarks, w, h, self.focal_px)
            yaw = pose[0] if pose else None

            m.present = True
            m.score = face.score
            m.bbox = (round(face.x), round(face.y), round(face.w), round(face.h))
            m.area_ratio = face.area_ratio(w, h)
            m.distance_cm = self._distance.update(
                estimate_distance_cm(face.landmarks, self.focal_px, yaw)
            )
            if pose:
                m.yaw_deg = self._yaw.update(pose[0])
                m.pitch_deg = self._pitch.update(pose[1])
                m.roll_deg = self._roll.update(pose[2])
        else:
            # Let the filters decay out rather than snapping to a stale value
            # the moment the next face appears.
            self._distance.reset()
            self._yaw.reset()
            self._pitch.reset()
            self._roll.reset()

        events = self.tracker.update(m, now)

        # Recognition rides along on the dwell: gather evidence while the timer
        # counts down, spend it the instant engagement fires, and forget the
        # visitor as soon as they leave.
        identity = None
        recognition_ms = 0.0
        if self.visits is not None:
            if face is not None and self.tracker.state in (State.CANDIDATE, State.ENGAGED):
                self.visits.observe(frame, face, m)
            if Event.ENGAGED in events:
                t_rec = time.perf_counter()
                identity = self.visits.identify()
                recognition_ms = (time.perf_counter() - t_rec) * 1000.0
            if Event.DISENGAGED in events or Event.CANDIDATE_LOST in events:
                self.visits.reset()

        # fps measures how fast frames arrive, so idle ticks must not count.
        if from_frame:
            if self._last_frame_ts is not None:
                dt = now - self._last_frame_ts
                if dt > 0:
                    self._fps.update(1.0 / dt)
            self._last_frame_ts = now

        return FrameResult(
            timestamp=time.time(),
            state=self.tracker.state.value,
            events=[e.value for e in events],
            dwell_seconds=round(self.tracker.dwell_seconds(now), 3),
            progress=round(self.tracker.progress(now), 3),
            measurement=m,
            detect_ms=round(detect_ms, 2),
            total_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            fps=round(self._fps.value or 0.0, 1),
            identity=identity,
            recognition_ms=round(recognition_ms, 2),
            focal_px=self.focal_px,
        )


def result_to_payload(result: FrameResult, kind: str = "state") -> dict:
    """Flatten a FrameResult into the JSON shape published to clients."""
    m = result.measurement
    return {
        "type": kind,
        "timestamp": result.timestamp,
        "state": result.state,
        "dwell_seconds": result.dwell_seconds,
        "progress": result.progress,
        "engaged": result.state == State.ENGAGED.value,
        "face": {
            "present": m.present,
            "score": round(m.score, 3),
            "bbox": m.bbox,
            "area_ratio": round(m.area_ratio, 5),
            "distance_cm": None if m.distance_cm is None else round(m.distance_cm, 1),
            "yaw_deg": None if m.yaw_deg is None else round(m.yaw_deg, 1),
            "pitch_deg": None if m.pitch_deg is None else round(m.pitch_deg, 1),
            "roll_deg": None if m.roll_deg is None else round(m.roll_deg, 1),
            "rejected_for": m.reasons,
        },
        "perf": {
            "detect_ms": result.detect_ms,
            "total_ms": result.total_ms,
            "fps": result.fps,
        },
        "calibrated": bool(result.focal_px),
        "identity": _identity_payload(result.identity),
    }


def _identity_payload(match):
    """The identity block, or None when this frame carried no identification."""
    if match is None:
        return None
    return {
        "id": match.identity_id,
        "returning": match.is_returning,
        "visit_count": match.visit_count,
        "timestamps": match.timestamps,
        "similarity": round(match.similarity, 4),
        "runner_up": round(match.runner_up, 4),
        "ambiguous": match.ambiguous,
    }
