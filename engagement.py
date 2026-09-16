"""Dwell state machine: turns per-frame measurements into engagement events.

This file, not the choice of detector, is what decides whether the kiosk feels
responsive or twitchy. Three ideas do the work:

  hysteresis  - it is harder to lose engagement than to gain it, so a person
                standing exactly on a threshold does not flicker.
  grace       - a short dropout (blink, glance aside, someone walking past the
                camera) does not reset the dwell timer to zero.
  cooldown    - after a disengagement we stay quiet briefly, so the avatar does
                not re-greet someone who is already walking away.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import config


class State(str, Enum):
    IDLE = "idle"            # nobody worth talking to
    CANDIDATE = "candidate"  # someone qualifies, dwell timer running
    ENGAGED = "engaged"      # dwell satisfied, avatar should be awake
    COOLDOWN = "cooldown"    # just disengaged, ignoring everyone briefly


class Event(str, Enum):
    CANDIDATE_DETECTED = "candidate_detected"
    CANDIDATE_LOST = "candidate_lost"
    ENGAGED = "engaged"
    DISENGAGED = "disengaged"


@dataclass
class Measurement:
    """Everything we know about the frontmost face this frame."""

    present: bool = False
    area_ratio: float = 0.0
    distance_cm: Optional[float] = None
    yaw_deg: Optional[float] = None
    pitch_deg: Optional[float] = None
    roll_deg: Optional[float] = None
    score: float = 0.0
    bbox: Optional[tuple] = None

    # Filled in by the state machine so callers can debug a rejection.
    reasons: List[str] = field(default_factory=list)


class EngagementTracker:
    def __init__(self):
        self.state = State.IDLE
        self.dwell_start = None
        self.last_qualified = None
        self.state_since = 0.0
        self.engaged_at = None
        self._cooldown_until = 0.0

    # -- gates ------------------------------------------------------------

    def _qualifies(self, m: Measurement, lenient: bool) -> bool:
        """Does this face meet the bar for being 'at the kiosk'?

        `lenient` applies the wider exit thresholds, used once the person is
        already engaged so we do not drop them for a small movement.
        """
        m.reasons = []
        if not m.present:
            m.reasons.append("no_face")
            return False

        min_area = config.MIN_FACE_AREA_RATIO
        if lenient:
            min_area *= config.FACE_AREA_EXIT_FACTOR
        if m.area_ratio < min_area:
            m.reasons.append("too_small")

        if m.distance_cm is not None:
            max_cm = config.MAX_ENGAGE_DISTANCE_CM
            if lenient:
                max_cm += config.DISTANCE_EXIT_MARGIN_CM
            if m.distance_cm > max_cm:
                m.reasons.append("too_far")
            if m.distance_cm < config.MIN_ENGAGE_DISTANCE_CM:
                m.reasons.append("too_close")

        if config.REQUIRE_FRONTAL_POSE and m.yaw_deg is not None:
            margin = config.ANGLE_EXIT_MARGIN_DEG if lenient else 0.0
            if abs(m.yaw_deg) > config.MAX_YAW_DEG + margin:
                m.reasons.append("looking_away_yaw")
            if m.pitch_deg is not None and abs(m.pitch_deg) > config.MAX_PITCH_DEG + margin:
                m.reasons.append("looking_away_pitch")

        return not m.reasons

    # -- main step --------------------------------------------------------

    def update(self, m: Measurement, now: float):
        """Advance the machine by one frame. Returns a list of Events."""
        events = []
        lenient = self.state is State.ENGAGED
        qualified = self._qualifies(m, lenient)

        if qualified:
            self.last_qualified = now

        if self.state is State.COOLDOWN:
            if now >= self._cooldown_until:
                self._to(State.IDLE, now)
            return events

        if self.state is State.IDLE:
            if qualified:
                self.dwell_start = now
                self._to(State.CANDIDATE, now)
                events.append(Event.CANDIDATE_DETECTED)
            return events

        # CANDIDATE and ENGAGED both tolerate a gap before giving up.
        grace = (config.ENGAGED_GRACE_SECONDS if self.state is State.ENGAGED
                 else config.CANDIDATE_GRACE_SECONDS)
        gap = now - (self.last_qualified or now)

        if gap > grace:
            if self.state is State.ENGAGED:
                events.append(Event.DISENGAGED)
                self._cooldown_until = now + config.DISENGAGE_COOLDOWN_SECONDS
                self._to(State.COOLDOWN, now)
            else:
                events.append(Event.CANDIDATE_LOST)
                self._to(State.IDLE, now)
            self.dwell_start = None
            self.engaged_at = None
            return events

        if self.state is State.CANDIDATE and self.dwell_seconds(now) >= config.ENGAGE_DWELL_SECONDS:
            self.engaged_at = now
            self._to(State.ENGAGED, now)
            events.append(Event.ENGAGED)

        return events

    def _to(self, state: State, now: float):
        self.state = state
        self.state_since = now

    # -- readouts ---------------------------------------------------------

    def dwell_seconds(self, now: float) -> float:
        """How long the current person has been continuously present."""
        if self.dwell_start is None:
            return 0.0
        return now - self.dwell_start

    def progress(self, now: float) -> float:
        """0..1 fraction of the way to engagement. Handy for a progress ring."""
        if self.state is State.ENGAGED:
            return 1.0
        if self.dwell_start is None:
            return 0.0
        return min(1.0, self.dwell_seconds(now) / config.ENGAGE_DWELL_SECONDS)

    def reset(self):
        self.__init__()
