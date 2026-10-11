"""Per-visit recognition.

The kiosk already makes someone wait five seconds before the avatar wakes up.
That waiting time is free evidence: roughly 150 frames of the same face. This
module spends it, keeping only the best handful of crops, so that the moment
engagement fires we can identify the visitor immediately with no extra delay
and no per-frame embedding cost.

Embedding is the expensive step (~10 ms), so it happens once per visit on the
selected crops - never in the capture loop.
"""

import numpy as np

import config
from identity_store import IdentityStore
from recognizer import FaceRecognizer, combine, crop_quality


class VisitRecognizer:
    def __init__(self, recognizer=None, store=None):
        self.recognizer = recognizer or FaceRecognizer()
        self.store = store if store is not None else IdentityStore()
        self._candidates = []   # (quality, crop), best kept
        self._identified = False
        # The provisional guess shown before engagement: the nearest stored
        # person to the best crop so far, refreshed only when a better crop
        # arrives and at most every PROVISIONAL_MATCH_INTERVAL_S.
        self._guess = None              # (identity_id, similarity) or None
        self._guess_quality = 0.0       # quality of the crop the guess used
        self._next_guess_at = None

    def observe(self, frame, face, measurement):
        """Offer one frame as evidence. Cheap; safe to call every frame."""
        if self._identified or not config.RECOGNITION_ENABLED:
            return

        # Cheap rejections first - pose and size need no image work at all.
        yaw = abs(measurement.yaw_deg or 0.0)
        pitch = abs(measurement.pitch_deg or 0.0)
        if yaw > config.RECOGNITION_MAX_YAW_DEG or pitch > config.RECOGNITION_MAX_PITCH_DEG:
            return
        if min(face.w, face.h) < config.RECOGNITION_MIN_FACE_PX:
            return

        crop = self.recognizer.align(frame, face)
        quality = crop_quality(face, measurement, crop)
        if quality <= 0.0:
            return

        self._candidates.append((quality, crop))
        # Keep only the best few so memory stays flat however long they linger.
        if len(self._candidates) > config.RECOGNITION_SAMPLES * 3:
            self._candidates.sort(key=lambda c: c[0], reverse=True)
            del self._candidates[config.RECOGNITION_SAMPLES:]

    @property
    def sample_count(self):
        return len(self._candidates)

    def closest(self, now):
        """Nearest stored person to the best crop so far, as (id, similarity).

        Returns None until there is a usable crop and someone stored to compare
        against. One embedding per refresh, so this is cheap enough to call
        every frame.
        """
        if self._identified or not self._candidates:
            return self._guess
        if self._next_guess_at is not None and now < self._next_guess_at:
            return self._guess
        quality, crop = max(self._candidates, key=lambda c: c[0])
        if quality <= self._guess_quality:
            return self._guess

        self._guess = self.store.closest(self.recognizer.embed(crop))
        self._guess_quality = quality
        self._next_guess_at = now + config.PROVISIONAL_MATCH_INTERVAL_S
        return self._guess

    def identify(self, now=None, new_id=None):
        """Embed the best crops, fuse them, and look the person up.

        A new person is enrolled under `new_id`, so the id clients were shown
        before engagement is the one stored.

        Returns a MatchResult, or None if the visit never produced enough
        usable frames - which is a normal outcome for someone who stood at an
        angle the whole time, and simply means we do not record them.
        """
        if self._identified or not config.RECOGNITION_ENABLED:
            return None
        if len(self._candidates) < config.RECOGNITION_MIN_SAMPLES:
            return None

        self._candidates.sort(key=lambda c: c[0], reverse=True)
        best = self._candidates[:config.RECOGNITION_SAMPLES]

        embeddings = [self.recognizer.embed(crop) for _, crop in best]
        template = combine(embeddings)

        self._identified = True
        return self.store.identify_and_record(template, now=now, new_id=new_id)

    def reset(self):
        """Called when the visitor leaves, so the next person starts clean."""
        self._candidates.clear()
        self._identified = False
        self._guess = None
        self._guess_quality = 0.0
        self._next_guess_at = None
