"""JSON-backed identity gallery, shared between kiosks.

The JSON file is the source of truth. Every kiosk reads it, appends to it, and
picks up other kiosks' enrolments by re-reading when the file changes.

Concurrency
-----------
Several kiosks writing one file is the part that can lose data, so every
read-modify-write runs inside an advisory lock on a companion .lock file, and
the file itself is replaced atomically (write a temp file, then os.replace,
which is atomic on POSIX). The lock lives on a separate file deliberately:
os.replace swaps the inode, so a lock held on the data file itself would be
released to a different file than the one the next writer opens.

That is safe on a local disk and on a properly-configured NFSv4 mount. It is
NOT safe on SMB/CIFS or NFSv3 with locking disabled, where flock is silently a
no-op. If you deploy across terminals on such a share, put a small HTTP service
in front of this class rather than pointing ten kiosks at one network path.

Scale
-----
JSON is comfortable to roughly 5,000 identities (~10 MB, ~100 ms to rewrite).
The retention window keeps you far below that. Beyond it, swap this class for a
SQLite-backed one - nothing outside this file knows the storage format.
"""

import contextlib
import fcntl
import json
import os
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import numpy as np

import config

SCHEMA_VERSION = 1
# Templates are unit vectors; 5 decimals is well past what matching can resolve
# and roughly halves the file size compared with full float repr.
_TEMPLATE_PRECISION = 5


def utcnow():
    return datetime.now(timezone.utc)


def _iso(dt):
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def _parse(text):
    return datetime.fromisoformat(text)


@dataclass
class MatchResult:
    """Outcome of looking one visitor up in the gallery."""

    identity_id: str
    is_returning: bool
    similarity: float = 0.0
    runner_up: float = 0.0
    visit_count: int = 1
    timestamps: List[str] = field(default_factory=list)
    # Set when the top two candidates were too close to call and we enrolled a
    # new identity rather than risk greeting a stranger by someone else's history.
    ambiguous: bool = False


class IdentityStore:
    def __init__(self, path=None, retention_hours=None):
        self.path = path or config.IDENTITY_STORE_PATH
        self.lock_path = self.path + ".lock"
        self.retention_hours = (config.IDENTITY_RETENTION_HOURS
                                if retention_hours is None else retention_hours)
        self._data = {"version": SCHEMA_VERSION, "identities": {}}
        self._mtime = None
        self._matrix = None        # (T, 128) all templates
        self._owner = None         # (T,) index into self._ids
        self._ids = []             # identity id per row of the grouped result
        self.load(force=True)

    # -- locking and io ---------------------------------------------------

    @contextlib.contextmanager
    def _locked(self):
        """Hold an exclusive advisory lock for a read-modify-write."""
        fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def _read_file(self):
        if not os.path.exists(self.path):
            return {"version": SCHEMA_VERSION, "identities": {}}
        try:
            with open(self.path) as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            # A truncated file must not take the kiosk down. Start clean and
            # let the next write repair it; the alternative is a crash loop.
            return {"version": SCHEMA_VERSION, "identities": {}}
        data.setdefault("identities", {})
        return data

    def _write_file(self, data):
        """Atomic replace, so a reader never sees a half-written gallery."""
        data["version"] = SCHEMA_VERSION
        data["updated_at"] = _iso(utcnow())
        directory = os.path.dirname(os.path.abspath(self.path))
        fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as fh:
                json.dump(data, fh, indent=2)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise

    # -- gallery ----------------------------------------------------------

    def load(self, force=False):
        """Re-read the file if another kiosk has touched it."""
        try:
            mtime = os.path.getmtime(self.path)
        except OSError:
            mtime = None
        if not force and mtime == self._mtime:
            return False
        self._data = self._read_file()
        self._mtime = mtime
        self._reindex()
        return True

    def _reindex(self):
        """Flatten every template into one matrix for a single matmul search."""
        rows, owners, ids = [], [], []
        for identity_id, record in self._data["identities"].items():
            index = len(ids)
            ids.append(identity_id)
            for template in record.get("templates", []):
                rows.append(template)
                owners.append(index)
        self._ids = ids
        if rows:
            self._matrix = np.asarray(rows, dtype=np.float32)
            self._owner = np.asarray(owners, dtype=np.int32)
        else:
            self._matrix = None
            self._owner = None

    def _purge_expired(self, data, now):
        """Drop identities last seen outside the retention window.

        This is the privacy policy and the accuracy control in one step: a
        small gallery is what keeps false matches rare.
        """
        cutoff = now - timedelta(hours=self.retention_hours)
        identities = data["identities"]
        expired = [k for k, v in identities.items() if _parse(v["last_seen"]) < cutoff]
        for key in expired:
            del identities[key]
        return len(expired)

    # -- search -----------------------------------------------------------

    def search(self, template):
        """Return (best_id, best_similarity, runner_up_similarity).

        Similarity is taken per identity as the best of that person's stored
        templates, so a match against any one view of them counts.
        """
        if self._matrix is None or not len(self._ids):
            return None, 0.0, 0.0

        sims = self._matrix @ template
        per_identity = np.full(len(self._ids), -1.0, dtype=np.float32)
        np.maximum.at(per_identity, self._owner, sims)

        best = int(np.argmax(per_identity))
        best_score = float(per_identity[best])

        runner_up = 0.0
        if len(per_identity) > 1:
            others = np.delete(per_identity, best)
            runner_up = float(np.max(others))

        return self._ids[best], best_score, runner_up

    # -- the main operation -----------------------------------------------

    def identify_and_record(self, template, now=None) -> MatchResult:
        """Match a visit template, then either append a timestamp or enrol.

        The whole read-modify-write is inside the lock so two kiosks seeing the
        same person at the same moment cannot both create an identity for them.
        """
        now = now or utcnow()
        template = np.asarray(template, dtype=np.float32)

        with self._locked():
            data = self._read_file()
            self._data = data
            self._purge_expired(data, now)
            self._reindex()

            best_id, best, runner_up = self.search(template)

            matched = (best_id is not None
                       and best >= config.MATCH_THRESHOLD
                       and (best - runner_up) >= config.MATCH_MARGIN)
            ambiguous = (best_id is not None
                         and best >= config.MATCH_THRESHOLD
                         and not matched)

            if matched:
                record = data["identities"][best_id]
                result = self._append_visit(record, best_id, template, now, best, runner_up)
            else:
                result = self._enrol(data, template, now, best, runner_up, ambiguous)

            self._write_file(data)
            self._mtime = os.path.getmtime(self.path)
            self._reindex()

        return result

    def _append_visit(self, record, identity_id, template, now, best, runner_up):
        last_seen = _parse(record["last_seen"])
        elapsed = (now - last_seen).total_seconds()

        # Kiosks share this file but not a clock. If another terminal's clock
        # runs ahead, its timestamps look like the future from here; elapsed
        # goes negative and must count as the same visit rather than a new one,
        # or a skewed pair of kiosks would invent visits for anyone they both
        # see. Keep NTP on the terminals; this only stops skew corrupting data.
        same_visit = elapsed < config.REVISIT_WINDOW_SECONDS

        if not same_visit:
            record["visits"].append(_iso(now))
        # Never let last_seen move backwards, or retention would expire someone
        # early just because a lagging kiosk saw them most recently.
        record["last_seen"] = _iso(max(now, last_seen))

        self._maybe_add_template(record, template)

        return MatchResult(
            identity_id=identity_id,
            is_returning=not same_visit,
            similarity=best,
            runner_up=runner_up,
            visit_count=len(record["visits"]),
            timestamps=list(record["visits"]),
        )

    def _maybe_add_template(self, record, template):
        """Keep a few genuinely different views of a person, not near-copies."""
        existing = np.asarray(record["templates"], dtype=np.float32)
        if len(existing) and float(np.max(existing @ template)) > config.TEMPLATE_NOVELTY_THRESHOLD:
            return
        record["templates"].append(_round(template))
        if len(record["templates"]) > config.MAX_TEMPLATES_PER_IDENTITY:
            # Drop the oldest; recent views track the person's current look.
            record["templates"].pop(0)

    def _enrol(self, data, template, now, best, runner_up, ambiguous):
        identity_id = str(uuid.uuid4())
        data["identities"][identity_id] = {
            "first_seen": _iso(now),
            "last_seen": _iso(now),
            "visits": [_iso(now)],
            "templates": [_round(template)],
        }
        return MatchResult(
            identity_id=identity_id,
            is_returning=False,
            similarity=best,
            runner_up=runner_up,
            visit_count=1,
            timestamps=[_iso(now)],
            ambiguous=ambiguous,
        )

    # -- housekeeping -----------------------------------------------------

    def purge(self, now=None):
        """Apply the retention policy. Call periodically, or on startup."""
        now = now or utcnow()
        with self._locked():
            data = self._read_file()
            removed = self._purge_expired(data, now)
            if removed:
                self._write_file(data)
                self._mtime = os.path.getmtime(self.path)
            self._data = data
            self._reindex()
        return removed

    def forget(self, identity_id):
        """Erase one person. Accepts a unique id prefix.

        A biometric store needs a delete path, and it needs to exist before
        somebody asks to be removed rather than after. Returns the full id that
        was erased, or None if the prefix matched nothing or was ambiguous -
        never a guess, because erasing the wrong traveller is unrecoverable.
        """
        with self._locked():
            data = self._read_file()
            matches = [k for k in data["identities"] if k.startswith(identity_id)]
            if len(matches) != 1:
                return None
            del data["identities"][matches[0]]
            self._write_file(data)
            self._data = data
            self._mtime = os.path.getmtime(self.path)
            self._reindex()
        return matches[0]

    def wipe(self):
        """Erase the whole gallery. Returns how many identities went."""
        with self._locked():
            data = self._read_file()
            count = len(data["identities"])
            self._write_file({"identities": {}})
            self._data = {"version": SCHEMA_VERSION, "identities": {}}
            self._mtime = os.path.getmtime(self.path)
            self._reindex()
        return count

    def stats(self):
        identities = self._data["identities"]
        visits = sum(len(v["visits"]) for v in identities.values())
        return {
            "identities": len(identities),
            "visits": visits,
            "returning": sum(1 for v in identities.values() if len(v["visits"]) > 1),
            "templates": 0 if self._matrix is None else int(self._matrix.shape[0]),
        }

    def get(self, identity_id):
        return self._data["identities"].get(identity_id)


def _round(vector):
    return [round(float(x), _TEMPLATE_PRECISION) for x in vector]
