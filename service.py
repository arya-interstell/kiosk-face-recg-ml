"""WebSocket presence service.

Clients stream encoded video frames in as binary messages; the service streams
JSON events back out as text. It never opens a camera - it has none. The kiosk
(or anything else with a lens) is the sender.

    python service.py                    # ws://0.0.0.0:8765
    KIOSK_AUTH_TOKEN=... python service.py

Roles
-----
A connection declares what it is in its hello. Both roles share one socket
type, and a client that sends binary without saying anything is an ingester
with defaults, so the simplest possible sender is `ws.send(jpeg_bytes)`.

  ingest      sends frames, receives the events derived from them. This is the
              kiosk. Every ingest connection is an independent stream with its
              own dwell timer, own smoothing and own visit evidence, so two
              kiosks on one server never contaminate each other.
  subscriber  sends nothing, receives another stream's events. This is the
              avatar app, which is a separate process - possibly on a separate
              machine, possibly in a different language - from whatever holds
              the camera.

Events
------
Three layers, coarsest first. A client is expected to use one layer and ignore
the rest; they are all published because the right layer differs by consumer.

  person_detected / person_absent
      Raw presence. Any face at all, no gates, no dwell. Cheap to react to,
      and it fires for somebody walking past six metres away, so it is the
      wrong signal for waking an avatar and the right one for a "someone is
      near" indicator.

  candidate_detected / engaged / candidate_lost / disengaged
      The gated state machine: close enough, facing us, and held it for
      ENGAGE_DWELL_SECONDS. `engaged` is the one that wakes the avatar.

  person_recognized
      Emitted immediately after `engaged` when recognition resolved the
      visitor. Carries whether they are returning and every visit timestamp
      held for them. Absent when the visit produced too few usable frames,
      which is a normal outcome and simply means we greet them generically.

  state       periodic telemetry at STATE_BROADCAST_HZ - dwell, distance, pose
  ready       handshake acknowledgement, sent once
  error       a frame we refused, or a control message we did not understand

Every event carries the full state block, so a client may ignore `type`
entirely and react to `engaged` being true.
"""

import argparse
import asyncio
import hmac
import json
import signal
import threading
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import config
import protocol
from detector import FaceDetector
from engagement import State
from identity_store import IdentityStore
from pipeline import KioskPipeline, result_to_payload
from recognizer import FaceRecognizer
from session import VisitRecognizer

# Events a slow client must never miss. Telemetry may be dropped under
# pressure; missing an `engaged` would leave the avatar asleep with somebody
# standing in front of it, and missing a `disengaged` leaves it talking to
# nobody.
TRANSITION_EVENTS = {
    "person_detected", "person_absent", "person_recognized",
    "candidate_detected", "engaged", "candidate_lost", "disengaged",
}

# How often the watchdog checks for streams that have gone quiet.
WATCHDOG_INTERVAL_S = 0.2


# ---------------------------------------------------------------------------
# Shared, process-wide resources
# ---------------------------------------------------------------------------

class SharedRecognizer:
    """One SFace model for the whole process, behind a mutex.

    The model is 37 MB and OpenCV's DNN backend is not safe to call from two
    threads at once, so the obvious per-session copy is both wasteful and
    wrong. Serialising costs nothing here: embedding runs once per visitor,
    not once per frame, so the lock is contended for a few milliseconds every
    time somebody new walks up.
    """

    def __init__(self):
        self._inner = FaceRecognizer()
        self._lock = threading.Lock()

    def align(self, frame, face):
        with self._lock:
            return self._inner.align(frame, face)

    def embed(self, crop):
        with self._lock:
            return self._inner.embed(crop)


class SharedGallery:
    """One identity store for the whole process, behind a mutex.

    IdentityStore already takes a file lock so separate processes cannot
    corrupt the gallery, but sessions in THIS process share one object whose
    in-memory index is rebuilt inside those critical sections. A plain mutex
    closes that gap. Like the recogniser it is touched once per visitor and
    once per purge, never per frame.
    """

    def __init__(self):
        self._store = IdentityStore()
        self._lock = threading.Lock()

    @property
    def path(self):
        return self._store.path

    def identify_and_record(self, template, now=None):
        with self._lock:
            return self._store.identify_and_record(template, now=now)

    def purge(self, now=None):
        with self._lock:
            return self._store.purge(now=now)

    def stats(self):
        with self._lock:
            return self._store.stats()


# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------

class LatestFrame:
    """A one-deep mailbox: a new frame replaces an unprocessed one.

    This is the backpressure policy, and it is deliberately not a queue. If a
    kiosk sends 30 fps and this server can detect at 20, the right thing to
    discard is the oldest frame, every time. A queue would instead build a
    growing backlog and answer questions about where somebody was standing two
    seconds ago, which for a presence detector is worse than not answering.
    """

    def __init__(self):
        self._item = None
        self._ready = asyncio.Event()
        self.dropped = 0
        self.received = 0

    def put(self, item):
        self.received += 1
        if self._item is not None:
            self.dropped += 1
        self._item = item
        self._ready.set()

    async def get(self):
        await self._ready.wait()
        item, self._item = self._item, None
        self._ready.clear()
        return item


def offer(queue, payload, droppable):
    """Put a payload on an outbound queue, applying the drop policy."""
    if queue.full():
        if droppable:
            return
        try:
            queue.get_nowait()      # make room by dropping the oldest
        except asyncio.QueueEmpty:
            pass
    try:
        queue.put_nowait(payload)
    except asyncio.QueueFull:
        pass


class EventHub:
    """Fan-out of one stream's events to its subscribers."""

    def __init__(self):
        self._queues = defaultdict(set)
        self.latest = {}

    def subscribe(self, stream):
        queue = asyncio.Queue(maxsize=config.CLIENT_QUEUE_SIZE)
        self._queues[stream].add(queue)
        return queue

    def unsubscribe(self, stream, queue):
        self._queues[stream].discard(queue)
        if not self._queues[stream]:
            del self._queues[stream]

    def subscriber_count(self, stream):
        return len(self._queues.get(stream, ()))

    def publish(self, stream, payload, droppable=False):
        self.latest[stream] = payload
        for queue in list(self._queues.get(stream, ())):
            offer(queue, payload, droppable)


# ---------------------------------------------------------------------------
# One connection
# ---------------------------------------------------------------------------

class Session:
    """One WebSocket connection and, for an ingester, one video stream."""

    def __init__(self, server, websocket):
        self.server = server
        self.ws = websocket
        self.id = uuid.uuid4().hex[:8]
        self.stream = self.id
        self.role = "ingest"
        self.out = asyncio.Queue(maxsize=config.CLIENT_QUEUE_SIZE)

        self.pipeline = None
        self.slot = LatestFrame()
        self.flip = config.DEFAULT_FLIP_HORIZONTAL
        self.clock = "server"

        # Focal length is a property of the sender's camera at the resolution
        # it sends. Held as a reference pair so it can be rescaled whenever the
        # decoded width changes - which it does the moment INGEST_MAX_WIDTH
        # kicks in on a client streaming 1080p.
        self._ref_focal = config.FOCAL_LENGTH_PX
        self._ref_width = config.CALIBRATION_WIDTH
        self._focal_width = None

        self._present = False           # last raw presence, for edge detection
        self._now = None                # client-clock mode only
        self._prev_client_ts = None
        self._last_frame_at = None
        self._next_state_at = 0.0
        self._frames = 0
        self._detect_total = 0.0
        self._lock = asyncio.Lock()     # serialises pump against watchdog

    # -- handshake --------------------------------------------------------

    def _authorized(self, hello):
        if not config.AUTH_TOKEN:
            return True
        token = ""
        request = getattr(self.ws, "request", None)
        header = getattr(request, "headers", {}).get("Authorization", "") if request else ""
        if header.startswith("Bearer "):
            token = header[len("Bearer "):]
        elif hello:
            token = str(hello.get("token") or "")
        # compare_digest rather than == so a wrong token cannot be recovered a
        # character at a time from how long the comparison took.
        return hmac.compare_digest(token, config.AUTH_TOKEN)

    def _configure(self, hello):
        """Apply a hello. Returns an error string, or None on success."""
        role = str(hello.get("role", "ingest")).lower()
        if role not in ("ingest", "subscriber"):
            return f"unknown role {role!r}"
        self.role = role

        stream = hello.get("stream")
        if stream:
            self.stream = str(stream)[:64]
        elif role == "subscriber":
            return "a subscriber must name the stream it wants"

        if role == "subscriber":
            return None

        self.flip = bool(hello.get("flip", config.DEFAULT_FLIP_HORIZONTAL))

        clock = str(hello.get("clock", "server")).lower()
        if clock not in ("server", "client"):
            return f"unknown clock {clock!r}"
        self.clock = clock

        focal = hello.get("focal_length_px")
        if focal:
            try:
                self._ref_focal = float(focal)
            except (TypeError, ValueError):
                return "focal_length_px must be a number"
            # The width that focal length was measured at. Absent means it
            # applies to the frames being sent, which we learn on frame one.
            self._ref_width = hello.get("frame_width") or None
        return None

    async def _handshake(self):
        """Read the first message. Returns (ok, first_frame_bytes_or_None)."""
        try:
            first = await asyncio.wait_for(
                self.ws.recv(), timeout=config.SESSION_IDLE_TIMEOUT_S)
        except asyncio.TimeoutError:
            await self._reject(4408, "no hello within the idle timeout")
            return False, None
        except Exception:
            return False, None

        hello, first_frame = None, None
        if isinstance(first, (bytes, bytearray)):
            # No hello at all: a bare sender. Defaults apply, and this is
            # already a frame.
            first_frame = bytes(first)
        else:
            try:
                hello = json.loads(first)
            except json.JSONDecodeError:
                await self._reject(4400, "first text message was not JSON")
                return False, None
            if not isinstance(hello, dict):
                await self._reject(4400, "hello must be a JSON object")
                return False, None

        if not self._authorized(hello):
            # Deliberately says nothing about which part was wrong.
            await self._reject(4401, "unauthorized")
            return False, None

        if hello:
            problem = self._configure(hello)
            if problem:
                await self._reject(4400, problem)
                return False, None

        if self.role == "ingest":
            self.pipeline = self.server.build_pipeline(self._ref_focal)
            self.server.hub.latest.setdefault(self.stream, None)

        await self.ws.send(json.dumps(self._ready_payload()))
        return True, first_frame

    def _ready_payload(self):
        return {
            "type": "ready",
            "service": "airport-kiosk-presence",
            "session": self.id,
            "stream": self.stream,
            "role": self.role,
            "accepts": "binary frames: bare JPEG/PNG bytes, or KIF1-framed",
            "engage_dwell_seconds": config.ENGAGE_DWELL_SECONDS,
            "min_face_area_ratio": config.MIN_FACE_AREA_RATIO,
            "max_engage_distance_cm": config.MAX_ENGAGE_DISTANCE_CM,
            "max_yaw_deg": config.MAX_YAW_DEG,
            "max_pitch_deg": config.MAX_PITCH_DEG,
            "focal_length_px": self._ref_focal,
            "calibrated": bool(self._ref_focal),
            "recognition": config.RECOGNITION_ENABLED,
            "retention_hours": config.IDENTITY_RETENTION_HOURS,
            "state_broadcast_hz": config.STATE_BROADCAST_HZ,
            "max_frame_bytes": config.MAX_FRAME_BYTES,
            "current": self.server.hub.latest.get(self.stream),
        }

    async def _reject(self, code, reason):
        try:
            await self.ws.send(json.dumps({"type": "error", "reason": reason}))
            await self.ws.close(code=code, reason=reason[:120])
        except Exception:
            pass

    # -- outbound ---------------------------------------------------------

    def emit(self, payload, droppable=False):
        offer(self.out, payload, droppable)

    async def _writer(self, queue):
        while True:
            payload = await queue.get()
            await self.ws.send(json.dumps(payload))

    # -- inbound ----------------------------------------------------------

    async def _reader(self, first_frame):
        if first_frame is not None:
            self.slot.put(first_frame)
        while True:
            try:
                message = await asyncio.wait_for(
                    self.ws.recv(), timeout=config.SESSION_IDLE_TIMEOUT_S)
            except asyncio.TimeoutError:
                self.emit({"type": "error", "reason": "idle_timeout",
                           "detail": f"no frames for {config.SESSION_IDLE_TIMEOUT_S:.0f}s"})
                await asyncio.sleep(0.1)     # let the writer flush it
                return
            if isinstance(message, (bytes, bytearray)):
                self.slot.put(bytes(message))
            else:
                self._control(message)

    def _control(self, text):
        """Handle a text message on an ingest socket."""
        try:
            message = json.loads(text)
        except json.JSONDecodeError:
            self.emit({"type": "error", "reason": "bad_control_json"})
            return
        kind = str(message.get("type", ""))
        if kind == "ping":
            self.emit({"type": "pong", "echo": message.get("echo")})
        elif kind == "reset":
            # Lets an operator clear a stuck visitor without reconnecting.
            self.pipeline.tracker.reset()
            if self.pipeline.visits is not None:
                self.pipeline.visits.reset()
            self._present = False
            self.emit({"type": "reset_ok"})
        else:
            self.emit({"type": "error", "reason": "unknown_control",
                       "detail": kind or "(no type)"})

    # -- processing -------------------------------------------------------

    def _focal_for(self, frame):
        """Rescale the reference focal length to this frame's decoded width.

        A focal length in pixels is only meaningful alongside the width it was
        measured at: halve the resolution and you halve the focal length. The
        client's own width is `scaled_from` when we downscaled it, otherwise
        the decoded width.
        """
        width = frame.decoded_size[0]
        if width == self._focal_width:
            return self.pipeline.focal_px

        source_width = frame.scaled_from or width
        if self._ref_width is None:
            # Nobody told us; assume the reference applies to what they send.
            self._ref_width = source_width

        focal = self._ref_focal
        if focal and self._ref_width:
            focal = focal * (width / float(self._ref_width))

        self._focal_width = width
        return focal

    def _tick_clock(self, client_ts):
        """The `now` handed to the state machine.

        Server clock by default: real elapsed time is what "stood there for two
        seconds" means, and it lets the watchdog below expire a stream that has
        gone silent. Client clock is opt-in and exists for replaying recorded
        footage faster than real time; each step is clamped so a client cannot
        leap its own dwell timer by sending one frame from the future.
        """
        if self.clock == "server" or client_ts is None:
            return time.monotonic()
        if self._now is None or self._prev_client_ts is None:
            self._now = time.monotonic()
        else:
            step = client_ts - self._prev_client_ts
            self._now += max(0.0, min(step, config.MAX_CLIENT_CLOCK_STEP_S))
        self._prev_client_ts = client_ts
        return self._now

    def _process(self, raw):
        """Decode and analyse one frame. Runs in a worker thread."""
        frame = protocol.decode(raw, flip=self.flip)
        self.pipeline.focal_px = self._focal_for(frame)
        result = self.pipeline.process(frame.image, self._tick_clock(frame.client_ts))
        return frame, result

    async def _pump(self):
        loop = asyncio.get_running_loop()
        while True:
            raw = await self.slot.get()
            async with self._lock:
                try:
                    frame, result = await loop.run_in_executor(
                        self.server.executor, self._process, raw)
                except protocol.FrameError as exc:
                    self.emit({"type": "error", "reason": exc.reason,
                               "detail": exc.detail})
                    continue
                self._last_frame_at = time.monotonic()
                self._publish(result, frame)

    async def _watchdog(self):
        """Expire a stream that stopped sending.

        Every timeout in the engagement machine is evaluated when a frame
        arrives, so a kiosk that unplugs its camera mid-conversation would
        otherwise leave the avatar talking to a frozen `engaged` forever. Once
        frames stop for longer than the grace period, idle ticks are fed in
        until the machine settles back to idle.
        """
        if self.clock != "server":
            return                       # replay mode has no real-time meaning
        grace = max(config.CANDIDATE_GRACE_SECONDS, config.ENGAGED_GRACE_SECONDS)
        while True:
            await asyncio.sleep(WATCHDOG_INTERVAL_S)
            if self._last_frame_at is None:
                continue
            quiet = time.monotonic() - self._last_frame_at
            if quiet < grace:
                continue
            async with self._lock:
                if self.pipeline.tracker.state is State.IDLE and not self._present:
                    continue
                self._publish(self.pipeline.idle(), None, stalled=True)

    # -- publishing -------------------------------------------------------

    def _publish(self, result, frame, stalled=False):
        """Turn one FrameResult into the events its consumers expect."""
        events = []

        # Layer 1: raw presence edges, ungated.
        present = result.measurement.present
        if present != self._present:
            events.append("person_detected" if present else "person_absent")
            self._present = present

        # Layer 2: the gated dwell machine.
        events.extend(result.events)

        # Layer 3: identity, which only ever resolves on the engaged frame.
        if result.identity is not None:
            events.append("person_recognized")

        for kind in events:
            self._send(result, frame, kind, stalled, droppable=False)

        now = time.monotonic()
        if config.STATE_BROADCAST_HZ and now >= self._next_state_at:
            self._send(result, frame, "state", stalled, droppable=True)
            self._next_state_at = now + (1.0 / config.STATE_BROADCAST_HZ)

        if frame is not None:
            self._frames += 1
            self._detect_total += result.detect_ms
            if config.LATENCY_LOG_EVERY and self._frames % config.LATENCY_LOG_EVERY == 0:
                print(f"[{self.stream}] {self._frames} frames  "
                      f"detect avg {self._detect_total / self._frames:.2f} ms  "
                      f"{result.fps:.1f} fps  dropped {self.slot.dropped}  "
                      f"state={result.state}")

    def _send(self, result, frame, kind, stalled, droppable):
        payload = result_to_payload(result, kind=kind)
        payload["stream"] = self.stream
        payload["session"] = self.id
        payload["frame"] = {
            "seq": None if frame is None else frame.seq,
            "bytes": 0 if frame is None else frame.encoded_bytes,
            "width": None if frame is None else frame.decoded_size[0],
            "height": None if frame is None else frame.decoded_size[1],
            "received": self.slot.received,
            "dropped": self.slot.dropped,
            "stalled": stalled,
        }
        self.emit(payload, droppable=droppable)
        self.server.hub.publish(self.stream, payload, droppable=droppable)

        if kind in TRANSITION_EVENTS:
            self._log(kind, result)

    def _log(self, kind, result):
        m = result.measurement
        line = (f"[{self.stream}] {kind:<19} dwell={result.dwell_seconds:5.2f}s "
                f"dist={_fmt(m.distance_cm)}cm yaw={_fmt(m.yaw_deg)}")
        if kind == "person_recognized":
            who = result.identity
            label = "RETURNING" if who.is_returning else "new visitor"
            note = " (ambiguous, enrolled fresh)" if who.ambiguous else ""
            line = (f"[{self.stream}] {kind:<19} {label} {who.identity_id[:8]} "
                    f"visits={who.visit_count} "
                    f"sim={who.similarity:.3f}/{who.runner_up:.3f} "
                    f"{result.recognition_ms:.0f}ms{note}")
        print(line)

    # -- lifecycle --------------------------------------------------------

    async def run(self):
        ok, first_frame = await self._handshake()
        if not ok:
            return

        if self.role == "subscriber":
            queue = self.server.hub.subscribe(self.stream)
            print(f"[{self.stream}] subscriber {self.id} attached "
                  f"({self.server.hub.subscriber_count(self.stream)} total)")
            try:
                # Both of these run forever on their own, so the session ends
                # when the FIRST one returns - which is the drain noticing the
                # client has gone. Waiting for both would hang here until the
                # process exited, holding the subscription and a session slot.
                await self._run_until_one_finishes([
                    self._writer(queue), self._idle_until_closed()])
            finally:
                self.server.hub.unsubscribe(self.stream, queue)
                print(f"[{self.stream}] subscriber {self.id} detached")
            return

        print(f"[{self.stream}] ingest {self.id} connected "
              f"(focal {_fmt(self._ref_focal)}px, clock {self.clock}, "
              f"flip {self.flip})")
        try:
            # The reader is the one that finishes on a clean close; the rest
            # run forever, so the first task to return ends the session.
            await self._run_until_one_finishes([
                self._writer(self.out), self._pump(),
                self._watchdog(), self._reader(first_frame)])
        finally:
            print(f"[{self.stream}] ingest {self.id} closed after "
                  f"{self.slot.received} frames ({self.slot.dropped} dropped)")

    async def _run_until_one_finishes(self, coroutines):
        tasks = [asyncio.create_task(c) for c in coroutines]
        try:
            await _first_to_finish(tasks)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def close(self):
        try:
            await self.ws.close(code=1001, reason="server shutting down")
        except Exception:
            pass

    async def _idle_until_closed(self):
        """A subscriber sends nothing; drain so a close is noticed promptly."""
        while True:
            try:
                await self.ws.recv()
            except Exception:
                return


async def _first_to_finish(tasks):
    done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for task in done:
        exc = task.exception()
        if exc is not None and not isinstance(exc, asyncio.CancelledError):
            from websockets.exceptions import ConnectionClosed
            if not isinstance(exc, ConnectionClosed):
                raise exc


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

class Server:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.hub = EventHub()
        self.executor = ThreadPoolExecutor(
            max_workers=config.WORKER_THREADS, thread_name_prefix="detect")
        self.recognizer = SharedRecognizer() if config.RECOGNITION_ENABLED else None
        self.gallery = SharedGallery() if config.RECOGNITION_ENABLED else None
        self.sessions = set()
        self.stop = asyncio.Event()

    def build_pipeline(self, focal_px):
        """A fresh pipeline per stream, sharing the process-wide models."""
        visits = None
        if config.RECOGNITION_ENABLED:
            visits = VisitRecognizer(recognizer=self.recognizer, store=self.gallery)
        return KioskPipeline(detector=FaceDetector(), visits=visits, focal_px=focal_px)

    async def handler(self, websocket):
        if len(self.sessions) >= config.MAX_SESSIONS:
            await websocket.close(code=4429, reason="too many sessions")
            return
        session = Session(self, websocket)
        self.sessions.add(session)
        try:
            await session.run()
        except Exception as exc:
            from websockets.exceptions import ConnectionClosed
            if not isinstance(exc, ConnectionClosed):
                print(f"[{session.stream}] session failed: {exc!r}")
        finally:
            self.sessions.discard(session)

    async def _purge_loop(self):
        while not self.stop.is_set():
            await asyncio.sleep(config.PURGE_INTERVAL_S)
            if self.gallery is None:
                continue
            removed = await asyncio.to_thread(self.gallery.purge)
            if removed:
                print(f"[store] purged {removed} identities past retention")

    def _banner(self):
        print(f"[service] ws://{self.host}:{self.port}")
        if config.AUTH_TOKEN:
            print("[service] auth: bearer token required")
        else:
            print("[service] WARNING: KIOSK_AUTH_TOKEN is unset - this socket "
                  "accepts face images from anyone who can reach it")
        if config.FOCAL_LENGTH_PX:
            print(f"[service] default calibration: {config.FOCAL_LENGTH_PX:.1f}px "
                  f"at {config.CALIBRATION_WIDTH}px wide "
                  f"(clients may override per session)")
        else:
            print("[service] WARNING: no calibration - distances unavailable, "
                  "gating on face area alone")
        if self.gallery is not None:
            stats = self.gallery.stats()
            print(f"[store] {self.gallery.path}: {stats['identities']} identities, "
                  f"{stats['visits']} visits, "
                  f"{config.IDENTITY_RETENTION_HOURS:.0f}h retention")
        else:
            print("[service] recognition disabled")

    async def run(self):
        from websockets.asyncio.server import serve

        self._banner()
        async with serve(
            self.handler, self.host, self.port,
            max_size=config.MAX_FRAME_BYTES,
            ping_interval=config.PING_INTERVAL_S,
            ping_timeout=config.PING_TIMEOUT_S,
        ):
            purge = asyncio.create_task(self._purge_loop())
            print("[service] running - ctrl-c to stop")
            await self.stop.wait()
            purge.cancel()
            # Hang up on everyone explicitly. Leaving it to the server's own
            # teardown means waiting on handlers that are parked on recv() and
            # have no reason to wake up.
            print(f"[service] closing {len(self.sessions)} session(s)")
            await asyncio.gather(*(s.close() for s in list(self.sessions)),
                                 return_exceptions=True)
        self.executor.shutdown(wait=False, cancel_futures=True)
        print("[service] stopped")


def _fmt(value):
    return "--" if value is None else f"{value:.1f}"


async def _main(args):
    server = Server(args.host, args.port)
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, server.stop.set)
        except NotImplementedError:
            pass
    await server.run()


def main():
    parser = argparse.ArgumentParser(description="Airport kiosk presence service")
    parser.add_argument("--host", default=config.SERVICE_HOST)
    parser.add_argument("--port", type=int, default=config.SERVICE_PORT)
    parser.add_argument("--no-recognition", action="store_true",
                        help="presence and engagement only; store nothing")
    args = parser.parse_args()

    if args.no_recognition:
        config.RECOGNITION_ENABLED = False

    try:
        asyncio.run(_main(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
