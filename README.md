# Airport kiosk presence service

A WebSocket service that takes a **stream of binary image frames in** and
streams **text events out**: someone is there, someone is standing at the kiosk
and paying attention, we have seen them before.

The server has no camera. Kiosks push frames to it, which means the vision
model is deployed and updated in one place rather than on every terminal.

```
  kiosk (or browser, or phone, or a recorded file)
      │  binary WebSocket messages, one JPEG per frame
      ▼
┌─────────────────────────────────────────────┐
│  service.py                                  │
│                                              │
│   decode ─► find a face ─► how big?          │  ~4 ms
│                            how far?          │
│                            facing us?        │
│                                  │           │
│                            ┌─────▼───────┐   │
│                            │ dwell timer │   │
│                            └─────┬───────┘   │
│                                  │           │
│                            recognise once    │  ~70 ms, per visitor
└──────────────────────────────────┼───────────┘
      │  JSON text messages         │
      ▼                             ▼
  the sender                   any subscriber
                               (the avatar app)
```

A person counts as **engaged** when, continuously for `ENGAGE_DWELL_SECONDS`,
their face is large enough in frame (`MIN_FACE_AREA_RATIO`), within
`MIN/MAX_ENGAGE_DISTANCE_CM` of the lens, and pointed at the kiosk
(`MAX_YAW_DEG` / `MAX_PITCH_DEG`). Every one of those numbers is in `config.py`.

## Install and run

```sh
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
sh models/fetch.sh                      # the two ONNX models, ~37 MB

export KIOSK_AUTH_TOKEN=$(openssl rand -hex 32)
python service.py                       # ws://0.0.0.0:8765
```

Then, from anywhere that has a camera:

```sh
python client_example.py --uri ws://your-server:8765 --token "$KIOSK_AUTH_TOKEN"
python client_example.py --subscribe kiosk-3   # listen only, send nothing
```

`client_example.py` is the whole client contract in one readable file. Read it
before writing your own.

### Testing with still images

No camera needed, and useful for checking a photograph against the gates before
blaming the service:

```sh
python send_images.py face.jpg --inspect        # what the gates think, sends nothing
python send_images.py face.jpg --token "$KIOSK_AUTH_TOKEN"
```

A single picture can never produce `engaged` - the dwell timer needs
`ENGAGE_DWELL_SECONDS` of *continuously* qualifying frames and one message is
instantaneous. So `send_images.py` re-sends each image at a steady rate for
`--hold` seconds, which is what a real camera would deliver if the person stood
still.

## The protocol

One duplex socket. **Binary messages in, text messages out.** Text messages
*in* are control, never frames.

### Connecting

Send a hello as the first text message:

```json
{"type": "hello", "role": "ingest", "stream": "kiosk-3",
 "focal_length_px": 1286.75, "flip": false, "clock": "server"}
```

| field | default | meaning |
|---|---|---|
| `role` | `ingest` | `ingest` sends frames; `subscriber` only listens |
| `stream` | random | names this stream so subscribers can find it |
| `token` | — | auth, if you cannot set an `Authorization` header |
| `focal_length_px` | from `calibration.json` | **your** camera's focal length — see *Calibration* |
| `frame_width` | width of frame 1 | the width that focal length was measured at |
| `flip` | `false` | mirror frames horizontally |
| `clock` | `server` | `client` drives the dwell timer from frame timestamps, for replay |

The hello is optional. A client that just sends binary is an ingester with
defaults — `ws.send(jpeg_bytes)` is a complete, working client. If
`KIOSK_AUTH_TOKEN` is set, the hello (or the header) is required.

The server replies once with `ready`, carrying the thresholds in force and the
current state of the stream.

### Sending frames

One binary message per frame, either shape — the server sniffs which:

```
bare      the whole message is an encoded image (JPEG, PNG, WebP)

framed    b'KIF1' + uint32 BE sequence + float64 BE timestamp + the image
          protocol.pack(jpeg_bytes, seq, time.monotonic())
```

The sequence number is echoed in every event as `frame.seq`, which is the only
way to line events up against a recording afterwards.

**Do not send a frame and wait for its reply** — there is no per-frame reply.
Send continuously; read continuously. The two halves are independent.

### Events out

Three layers, coarsest first. Use one and ignore the rest.

| type | meaning |
|---|---|
| `ready` | handshake ack: thresholds, calibration, current state |
| `person_detected` | **a face is visible.** No gates, no dwell — fires for someone six metres away |
| `person_absent` | **no face is visible** |
| `candidate_detected` | someone passed every gate; the dwell timer started |
| `engaged` | they held it for the full dwell — **wake the avatar** |
| `person_recognized` | who they are; follows `engaged` immediately |
| `candidate_lost` | they left before qualifying |
| `disengaged` | an engaged user left — stand the avatar down |
| `state` | telemetry at `STATE_BROADCAST_HZ` (distance, pose, dwell, fps) |
| `error` | a frame we refused, or a control message we did not understand |

Every message carries the whole state block, so a simple client can ignore
`type` entirely and watch the `engaged` boolean.

```json
{
  "type": "engaged", "state": "engaged", "engaged": true,
  "person_id": "c1007bf7-...", "visit_id": "e5288113a0a8",
  "stream": "kiosk-3", "session": "aef0f36d",
  "dwell_seconds": 2.01, "progress": 1.0, "calibrated": true,
  "face": {"present": true, "distance_cm": 70.0, "yaw_deg": 0.0,
           "pitch_deg": -0.2, "area_ratio": 0.0971, "rejected_for": []},
  "perf": {"detect_ms": 5.1, "total_ms": 5.4, "fps": 30.0},
  "frame": {"seq": 412, "bytes": 51204, "width": 640, "height": 480,
            "received": 415, "dropped": 3, "stalled": false},
  "identity": null
}
```

`rejected_for` says *why* a visible face did not qualify (`too_far`,
`too_small`, `looking_away_yaw`, …). It is the fastest way to tune thresholds
on site. `person_recognized` carries the same block with `identity` filled in:

```json
{"id": "c1007bf7-...", "returning": true, "visit_count": 3,
 "timestamps": ["2026-09-07T08:53:29+00:00", "..."],
 "similarity": 0.81, "runner_up": 0.12, "ambiguous": false}
```

`identity` is absent when the visit never yielded enough clean frames, which is
a normal outcome — greet them generically.

### Identity lasts the whole visit, not one message

Recognition resolves on exactly one frame — the one engagement fires on — but it
describes the person, not that frame. So once it resolves, every later message
in the visit carries it too:

| field | present | meaning |
|---|---|---|
| `person_id` | always, `null` until known | the identified person, flat and null-safe |
| `visit_id` | from `candidate_detected` | this stay at the kiosk, minted when the dwell starts |
| `identity` | from `engaged` onwards | the full block: returning, visit count, history, scores |

Both clear when the visitor leaves — but not before `disengaged` is sent, so
that event still names who it was that left. `visit_id` is minted when the dwell
timer starts rather than at identification, so the events that arrive *before*
recognition can still be grouped with the ones after it.

Without this a client has to catch the single `engaged` message and correlate
every later event back to it by hand, which is the kind of bookkeeping that
works until a reconnect.

### Control messages in

`{"type": "ping", "echo": 1}` → `pong`. `{"type": "reset"}` clears a stuck
visitor without reconnecting.

## Backpressure: frames are dropped, never queued

The inbound mailbox is **one frame deep**. If a kiosk sends 30 fps and the
server detects at 20, the oldest frame is discarded every time.

This is deliberate. A queue would build a backlog and start answering questions
about where somebody was standing two seconds ago, which for a presence
detector is worse than not answering. `frame.received` and `frame.dropped` in
every event tell you exactly how much is being shed — measured here, 565 frames
offered in two seconds, 528 dropped, engagement still correct.

Outbound events are queued (`CLIENT_QUEUE_SIZE`) and `state` telemetry is
dropped under pressure, but **transitions never are**: missing an `engaged`
would leave the avatar asleep with someone standing in front of it.

If frames stop arriving entirely, a watchdog winds the state machine down
anyway. Every timeout is evaluated on frame arrival, so without it a kiosk that
unplugged its camera mid-conversation would leave the avatar talking to a
frozen `engaged` forever. Those events carry `frame.stalled: true`.

## Multiple kiosks

Every ingest connection is an independent stream with its own dwell timer, own
smoothing and own visit evidence, so two kiosks on one server never contaminate
each other. `MAX_SESSIONS` caps them; `WORKER_THREADS` sizes the detection pool
(OpenCV releases the GIL, so these genuinely run in parallel).

What *is* shared, deliberately: one SFace model behind a mutex (37 MB, and
OpenCV's DNN backend is not thread-safe), and one identity gallery, so a
traveller recognised at gate 4 is recognised at gate 9.

## Calibration

Distance in centimetres needs one number that cannot be guessed: the focal
length **of the camera that took the picture**, in pixels. That is a property
of the sender, not of this server.

`calibration.json` holds the default and is loaded automatically. A client
streaming from a different camera should send its own `focal_length_px` at
handshake, which overrides it for that session only.

A focal length in pixels is tied to the resolution it was measured at — halve
the resolution and you halve the focal length. The service rescales
automatically, including when `INGEST_MAX_WIDTH` downscales a 1080p sender, as
long as it knows the reference width (`frame_width`, or `calibration.json`'s).

To measure one: have someone stand a known distance `D` cm from the lens
looking straight ahead, read the pixel gap between their pupils `ipd_px` from a
`state` event's landmarks, and

```
focal_length_px = ipd_px * (D * 10 + 27) / 63
```

where 63 mm is the mean adult inter-pupillary distance and 27 mm is how far the
eye plane sits behind the nose tip.

Skipping calibration is survivable: the service still runs and still gates on
face area and head pose, it just reports `distance_cm: null` and warns at
startup.

## Deployment

The socket accepts face images and writes biometric templates. Three things
before it faces anything but localhost:

1. **Set `KIOSK_AUTH_TOKEN`.** Without it the service warns at startup and
   accepts frames from anyone who can reach the port. Clients send it as
   `Authorization: Bearer <token>` or as `token` in the hello. Comparison is
   constant-time.
2. **Terminate TLS in front of it** (nginx, Caddy, a load balancer) so kiosks
   connect to `wss://`, not `ws://`. This service speaks plain WebSocket by
   design — it does not hold your certificates.
3. **Firewall the port** to the kiosks' addresses. A bearer token is not a
   substitute for not being reachable.

Environment variables: `KIOSK_HOST`, `KIOSK_PORT`, `KIOSK_AUTH_TOKEN`,
`KIOSK_IDENTITY_STORE`.

```ini
# /etc/systemd/system/kiosk-presence.service
[Service]
WorkingDirectory=/opt/airport_kiosk
Environment=KIOSK_AUTH_TOKEN=...
ExecStart=/opt/airport_kiosk/.venv/bin/python service.py
Restart=always
```

Put `identities.json` on a local disk. `flock` is a silent no-op on SMB/CIFS
and on NFSv3 without locking, and the gallery's write safety depends on it.

## Why not YOLO

The job is one large, near-field, frontal face at about a metre. That is the
easiest case in face detection, and YOLO's strengths — small objects, crowded
scenes, many classes — buy nothing here while costing an order of magnitude
more latency.

| | YuNet (used here) | YOLOv8n-face |
|---|---|---|
| Model | 232 KB ONNX, inside OpenCV | ~6 MB |
| Deps | opencv-python | torch + ultralytics (GBs) |
| Latency | ~4–8 ms CPU @320px | ~20–30 ms CPU |
| Licence | BSD-3-Clause | **AGPL-3.0** |

That licence row matters for a commercial airport deployment and is worth
raising before anyone builds on Ultralytics.

Distance comes from the gap between the pupils, ~63 mm on virtually every
adult — nature's built-in ruler. Facing direction comes from the same five
landmarks YuNet already returns, so "is he looking at it" costs microseconds
rather than a second model.

## How it behaves

The tuning that makes this feel right is in `engagement.py`, not the detector:

* **Hysteresis** — losing engagement is harder than gaining it, so someone
  standing on a threshold does not flicker the avatar on and off.
* **Grace** — a blink, a glance at a departure board, or a passer-by crossing
  the camera does not reset the dwell timer.
* **Cooldown** — after a disengagement the kiosk stays quiet briefly, so it
  does not re-greet someone already walking away.

## Recognising returning visitors

When someone engages, the service works out whether it has seen them before,
gives them a UUID, and appends a timestamp to their visit list. Same person
again — same UUID, one more timestamp.

```
        ENGAGED  (the existing dwell — unchanged)
            │
            ▼
   collect ~10 crops        sharpest and most frontal frames only,
   while the timer runs     gathered during the countdown so identification
            │               costs no extra waiting
            ▼
   SFace → 128 numbers      ~10 ms each, median of them = this visit
            │
            ▼
   search the gallery       one matmul; 0.08 ms at 10,000 identities
            │
     ┌──────┴────────┬──────────────────┐
     ▼               ▼                  ▼
 clear match     too close to call    no match
     │               │                  │
     ▼               ▼                  ▼
 append           enrol new           enrol new
 timestamp        (never guess)       identity
```

**Model: SFace**, via `cv2.FaceRecognizerSF` — already inside OpenCV, so it
adds no dependency. Its `alignCrop()` consumes YuNet's output row directly; the
detector and recogniser were designed as a pair. Roughly **70 ms** for a whole
visit, paid **once per visitor** rather than per frame.

### Why the threshold is stricter than the textbook one

OpenCV documents 0.363 as SFace's 1:1 threshold — for answering "are these two
photos the same person?". We ask a harder question: "is this *any* of the N
people we know?" Every new traveller is compared against the whole gallery, so
the chance of a false match grows with N. If one comparison has false-match
rate `f`, a genuinely new person is wrongly recognised with probability
`1 − (1−f)^N`:

| gallery | f = 10⁻³ | f = 10⁻⁴ | f = 10⁻⁵ |
|---|---|---|---|
| 100 | 9.5% | 1.0% | 0.1% |
| 1,000 | **63%** | 9.5% | 1.0% |
| 10,000 | ~100% | **63%** | 9.5% |

Three things keep this under control:

* **A stricter threshold** (`MATCH_THRESHOLD`, default 0.45).
* **A margin test** (`MATCH_MARGIN`) — the best candidate must beat the
  runner-up clearly. Two close candidates means ambiguity, and we enrol a new
  identity rather than guess. The errors are not symmetric: a spare row costs
  nothing, greeting a stranger with someone else's history is the incident.
* **The retention window**, which is the real control. Purging after 48 hours
  keeps the gallery in the hundreds — the comfortable row of that table — *and*
  gives you a defensible privacy policy. **The privacy fix and the accuracy fix
  are the same fix.** Keeping data forever makes the system both less lawful
  and less accurate.

### Calibrate the threshold on your own footage

Measured on two sample faces, the same person under degradation scored
**0.67–1.00**, two different people **0.19** — a wide gap either side of 0.45:

| condition | similarity |
|---|---|
| identical frame | 1.00 |
| brighter / darker | 0.80 / 0.94 |
| blurred / heavily blurred | 0.94 / 0.79 |
| half / quarter resolution | 0.94 / 0.67 |
| JPEG quality 30 | 0.93 |
| **a different person** | **0.19** |

That is two people, not a calibration. Before going live, record real footage
at the kiosk, run known repeat visitors through it, and check the gap holds
with your camera and lighting. Every `person_recognized` event reports
`similarity` and `runner_up`, which is the data you need.

**Note that JPEG compression is now in that path.** The default quality of 80
is comfortably inside the range above, but if you lower it to save bandwidth,
re-measure. Compression artefacts degrade an embedding before they degrade
anything a human would notice.

### Storage

`identities.json` is the source of truth:

```json
{ "version": 1,
  "identities": {
    "c1007bf7-...": { "first_seen": "...", "last_seen": "...",
                      "visits": ["2026-09-07T08:53:29+00:00", "..."],
                      "templates": [[128 numbers], ...] } } }
```

Each person keeps up to `MAX_TEMPLATES_PER_IDENTITY` genuinely different views
(glasses, lighting, angle) and matches against their best one. Lingering at the
kiosk is **one** visit, not forty — `REVISIT_WINDOW_SECONDS` collapses repeats.

Writes are atomic (temp file plus `os.replace`) and serialised by an advisory
lock on a companion `.lock` file. JSON is comfortable to roughly 5,000
identities; the retention window keeps you far below that. Beyond it, swap
`IdentityStore` for a SQLite version — nothing outside that file knows the
storage format.

### Erasing someone

```sh
python -c "from identity_store import IdentityStore; print(IdentityStore().forget('c1007bf7'))"
python -c "from identity_store import IdentityStore; print(IdentityStore().wipe(), 'erased')"
```

`forget` takes a unique id prefix and returns the full id, or `None` if the
prefix matched nothing or was ambiguous — it will never guess. Both take the
same file lock as the service, so they are safe to run while it is up, and the
running service picks up the deletion on its next identification.

This delete path needs to exist before somebody asks to be removed, not after.

## Known limits

* **Head pose, not eye gaze.** We know the head points at the kiosk, not that
  the eyes do. At kiosk range that is the right proxy; if you later need true
  gaze, add MediaPipe FaceLandmarker for iris landmarks and run it *only* on
  the face that already passed the size gate.
* **Distance assumes an average face.** `IPD_MM = 63` is the adult mean. Adult
  IPD spans roughly 55–72 mm, so an individual can read up to ~12% near or far.
  Children read further away than they are. The gates are bands, not precision
  measurements, and are wide enough to absorb this.
* **Distance assumes an honest sender.** The focal length comes from the client
  and is not verifiable from the pixels. A wrong one gives confidently wrong
  centimetres. The face-area gate does not depend on it.
* **One person at a time.** By design: the kiosk talks to whoever's face is
  largest. A person walking between the user and the camera can briefly steal
  the "largest face" slot; the grace window absorbs that.
* **Images are never stored — but face templates are.** Frames are decoded,
  analysed and dropped; no picture is ever written to disk. What *is* written
  is a 128-number template per person, which is biometric data under GDPR and
  the DPDP Act. It is pseudonymous, not anonymous. See *Recognising returning
  visitors* for the retention and deletion story, and do not deploy this to a
  live terminal without signage and a lawful basis.
* **Frames now cross a network.** They are face images in transit. Terminate
  TLS in front of this service; see *Deployment*.
