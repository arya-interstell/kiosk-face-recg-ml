"""Reference sender: push frames up, print events back.

This is the whole client contract in one file. Point it at a webcam, a video
file or an RTSP URL, and it streams JPEG frames to the service and prints the
events that come back.

    python client_example.py                          # default webcam
    python client_example.py --source clip.mp4
    python client_example.py --uri ws://kiosk-server:8765 --token $KIOSK_AUTH_TOKEN
    python client_example.py --subscribe kiosk-3      # listen only, send nothing

The two halves run concurrently and never block each other: the sender keeps
capturing at the camera's own rate regardless of how fast events arrive, and
the service drops whatever it cannot keep up with. That is the intended shape
for any real client - do not send a frame and then wait for its event, because
there is no per-frame reply.
"""

import argparse
import asyncio
import time

import cv2
import websockets

import protocol

# Quality 80 is the knee of the curve for face work: visibly identical to 95
# for our purposes at roughly a third of the bytes.
JPEG_QUALITY = 80


async def send_frames(ws, source, fps, width, quality, seq_start=0):
    """Capture, encode and send. Never waits for a reply."""
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise SystemExit(f"could not open source {source!r}")
    if width:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    # A backed-up capture buffer means we send a face that left seconds ago.
    try:
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except cv2.error:
        pass

    interval = 1.0 / fps if fps else 0.0
    params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    seq = seq_start
    try:
        while True:
            start = time.monotonic()
            ok, frame = capture.read()
            if not ok or frame is None:
                print("[client] source ended")
                return
            ok, encoded = cv2.imencode(".jpg", frame, params)
            if not ok:
                continue
            # The header is optional - `await ws.send(encoded.tobytes())` alone
            # works fine. It is used here so events can name the frame that
            # produced them, which is what makes a recording debuggable.
            await ws.send(protocol.pack(encoded.tobytes(), seq, time.monotonic()))
            seq += 1
            if interval:
                await asyncio.sleep(max(0.0, interval - (time.monotonic() - start)))
            else:
                await asyncio.sleep(0)
    finally:
        capture.release()


async def read_events(ws):
    """React to the events. This is the half a real avatar app cares about."""
    import json

    async for raw in ws:
        message = json.loads(raw)
        kind = message["type"]

        if kind == "ready":
            print(f"[client] connected as stream {message['stream']!r}; "
                  f"engages after {message['engage_dwell_seconds']}s, "
                  f"calibrated={message['calibrated']}")

        elif kind == "person_detected":
            print("   a face is visible")

        elif kind == "person_absent":
            print("   no face in view")

        elif kind == "candidate_detected":
            print("   someone qualified, dwell timer started")

        elif kind == "engaged":
            face = message["face"]
            print(f"-> WAKE AVATAR: user at {face['distance_cm']} cm, "
                  f"yaw {face['yaw_deg']} deg")

        elif kind == "person_recognized":
            who = message["identity"]
            if who["returning"]:
                print(f"-> WELCOME BACK: visit #{who['visit_count']}, "
                      f"last seen {who['timestamps'][-2]}")
            else:
                print("-> FIRST VISIT: greet them fresh")

        elif kind == "disengaged":
            print("-> USER LEFT: stand the avatar down")

        elif kind == "candidate_lost":
            print("   they left before qualifying")

        elif kind == "error":
            print(f"[client] service refused something: {message['reason']} "
                  f"({message.get('detail', '')})")

        elif kind == "state" and message["state"] == "candidate":
            print(f"   dwell {message['dwell_seconds']:.1f}s "
                  f"({message['progress'] * 100:.0f}%)", end="\r")


async def run(args):
    headers = {"Authorization": f"Bearer {args.token}"} if args.token else None

    hello = {"type": "hello"}
    if args.subscribe:
        hello.update(role="subscriber", stream=args.subscribe)
    else:
        hello.update(role="ingest", clock=args.clock)
        if args.stream:
            hello["stream"] = args.stream
        if args.focal:
            hello["focal_length_px"] = args.focal
        if args.flip:
            hello["flip"] = True

    # Reconnects by itself, which is the only sane default for a kiosk that
    # has to survive the server being restarted under it.
    async for ws in websockets.connect(args.uri, additional_headers=headers,
                                       max_size=None):
        try:
            import json
            await ws.send(json.dumps(hello))
            if args.subscribe:
                await read_events(ws)
            else:
                await asyncio.gather(
                    send_frames(ws, args.source, args.fps, args.width, args.quality),
                    read_events(ws),
                )
        except websockets.ConnectionClosed as exc:
            print(f"\n[client] disconnected ({exc.code}), reconnecting...")
            await asyncio.sleep(1.0)
            continue


def main():
    parser = argparse.ArgumentParser(description="Reference kiosk client")
    parser.add_argument("--uri", default="ws://127.0.0.1:8765")
    parser.add_argument("--token", default=None, help="bearer token, if the service requires one")
    parser.add_argument("--source", default="0", help="camera index, video file or stream URL")
    parser.add_argument("--stream", default=None, help="name this stream so subscribers can find it")
    parser.add_argument("--subscribe", default=None, metavar="STREAM",
                        help="listen to another stream's events instead of sending")
    parser.add_argument("--fps", type=float, default=15.0, help="send rate; 0 = as fast as the source")
    parser.add_argument("--width", type=int, default=640, help="capture width")
    parser.add_argument("--quality", type=int, default=JPEG_QUALITY, help="JPEG quality 1-100")
    parser.add_argument("--focal", type=float, default=None,
                        help="this camera's focal length in pixels; overrides the server default")
    parser.add_argument("--flip", action="store_true", help="ask the service to mirror the frames")
    parser.add_argument("--clock", choices=("server", "client"), default="server",
                        help="client = drive the dwell timer from frame timestamps, for replay")
    args = parser.parse_args()

    if args.source.isdigit():
        args.source = int(args.source)

    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
