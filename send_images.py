"""Send still images over the socket, for testing without a camera.

A single picture cannot produce an `engaged` event: the dwell timer needs
ENGAGE_DWELL_SECONDS of continuously qualifying frames, and one message is
instantaneous. So each image is re-sent at a steady rate for long enough to
clear the dwell, which is exactly what a real camera would deliver if the
person stood still.

    python send_images.py face.jpg --token abc123
    python send_images.py face.jpg other.jpg --hold 4 --uri ws://127.0.0.1:9765
    python send_images.py face.jpg --inspect       # what the gates think, no sending
"""

import argparse
import asyncio
import json
import sys
import time

import cv2
import websockets


def load(path, width):
    """Read an image, resize to the target width, and report what we got."""
    image = cv2.imread(path)
    if image is None:
        raise SystemExit(f"could not read {path!r} - is it an image file?")
    h, w = image.shape[:2]
    if width and w != width:
        image = cv2.resize(image, (width, int(round(h * width / w))),
                           interpolation=cv2.INTER_AREA)
    return image


def inspect(image, path):
    """Run the gates locally and say why a picture would or would not qualify."""
    sys.path.insert(0, ".")
    import config
    from detector import FaceDetector
    from geometry import estimate_distance_cm, estimate_head_pose

    h, w = image.shape[:2]
    face = FaceDetector().largest(image)
    print(f"\n{path}  {w}x{h}")
    if face is None:
        print("  NO FACE FOUND - nothing will be detected at all")
        return False

    area = face.area_ratio(w, h)
    pose = estimate_head_pose(face.landmarks, w, h, config.FOCAL_LENGTH_PX)
    dist = estimate_distance_cm(face.landmarks, config.FOCAL_LENGTH_PX,
                                pose[0] if pose else None)

    print(f"  face        {face.w:.0f}x{face.h:.0f} px  (score {face.score:.2f})")
    print(f"  frame area  {area * 100:.2f}%   need >= {config.MIN_FACE_AREA_RATIO * 100:.1f}%"
          f"   {'OK' if area >= config.MIN_FACE_AREA_RATIO else 'TOO SMALL'}")
    if pose:
        yaw, pitch = pose[0], pose[1]
        print(f"  yaw         {yaw:+.1f} deg   need within +/-{config.MAX_YAW_DEG:.0f}"
              f"   {'OK' if abs(yaw) <= config.MAX_YAW_DEG else 'LOOKING AWAY'}")
        print(f"  pitch       {pitch:+.1f} deg   need within +/-{config.MAX_PITCH_DEG:.0f}"
              f"   {'OK' if abs(pitch) <= config.MAX_PITCH_DEG else 'LOOKING AWAY'}")
    if dist is not None:
        ok = config.MIN_ENGAGE_DISTANCE_CM <= dist <= config.MAX_ENGAGE_DISTANCE_CM
        print(f"  distance    {dist:.0f} cm   need {config.MIN_ENGAGE_DISTANCE_CM:.0f}-"
              f"{config.MAX_ENGAGE_DISTANCE_CM:.0f}   {'OK' if ok else 'OUT OF RANGE'}"
              "   (depends on calibration matching your camera)")
    big_enough = min(face.w, face.h) >= config.RECOGNITION_MIN_FACE_PX
    print(f"  recognition {'usable' if big_enough else 'face too small to identify'}"
          f"  (needs >= {config.RECOGNITION_MIN_FACE_PX}px)")
    return True


async def read_events(ws, seen):
    try:
        async for raw in ws:
            msg = json.loads(raw)
            kind = msg["type"]
            if kind == "state":
                continue
            seen.append(kind)
            if kind == "ready":
                print(f"connected  stream={msg['stream']}  calibrated={msg['calibrated']}")
            elif kind == "error":
                print(f"  !! {msg['reason']} {msg.get('detail', '')}")
            elif kind == "person_recognized":
                who = msg["identity"]
                label = "RETURNING" if who["returning"] else "new visitor"
                print(f"  {kind:<20} {label}  {who['id'][:8]}  "
                      f"visit #{who['visit_count']}  sim={who['similarity']:.3f}")
            else:
                face = msg["face"]
                why = ",".join(face["rejected_for"]) or "-"
                print(f"  {kind:<20} dist={face['distance_cm']}  "
                      f"yaw={face['yaw_deg']}  rejected_for={why}")
    except websockets.ConnectionClosed:
        pass


async def run(args):
    images = [(p, load(p, args.width)) for p in args.images]
    params = [cv2.IMWRITE_JPEG_QUALITY, args.quality]

    hello = {"type": "hello", "role": "ingest", "stream": args.stream}
    if args.token:
        hello["token"] = args.token
    if args.focal:
        hello["focal_length_px"] = args.focal

    seen = []
    async with websockets.connect(args.uri, max_size=None) as ws:
        await ws.send(json.dumps(hello))
        reader = asyncio.create_task(read_events(ws, seen))
        await asyncio.sleep(0.3)

        for path, image in images:
            ok, encoded = cv2.imencode(".jpg", image, params)
            if not ok:
                print(f"could not encode {path}")
                continue
            blob = encoded.tobytes()
            print(f"\nsending {path}  ({len(blob) // 1024} KB)  "
                  f"for {args.hold}s at {args.fps} fps")
            end = time.monotonic() + args.hold
            while time.monotonic() < end:
                await ws.send(blob)
                await asyncio.sleep(1.0 / args.fps)

            # Let them "walk away" so the next image is a fresh visitor rather
            # than a continuation of the last one.
            if args.gap and path is not images[-1][0]:
                print(f"pausing {args.gap}s so the state machine resets")
                await asyncio.sleep(args.gap)

        await asyncio.sleep(1.5)
        reader.cancel()

    print("\n--- events seen ---")
    for kind in dict.fromkeys(seen):
        print(f"  {kind}")
    if "engaged" not in seen:
        print("\nno `engaged`: the face was found but did not hold every gate for "
              "the full dwell. Run with --inspect to see which gate failed.")


def main():
    p = argparse.ArgumentParser(description="Send still images to the presence service")
    p.add_argument("images", nargs="+", help="one or more image files")
    p.add_argument("--uri", default="ws://127.0.0.1:9765")
    p.add_argument("--token", default=None)
    p.add_argument("--stream", default="still-test")
    p.add_argument("--hold", type=float, default=4.0,
                   help="seconds to hold each image (must exceed the dwell)")
    p.add_argument("--gap", type=float, default=3.0,
                   help="seconds of nothing between images, to reset the machine")
    p.add_argument("--fps", type=float, default=12.0)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--quality", type=int, default=80)
    p.add_argument("--focal", type=float, default=None,
                   help="your camera's focal length in pixels")
    p.add_argument("--inspect", action="store_true",
                   help="report what the gates think locally, send nothing")
    args = p.parse_args()

    if args.inspect:
        for path in args.images:
            inspect(load(path, args.width), path)
        return

    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
