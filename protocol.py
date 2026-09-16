"""The wire format for inbound frames.

A client sends one WebSocket **binary** message per frame. Two shapes are
accepted and they are told apart by sniffing the first four bytes, so a client
that has nothing to declare can stay as simple as `ws.send(jpeg_bytes)`:

    bare      the whole message is an encoded image (JPEG, PNG, WebP - anything
              cv2.imdecode reads). No metadata.

    framed    a 16-byte little-header, then the encoded image:

                  0..3    b'KIF1'      magic
                  4..7    uint32 BE    sequence number, client's own counter
                  8..15   float64 BE   client timestamp in seconds

              The sequence number lets an event name the frame that caused it,
              which is the only way to line events up against a recording after
              the fact. The timestamp is only consulted in client-clock mode -
              see service.Session - and is otherwise carried through untouched.

Text messages on the same socket are control JSON, never frames; the service
handles those, not this module.

Decoding is deliberately strict. This is the one place where bytes from the
network become an array we run a neural net over, so a frame that is truncated,
mislabelled, absurdly large or not an image at all has to die here with a
diagnosable reason rather than three call frames deeper.
"""

import struct
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

import config

MAGIC = b"KIF1"
HEADER = struct.Struct(">4sId")
HEADER_LEN = HEADER.size          # 16

# Smallest plausible encoded image. Anything shorter is a bug or a probe, and
# imdecode on it is a waste of a thread.
MIN_IMAGE_BYTES = 64


class FrameError(ValueError):
    """An inbound binary message we could not turn into a frame.

    Carries a short machine-readable `reason` because it is reported back to
    the client over the socket, and a client retrying blindly on an
    unrecoverable error is worse than one that logs and stops.
    """

    def __init__(self, reason, detail=""):
        super().__init__(detail or reason)
        self.reason = reason
        self.detail = detail or reason


@dataclass
class InboundFrame:
    """One decoded frame plus whatever the sender told us about it."""

    image: np.ndarray
    seq: Optional[int] = None
    client_ts: Optional[float] = None
    encoded_bytes: int = 0
    decoded_size: tuple = (0, 0)      # (width, height) as decoded
    scaled_from: Optional[int] = None  # original width, if we downscaled


def split(message: bytes):
    """Separate the optional header from the image bytes.

    Returns (payload, seq, client_ts). A bare message yields (message, None, None).
    """
    if len(message) >= HEADER_LEN and message[:4] == MAGIC:
        _, seq, client_ts = HEADER.unpack_from(message, 0)
        # A client with no clock sends 0 rather than omitting the field, since
        # the header is fixed-width. Treat it as absent.
        return message[HEADER_LEN:], int(seq), (float(client_ts) or None)
    return message, None, None


def decode(message: bytes, flip: bool = False,
           max_width: Optional[int] = None) -> InboundFrame:
    """Turn one binary WebSocket message into a BGR frame.

    Raises FrameError with a reason a client can act on.
    """
    if not message:
        raise FrameError("empty_frame")
    if len(message) > config.MAX_FRAME_BYTES:
        raise FrameError("frame_too_large",
                         f"{len(message)} bytes > {config.MAX_FRAME_BYTES}")

    payload, seq, client_ts = split(message)
    if len(payload) < MIN_IMAGE_BYTES:
        raise FrameError("frame_too_small", f"{len(payload)} bytes after header")

    buffer = np.frombuffer(payload, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise FrameError("undecodable",
                         "not a JPEG/PNG/WebP, or truncated mid-image")

    height, width = image.shape[:2]
    original_width = width

    limit = config.INGEST_MAX_WIDTH if max_width is None else max_width
    scaled_from = None
    if limit and width > limit:
        scale = limit / float(width)
        image = cv2.resize(image, (limit, max(1, int(round(height * scale)))),
                           interpolation=cv2.INTER_AREA)
        scaled_from = original_width
        height, width = image.shape[:2]

    if flip:
        image = cv2.flip(image, 1)

    # imdecode hands back a read-only view onto the buffer in some builds, and
    # alignCrop downstream writes into what it is given.
    if not image.flags.writeable:
        image = image.copy()

    return InboundFrame(
        image=image,
        seq=seq,
        client_ts=client_ts,
        encoded_bytes=len(message),
        decoded_size=(width, height),
        scaled_from=scaled_from,
    )


def pack(image_bytes: bytes, seq: int, client_ts: float) -> bytes:
    """Build a framed message. Provided so senders do not re-derive the struct."""
    return HEADER.pack(MAGIC, seq & 0xFFFFFFFF, float(client_ts)) + image_bytes
