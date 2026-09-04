"""Where an astronomer decides: a local page, one candidate at a time.

The pipeline ranks; a person judges; the judgement is recorded under a name,
next to what the model said. This package is the page and the server behind
it, and nothing here runs without a reviewer's name.
"""

from .png import encode_png, stamp_png, stretch
from .queue import LABELS, VettingItem, VettingQueue, build_queue
from .server import VettingServer, VettingSession, serve

__all__ = ["LABELS", "VettingItem", "VettingQueue", "build_queue", "VettingServer",
           "VettingSession", "serve", "encode_png", "stamp_png", "stretch"]
