"""Where an astronomer decides: a local page, one candidate at a time.

The pipeline ranks; a person judges; the judgement is recorded under a name,
next to what the model said. This package is the page and the server behind
it, and nothing here runs without a reviewer's name.
"""

from .png import encode_png, stamp_png, stretch
from .queue import (LABELS, VettingItem, VettingQueue, build_queue, is_alert_file,
                    queue_for_alert_file, queue_from_alerts)
from .server import VettingServer, VettingSession, serve

__all__ = ["LABELS", "VettingItem", "VettingQueue", "build_queue", "queue_from_alerts",
           "queue_for_alert_file", "is_alert_file", "VettingServer",
           "VettingSession", "serve", "encode_png", "stamp_png", "stretch"]
