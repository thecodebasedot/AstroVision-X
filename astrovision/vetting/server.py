"""A local server that shows candidates and records what people decide.

Standard library only: ``http.server`` for the transport, JSON for the
routes, the hand-written PNG encoder for the stamps. It binds to localhost
by default and has no authentication, because it is a tool for one
astronomer at one desk; put it behind something else before exposing it.

The one rule it enforces is the boundary this whole project keeps: a
verdict without a named reviewer is refused, because an unattributed
decision cannot be told apart from the model's own output and training on
that is self-training. Every accepted verdict is appended to the
active-learning :class:`~astrovision.ml.active.VerdictLog` together with
what the model had said, so the log can later measure where the model and
the people disagreed.
"""

from __future__ import annotations

import json
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

from ..core.logging import get_logger
from ..ml.active import HumanVerdict, VerdictLog
from .page import PAGE
from .png import stamp_png
from .queue import LABELS, VettingItem, VettingQueue

log = get_logger("vetting.server")


class VettingSession:
    """The queue, the log, and the bookkeeping between them."""

    def __init__(self, queue: VettingQueue, log_path: Optional[str] = None,
                 verdict_log: Optional[VerdictLog] = None):
        self.queue = queue
        self.log_path = log_path
        if verdict_log is not None:
            self.log = verdict_log
        elif log_path:
            try:
                self.log = VerdictLog.load(log_path)
            except (OSError, ValueError):
                self.log = VerdictLog()
        else:
            self.log = VerdictLog()
        self._lock = threading.Lock()

    # -- state ---------------------------------------------------------------
    def verdicts_for(self, item: VettingItem) -> List[HumanVerdict]:
        key = item.verdict_key
        return [r for r in self.log.records if int(r.source_id) == key
                and (not r.kind or r.kind == item.kind)]

    def decided_by(self, item: VettingItem, reviewer: str) -> bool:
        return any(r.reviewer == reviewer for r in self.verdicts_for(item))

    def next_item(self, after: int = 0, direction: str = "next",
                  reviewer: str = "") -> Optional[VettingItem]:
        """The next item this reviewer has not decided, in queue order.

        ``prev`` steps back one item regardless of its state, so a decision
        can be revisited (a new verdict is appended, never overwritten).
        """
        items = self.queue.items
        if not items:
            return None
        index = {item.item_id: i for i, item in enumerate(items)}
        start = index.get(int(after), -1)
        if direction == "prev":
            return items[max(start - 1, 0)] if start > 0 else items[0]
        for item in items[start + 1:]:
            if not reviewer or not self.decided_by(item, reviewer):
                return item
        for item in items[:start + 1]:
            if not reviewer or not self.decided_by(item, reviewer):
                return item
        return None

    def record(self, item_id: int, label: str, reviewer: str, note: str = "") -> HumanVerdict:
        item = self.queue.get(item_id)
        if item is None:
            raise KeyError(f"no item {item_id}")
        if label not in LABELS:
            raise ValueError(f"label must be one of {sorted(LABELS)}")
        if not str(reviewer).strip():
            raise ValueError("a verdict needs a reviewer")
        verdict = HumanVerdict(
            source_id=item.verdict_key, label=label, reviewer=str(reviewer).strip(),
            confident=(label != "unsure"), model_label=item.model_label,
            model_confidence=item.model_confidence, model_verdict=item.model_verdict,
            note=str(note or ""), timestamp=time.time(), kind=item.kind,
            candidate_id=item.candidate_id)
        with self._lock:
            self.log.add(verdict)
            if self.log_path:
                self.log.save(self.log_path)
        return verdict

    def progress(self) -> Dict[str, Any]:
        counts = {label: 0 for label in LABELS}
        done = 0
        for item in self.queue.items:
            verdicts = self.verdicts_for(item)
            if verdicts:
                done += 1
                latest = max(verdicts, key=lambda r: r.timestamp)
                counts[latest.label] = counts.get(latest.label, 0) + 1
        return {"n_items": len(self.queue), "n_done": done, "counts": counts,
                "disagreements": len(self.log.disagreements()),
                "agreement_with_model": self.log.agreement_with_model(),
                "n_verdicts": len(self.log)}

    def item_payload(self, item: VettingItem) -> Dict[str, Any]:
        payload = item.to_dict()
        payload["n_items"] = len(self.queue)
        payload["previous"] = [{"reviewer": r.reviewer, "label": r.label,
                                "note": r.note, "timestamp": r.timestamp}
                               for r in self.verdicts_for(item)]
        return payload


def _handler_for(session: VettingSession):
    class Handler(BaseHTTPRequestHandler):
        server_version = "AstroVisionVetting/1"

        def log_message(self, fmt, *args):            # quiet; the CLI logs itself
            log.debug("http %s", fmt % args)

        def _send(self, status: int, body: bytes, content_type: str = "application/json") -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload: Any) -> None:
            self._send(status, json.dumps(payload, default=str).encode("utf-8"))

        def do_GET(self) -> None:                     # noqa: N802 (http.server API)
            url = urlparse(self.path)
            query = {k: v[-1] for k, v in parse_qs(url.query).items()}
            if url.path in ("/", "/index.html"):
                self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            elif url.path == "/api/queue":
                self._json(200, session.queue.summary())
            elif url.path == "/api/progress":
                self._json(200, session.progress())
            elif url.path == "/api/next":
                item = session.next_item(int(query.get("after", 0) or 0),
                                         query.get("direction", "next"),
                                         query.get("reviewer", ""))
                self._json(200, {} if item is None else session.item_payload(item))
            elif url.path.startswith("/api/item/"):
                item = session.queue.get(int(url.path.rsplit("/", 1)[1]))
                if item is None:
                    self._json(404, {"error": "no such item"})
                else:
                    self._json(200, session.item_payload(item))
            elif url.path.startswith("/api/cutout/"):
                name = url.path.rsplit("/", 1)[1]
                item = session.queue.get(int(name.split(".")[0]))
                if item is None or item.stamp is None:
                    self._json(404, {"error": "no cutout"})
                    return
                stamp = (item.stamp_subtracted if query.get("kind") == "subtracted"
                         and item.stamp_subtracted is not None else item.stamp)
                self._send(200, stamp_png(stamp, factor=4), "image/png")
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:                    # noqa: N802
            url = urlparse(self.path)
            length = int(self.headers.get("Content-Length", 0) or 0)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(raw.decode("utf-8") or "{}")
            except ValueError:
                self._json(400, {"error": "body must be JSON"})
                return
            if url.path == "/api/verdict":
                try:
                    verdict = session.record(int(payload.get("item_id", 0)),
                                             str(payload.get("label", "")),
                                             str(payload.get("reviewer", "")),
                                             str(payload.get("note", "")))
                except (KeyError, ValueError) as error:
                    self._send(400, str(error).encode("utf-8"), "text/plain; charset=utf-8")
                    return
                self._json(200, {"recorded": verdict.to_dict(), "progress": session.progress()})
            else:
                self._json(404, {"error": "not found"})

    return Handler


class VettingServer:
    """The HTTP server around a :class:`VettingSession`; start, serve, stop."""

    def __init__(self, session: VettingSession, host: str = "127.0.0.1", port: int = 8765):
        self.session = session
        self.httpd = ThreadingHTTPServer((host, int(port)), _handler_for(session))
        self.httpd.daemon_threads = True
        self._thread: Optional[threading.Thread] = None

    @property
    def url(self) -> str:
        host, port = self.httpd.server_address[:2]
        return f"http://{host}:{port}/"

    def start(self) -> "VettingServer":
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    def serve_forever(self) -> None:
        try:
            self.httpd.serve_forever()
        finally:
            self.httpd.server_close()


def serve(queue: VettingQueue, log_path: Optional[str] = None, host: str = "127.0.0.1",
          port: int = 8765, open_browser: bool = True, block: bool = True
          ) -> VettingServer:
    """Serve the queue; returns the server (already running when ``block`` is False)."""
    session = VettingSession(queue, log_path=log_path)
    server = VettingServer(session, host=host, port=port)
    log.info("vetting %d items at %s (verdicts -> %s)", len(queue), server.url,
             log_path or "memory only")
    if open_browser:
        try:
            webbrowser.open(server.url)
        except Exception:                                  # pragma: no cover
            pass
    if block:
        server.serve_forever()
        return server
    return server.start()
