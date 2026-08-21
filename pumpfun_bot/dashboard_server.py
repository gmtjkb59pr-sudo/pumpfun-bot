"""
Lokale webserver voor het dashboard. Gebruikt alleen Python's ingebouwde
http.server - geen extra dependency nodig. Luistert standaard alleen op
127.0.0.1 (niet bereikbaar van buiten je eigen machine).
"""
from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .activity_log import DATA_LOG_PATH
from .state import bot_state
from .stats import compute_stats

logger = logging.getLogger("pumpfun_bot.dashboard")

STATIC_DIR = Path(__file__).parent / "static"


class DashboardHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:  # noqa: A002
        pass  # onderdrukt de standaard access-log ruis, we loggen zelf al genoeg

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/", "/index.html"):
            self._serve_file(STATIC_DIR / "dashboard.html", "text/html; charset=utf-8")
        elif self.path == "/api/status":
            self._serve_json(bot_state.snapshot())
        elif self.path == "/api/stats":
            self._serve_json(compute_stats(DATA_LOG_PATH))
        else:
            self.send_error(404, "Niet gevonden")

    def _serve_file(self, path: Path, content_type: str) -> None:
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            self.send_error(404, "Dashboard bestand ontbreekt")
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_json(self, data: dict) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def start_dashboard_server(port: int = 8765) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", port), DashboardHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True, name="dashboard-server")
    thread.start()
    logger.info("Dashboard gestart op http://127.0.0.1:%d", port)
    return server
