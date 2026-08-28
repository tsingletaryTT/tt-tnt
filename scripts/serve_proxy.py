#!/usr/bin/env python3
# scripts/serve_proxy.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Transparent reverse proxy in front of the served vLLM backend -- the client-facing
half of this project's crash-proofing pass (`scripts/serve_supervisor.py` is the other
half). Point Open WebUI (or any OpenAI-compatible client) at THIS proxy's port instead
of the backend's directly.

Two things it does that a client should never have to know about:

1. **Proactive restart signalling.** The known engine-death bug
   (`docs/serving-with-tt-kernel.md` sec.7) fires after roughly 18-20 completion
   requests, cumulative, regardless of prompt length. Rather than wait for that crash
   and let `serve_supervisor.py` react to it, this proxy counts every request THROUGH
   ITSELF and drops a flag file well under that threshold (default 10). The
   supervisor checks for that flag on each health-poll and cycles the server
   proactively -- a scheduled, low-impact restart instead of an unpredictable crash.
2. **Transparent retry across a restart.** If the backend is down (mid-restart, or
   still starting up), this proxy does NOT immediately return an error to the
   client -- it polls until the backend is healthy again (bounded by
   `--retry-timeout`) and then forwards the original request. From the client's
   point of view, a restart is added latency, not a failed request.

Concurrency is not something this proxy has to implement specially: it is a
multi-threaded server (`ThreadingHTTPServer`), so N simultaneous client connections
run in N threads, each independently forwarding to and blocking on the backend --
which is itself already an async, continuously-batching engine designed for exactly
this. The proxy's only shared, thread-sensitive state is the request counter, guarded
by a lock.

    python3 scripts/serve_proxy.py --backend-port 8003 --listen-port 8004
"""

from __future__ import annotations

import argparse
import http.client
import http.server
import socketserver
import threading
import time
import urllib.error
import urllib.request

DEFAULT_BACKEND_PORT = 8003
DEFAULT_LISTEN_PORT = 8004
DEFAULT_PROACTIVE_RESTART_EVERY = 10
DEFAULT_RETRY_TIMEOUT_S = 120.0
RETRY_POLL_INTERVAL_S = 3.0
RESTART_FLAG_PATH = "/tmp/tt-tnt-serve-restart-requested"
HEALTH_CHECK_TIMEOUT_S = 3.0


def log(msg: str) -> None:
    import datetime

    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    print(f"[{ts}] {msg}", flush=True)


def backend_is_healthy(backend_port: int) -> bool:
    try:
        urllib.request.urlopen(
            f"http://localhost:{backend_port}/v1/models", timeout=HEALTH_CHECK_TIMEOUT_S
        )
        return True
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


class RequestCounter:
    """Thread-safe counter that drops the proactive-restart flag every N requests.

    A plain `threading.Lock`, not anything fancier: this proxy has exactly one
    piece of shared mutable state, and a lock held for the few instructions it
    takes to increment-and-maybe-reset an int is not a concurrency bottleneck worth
    engineering around.
    """

    def __init__(self, restart_every: int):
        self._count = 0
        self._restart_every = restart_every
        self._lock = threading.Lock()

    def increment_and_maybe_flag(self) -> None:
        with self._lock:
            self._count += 1
            if self._count >= self._restart_every:
                self._count = 0
                self._write_flag()

    @staticmethod
    def _write_flag() -> None:
        with open(RESTART_FLAG_PATH, "w") as f:
            f.write("requested by serve_proxy.py at count threshold\n")
        log(f"wrote {RESTART_FLAG_PATH} -- proactive restart requested before the "
            f"known bug's threshold, not in reaction to it")


class ProxyHandler(http.server.BaseHTTPRequestHandler):
    backend_port: int = DEFAULT_BACKEND_PORT
    retry_timeout_s: float = DEFAULT_RETRY_TIMEOUT_S
    counter: RequestCounter | None = None

    #: Paths that actually trigger a prefill/decode pass on the backend -- the
    #: request volume the ~18-20-request crash threshold is measured against.
    #: Everything else (/v1/models, /health, /metrics) is metadata traffic.
    _GENERATION_PATHS = ("/v1/chat/completions", "/v1/completions")

    def _is_generation_request(self) -> bool:
        return self.command == "POST" and self.path.split("?", 1)[0] in self._GENERATION_PATHS

    def _proxy(self) -> None:
        body = self.rfile.read(int(self.headers.get("Content-Length", 0) or 0))

        deadline = time.monotonic() + self.retry_timeout_s
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                conn = http.client.HTTPConnection("localhost", self.backend_port, timeout=90)
                headers = {k: v for k, v in self.headers.items() if k.lower() != "host"}
                conn.request(self.command, self.path, body=body, headers=headers)
                resp = conn.getresponse()
                resp_body = resp.read()
                self.send_response(resp.status)
                for k, v in resp.headers.items():
                    if k.lower() in ("transfer-encoding", "connection"):
                        continue
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(resp_body)
                conn.close()
                if self.counter is not None and self._is_generation_request():
                    # Only count requests that actually drive the crash mechanism.
                    # Open WebUI polls /health and /metrics every ~15-20s regardless
                    # of chat activity; counting those would trigger proactive
                    # restarts far more often than the real request volume needs.
                    self.counter.increment_and_maybe_flag()
                return
            except (ConnectionRefusedError, http.client.HTTPException, OSError) as exc:
                last_error = exc
                log(f"backend unreachable ({exc!r}); retrying in "
                    f"{RETRY_POLL_INTERVAL_S:.0f}s (deadline in "
                    f"{deadline - time.monotonic():.0f}s) -- hiding this from the "
                    f"client rather than failing the request")
                time.sleep(RETRY_POLL_INTERVAL_S)

        log(f"backend still unreachable after {self.retry_timeout_s:.0f}s; giving up "
            f"and returning 503 (last error: {last_error!r})")
        self.send_response(503)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(
            b'{"error":{"message":"backend unavailable after retrying across a '
            b'restart window","type":"ServiceUnavailable","code":503}}'
        )

    def do_GET(self) -> None:
        self._proxy()

    def do_POST(self) -> None:
        self._proxy()

    def log_message(self, fmt: str, *args) -> None:
        pass  # the backend's own access log already covers this; avoid double noise


class ThreadingProxyServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend-port", type=int, default=DEFAULT_BACKEND_PORT)
    ap.add_argument("--listen-port", type=int, default=DEFAULT_LISTEN_PORT)
    ap.add_argument("--restart-every", type=int, default=DEFAULT_PROACTIVE_RESTART_EVERY,
                    help="write the proactive-restart flag after this many requests "
                         "-- keep well under the observed ~18-20-request crash "
                         "threshold")
    ap.add_argument("--retry-timeout", type=float, default=DEFAULT_RETRY_TIMEOUT_S,
                    help="max seconds to keep retrying a request against a down "
                         "backend before giving up and returning 503")
    args = ap.parse_args(argv)

    ProxyHandler.backend_port = args.backend_port
    ProxyHandler.retry_timeout_s = args.retry_timeout
    ProxyHandler.counter = RequestCounter(args.restart_every)

    log(f"proxying :{args.listen_port} -> localhost:{args.backend_port}, "
        f"proactive restart every {args.restart_every} requests, "
        f"retry window {args.retry_timeout:.0f}s")
    server = ThreadingProxyServer(("0.0.0.0", args.listen_port), ProxyHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
