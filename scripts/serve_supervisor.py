#!/usr/bin/env python3
# scripts/serve_supervisor.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Resilience wrapper for serving a tt-tnt checkpoint via vLLM.

Why this exists: `docs/serving-with-tt-kernel.md` sec.7 documents a real, unresolved
upstream defect -- the served engine dies (`EngineDeadError`) after roughly 18-20
completion requests, REGARDLESS of how short each prompt is, because cumulative
KV-cache position state is not reset properly between separate, independent
requests. That code lives in `models/tt_transformers` (tt-metal), outside this
repo's tree; we cannot patch it here. What we CAN do is stop treating that crash as
"the service is now permanently dead until a human notices and manually restarts
it" and instead make it "one failed request, then automatically back up".

This script owns the gozer lease indirectly: it launches the SAME `gozer run ...
tt-model serve ...` recipe as a subprocess in its own process group, so killing that
one group on a health-check failure or unexpected exit takes the whole tree with it
(server_example_tt.py, its EngineCore child, and the gozer wrapper) -- avoiding the
orphaned-child problem hit twice by hand earlier this session, where killing only the
top-level `gozer run` pid left `server_example_tt.py`/`EngineCore` running and the
lease stuck `HELD-FOREIGN`.

Health is checked two ways, either one triggers a relaunch:
1. The subprocess has exited (`poll() is not None`) -- the crash's own observed
   signature: the APIServer process calls its own clean shutdown and exits 0.
2. The subprocess is alive but `/v1/models` stops answering within a short timeout --
   catches a hang that doesn't exit, which the crash signature above does not, but a
   watchdog should not assume it never will.

Restarts are logged with a timestamp and count, printed to stdout, so a human
watching the terminal (or the log file if redirected) can see exactly when and how
often this fired -- a silently-recovering service that never says so would hide a
real reliability problem behind an illusion of stability.

    gozer run --chips all --who "claude:serve-supervisor" \\
        --reason "resilient 4-chip serving" -- \\
        python3 scripts/serve_supervisor.py
    # ^ WRONG: do not wrap this script itself in a lease. It launches its OWN
    # leased subprocess per restart; wrapping it in an outer lease would hold chips
    # for the supervisor's entire lifetime even between restarts, which defeats the
    # point (a crashed server should release its chips so gozer's own accounting
    # stays honest, and the next launch takes a fresh lease).

    python3 scripts/serve_supervisor.py
"""

from __future__ import annotations

import argparse
import datetime
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

from serve_proxy import RESTART_FLAG_PATH  # same path serve_proxy.py writes to

EXAMPLES_DIR = "/home/ttuser/vllm-tt-plugin-standalone/examples"
DEFAULT_MODEL = "episod/tt-tnt-1024"
DEFAULT_PORT = 8003
DEFAULT_MESH_DESC = "/home/ttuser/code/tt-tnt/train/configs/mesh/mesh-1x4-ring.textproto"
#: All four p300c BDFs on this box, in the ring order the mesh descriptor assumes.
DEFAULT_VISIBLE_DEVICES = "0000:01:00.0,0000:02:00.0,0000:03:00.0,0000:04:00.0"
HEALTH_TIMEOUT_S = 5.0
POLL_INTERVAL_S = 5.0
STARTUP_TIMEOUT_S = 240.0
#: Seconds to wait before relaunching after a NORMAL crash -- one where the server
#: was healthy for a while first, then died the documented KV-cache-position way.
#: This is the fast path: gives the just-killed lease time to actually release so
#: the next `gozer run` is granted rather than queued (exit code 10), and nothing
#: more.
RELAUNCH_BACKOFF_S = 10.0
#: When a launch attempt fails WITHOUT ever becoming healthy -- gozer itself
#: erroring, the device refusing to open, anything other than "served fine, then
#: hit the known bug" -- back off exponentially instead of retrying at the fast,
#: fixed interval. Observed live: hammering `gozer run` every ~15s for several
#: minutes straight against a device that would not open turned a recoverable
#: hiccup into a sustained "unreadable tt_serial" driver-level wedge. A struggling
#: driver needs LESS traffic while it settles, not more.
STARTUP_FAILURE_BACKOFF_BASE_S = 15.0
STARTUP_FAILURE_BACKOFF_MAX_S = 300.0
#: After this many consecutive failed-to-become-healthy attempts, stop retrying
#: automatically and wait for a human. A tight loop that never succeeds is not
#: "resilience" -- past this point the device is plausibly wedged in a way no
#: amount of relaunching fixes, and continuing to hammer it risks making that
#: worse, per the observed pattern above.
MAX_CONSECUTIVE_STARTUP_FAILURES = 5
GIVE_UP_POLL_INTERVAL_S = 60.0
#: gozer's own convention for "the lease request was queued, not granted" (see the
#: gozer-keymaster skill) -- distinct from every other nonzero exit, which means the
#: launch genuinely failed or the served process crashed.
GOZER_QUEUED_EXIT_CODE = 10


def log(msg: str) -> None:
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    print(f"[{ts}] {msg}", flush=True)


def build_launch_cmd(args) -> list:
    return [
        "gozer", "run", "--chips", "all",
        "--who", "claude:serve-supervisor",
        "--reason", f"resilient serve of {args.model} (auto-restart on the known "
                    f"KV-cache-position engine-death bug)",
        "--",
        "env",
        f"TT_VISIBLE_DEVICES={args.visible_devices}",
        f"TT_MESH_GRAPH_DESC_PATH={args.mesh_desc}",
        "tt-model", "serve", args.model, "--force", "--instance", "metal-src-vllm",
        "--",
        "--additional-config", '{"tt":{"fabric_config":"FABRIC_2D_TORUS_XY"}}',
        "--port", str(args.port),
        "--enable-auto-tool-choice", "--tool-call-parser", "hermes",
        "--no-enable-prefix-caching",
    ]


def is_healthy(port: int) -> bool:
    try:
        urllib.request.urlopen(f"http://localhost:{port}/v1/models", timeout=HEALTH_TIMEOUT_S)
        return True
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def kill_process_group(proc: subprocess.Popen) -> None:
    """Kill the WHOLE process group the launch command started, not just its own pid --
    `gozer run` does not reliably propagate SIGTERM to the server subprocess it
    launches (observed directly, twice, earlier in this session: killing only the
    `gozer run` pid left `server_example_tt.py` and its `EngineCore` child running,
    with the lease stuck HELD-FOREIGN until they were found and killed by hand).
    """
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--mesh-desc", default=DEFAULT_MESH_DESC)
    ap.add_argument("--visible-devices", default=DEFAULT_VISIBLE_DEVICES)
    args = ap.parse_args(argv)

    launch_cmd = build_launch_cmd(args)
    restarts = 0
    #: Consecutive launches that NEVER became healthy -- reset to 0 the moment one
    #: does. This is the counter exponential backoff and the give-up threshold key
    #: off, deliberately separate from `restarts` (which also counts the normal,
    #: fast-recovery crash-after-served case and should not slow that path down).
    consecutive_startup_failures = 0
    proc: subprocess.Popen | None = None
    shutting_down = False

    def handle_signal(signum, _frame):
        nonlocal shutting_down
        log(f"received signal {signum}; shutting down the served process and exiting")
        shutting_down = True
        if proc is not None:
            kill_process_group(proc)
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    while not shutting_down:
        if proc is None:
            if consecutive_startup_failures >= MAX_CONSECUTIVE_STARTUP_FAILURES:
                log(f"{consecutive_startup_failures} consecutive launches never became "
                    f"healthy -- this looks like a wedged device, not the crash this "
                    f"script exists to absorb. Waiting {GIVE_UP_POLL_INTERVAL_S:.0f}s "
                    f"before trying again rather than hammering it further; a human "
                    f"should check `gozer status` and the driver "
                    f"(`cat /sys/class/tenstorrent/tenstorrent!0/tt_serial`).")
                time.sleep(GIVE_UP_POLL_INTERVAL_S)
            elif restarts > 0:
                if consecutive_startup_failures > 0:
                    backoff = min(
                        STARTUP_FAILURE_BACKOFF_BASE_S * (2 ** (consecutive_startup_failures - 1)),
                        STARTUP_FAILURE_BACKOFF_MAX_S,
                    )
                    log(f"previous launch never became healthy "
                        f"({consecutive_startup_failures} in a row); backing off "
                        f"{backoff:.0f}s before retrying instead of hammering a "
                        f"possibly-struggling device")
                else:
                    # Give the just-killed lease time to actually release before asking
                    # gozer for a new one -- the fast path, for the normal crash. Without
                    # this a relaunch can race the previous process's release/reset and
                    # gozer QUEUES the request instead of granting it (exit code 10).
                    backoff = RELAUNCH_BACKOFF_S
                    log(f"waiting {backoff:.0f}s for the previous lease to release "
                        f"before relaunching")
                time.sleep(backoff)
            log(f"launching (restart #{restarts}): {' '.join(launch_cmd)}")
            proc = subprocess.Popen(launch_cmd, cwd=EXAMPLES_DIR, start_new_session=True)
            start = time.monotonic()
            while time.monotonic() - start < STARTUP_TIMEOUT_S:
                if proc.poll() is not None:
                    if proc.returncode == GOZER_QUEUED_EXIT_CODE:
                        log("gozer queued this request rather than granting it "
                            "(exit code 10) -- the backoff above wasn't long enough "
                            "this time; retrying")
                    else:
                        log(f"process exited during startup (code {proc.returncode}) "
                            f"before ever becoming healthy")
                    break
                if is_healthy(args.port):
                    log(f"healthy after {time.monotonic() - start:.1f}s")
                    consecutive_startup_failures = 0
                    break
                time.sleep(POLL_INTERVAL_S)
            else:
                log(f"did not become healthy within {STARTUP_TIMEOUT_S}s; killing "
                    f"and retrying")
                kill_process_group(proc)
                proc = None
                restarts += 1
                consecutive_startup_failures += 1
                continue
            if proc.poll() is not None:
                # The inner loop above broke out of a failed launch (either the
                # queued case or a genuine failure); either way this attempt never
                # got healthy, so it counts toward the exponential-backoff/give-up
                # tracking, not the fast normal-crash path.
                proc = None
                restarts += 1
                consecutive_startup_failures += 1
                continue

        time.sleep(POLL_INTERVAL_S)
        if proc.poll() is not None:
            log(f"served process exited on its own (code {proc.returncode}) -- "
                f"this is the crash's own signature (a clean self-shutdown after "
                f"EngineDeadError); relaunching")
            proc = None
            restarts += 1
            continue
        if not is_healthy(args.port):
            log("process alive but /v1/models is not responding -- treating as a "
                "hang, killing the whole process group and relaunching")
            kill_process_group(proc)
            proc = None
            restarts += 1
            continue
        if os.path.exists(RESTART_FLAG_PATH):
            log(f"proactive restart requested ({RESTART_FLAG_PATH} present) -- "
                f"cycling the server BEFORE the known bug's ~18-20-request "
                f"threshold, not in reaction to it")
            os.remove(RESTART_FLAG_PATH)
            kill_process_group(proc)
            proc = None
            restarts += 1
            continue

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
