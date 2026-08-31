#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# Launch the demo's training run so that tt-toplike's Training view can actually read it,
# and tail it into this pane so the recording has something to show.
#
# WHY THE REDIRECT IS NOT OPTIONAL. tt-toplike attaches to a training process by resolving
# /proc/<pid>/fd/1 (src/workload/train/logsrc.rs). If stdout is a regular file it tails it;
# if stdout is a pipe or a tty it reports NotRedirected and the view says so instead of
# inventing data. Take 1 of this demo ran the trainer straight into the pane, so fd 1 was a
# pty and the Training view spent nine minutes displaying
# "stdout is not redirected to a file — relaunch with '> train.log'" over an empty panel.
#
# THIS IS ALSO WHY `| tee` DOES NOT WORK. tee makes fd 1 a pipe, which classify_fd_target
# rejects by the same rule. The process's own stdout has to be the file; a second reader is
# what the `tail -f` below is for.
#
# WHY THE PANE IS NOT FILTERED. The obvious way to hide ttnn's startup DEBUG wall is
# `tail -f | grep -v`, and it would freeze the pane: tqdm writes its progress bar as ONE
# newline-less line updated with carriage returns, so any line-oriented filter buffers the
# whole run waiting for a `\n` that does not arrive until the bar finishes. Waiting for
# training to start and then clearing once costs nothing and cannot stall.
set -uo pipefail

LOG="${1:-demo/train.log}"
STEPS="${2:-3000}"

: >"$LOG"

# APPEND, not truncate-and-write. Both this shell and the trainer write to this file, and
# O_APPEND is what makes that safe: every write goes to the current end rather than to a
# per-fd offset that the other writer has since moved past. See the heartbeat below.
gozer run --chips 1 --who "claude:editor-lora-demo" \
    --reason "editor LoRA payoff run, recorded" \
    -- python scripts/train_editor_lora.py \
       --steps "$STEPS" --save-every 100 --eval-every 250 \
    >>"$LOG" 2>&1 &
TRAIN_PID=$!

# THE NEWLINE HEARTBEAT — why the Training view was blank for five minutes.
#
# tt-toplike's tailer accepts a chunk only if it ends with '\n' and otherwise leaves it for
# the next poll (monitor.rs:206). ttml's SFTTrainer reports loss SOLELY as tqdm bar postfix,
# and tqdm redraws with carriage returns, so between the startup lines and the final summary
# this log contains no newline at all for ~300 seconds. The tailer therefore buffers the
# entire run and flushes it in one burst at the end -- which is why take 2's loss "mountains"
# appeared only in the last frames, and why MODEL/tokens/topology (all parsed from log lines)
# stayed unknown while the LIVE block froze after startup.
#
# The upstream code means to handle this: the '\r'-splitting immediately below that guard
# exists, by its own comment, so "a tqdm-only trainer yields nothing until the run ends". The
# newline requirement above it reintroduces exactly that. Appending an empty line every few
# seconds terminates the pending chunk, so the tailer emits the accumulated bar frames and
# the view advances in real time. It adds blank lines to the log and changes nothing about
# the run.
( while kill -0 "$TRAIN_PID" 2>/dev/null; do printf '\n' >>"$LOG"; sleep 3; done ) &
HEARTBEAT_PID=$!

TAIL_PID=""
cleanup() {
    [ -n "$TAIL_PID" ] && kill "$TAIL_PID" 2>/dev/null
    [ -n "${HEARTBEAT_PID:-}" ] && kill "$HEARTBEAT_PID" 2>/dev/null
    # TERM gozer first and give it a moment to release its lease. Killing the process group
    # outright (take 1) leaves the lease STALE on a shared box -- recoverable with
    # `gozer reconcile`, but every take shouldn't need one.
    if kill -0 "$TRAIN_PID" 2>/dev/null; then
        kill -TERM "$TRAIN_PID" 2>/dev/null
        for _ in $(seq 1 20); do
            kill -0 "$TRAIN_PID" 2>/dev/null || break
            sleep 0.5
        done
        kill -KILL "$TRAIN_PID" 2>/dev/null
    fi
    wait "$TRAIN_PID" 2>/dev/null
}
trap cleanup EXIT INT TERM HUP

# Hold the pane until the model is on the device and stepping, then clear the ttnn/UMD
# startup wall so the shot opens on training rather than on driver initialisation. The log
# keeps every line -- this hides setup noise from the camera, it does not discard it.
#
# Report real milestones while waiting rather than sitting silent. Warm start plus kernel
# compilation is a fixed ~37s, and the probe take proved what silence costs: a black pane
# for the first 82% of a short shot. These lines are read out of the log as they land, so
# the pane is honest about what the box is doing, not a spinner pretending.
printf '  tt-tnt · editor-LoRA\n  warm start + kernel compilation — the board is busy, the trainer is not stepping yet\n\n'
SEEN=""
for _ in $(seq 1 600); do
    grep -q "SFTTrainer:" "$LOG" 2>/dev/null && break
    kill -0 "$TRAIN_PID" 2>/dev/null || break
    for marker in "Opening user mode device driver:opening device" \
                  "Starting topology discovery:topology discovery" \
                  "Cluster constructor completed:cluster up" \
                  "warm start:warm start — 66 parameters copied from the dialogue checkpoint" \
                  "total parameters:LoRA injected — 48 of 114 parameters trainable"; do
        pat=${marker%%:*}; msg=${marker#*:}
        case "$SEEN" in *"|$pat|"*) continue;; esac
        if grep -q "$pat" "$LOG" 2>/dev/null; then
            printf '  · %s\n' "$msg"; SEEN="$SEEN|$pat|"
        fi
    done
    sleep 0.5
done
printf '  · stepping\n'
sleep 1
clear

# Follow the same file tt-toplike is reading, so the pane and the Training view are two
# views of one run rather than two independent runs.
#
# `tr '\r' '\n'` is not cosmetic. tqdm renders its progress bar as a SINGLE newline-less
# line updated with carriage returns, so tailed straight into a 52-row pane it repaints one
# row at the top and leaves the other 51 blank -- which is exactly how the probe take came
# back with a black left pane despite the capture containing 40 progress frames. Turning
# each \r into a newline makes those frames scroll.
#
# stdbuf -o0 is what makes it appear live: tr writes to a pipe, so without it stdio
# block-buffers 4 KB and the pane arrives in lurches (or not at all inside a short take).
tail -f -n 5 "$LOG" | stdbuf -o0 tr '\r' '\n' &
TAIL_PID=$!

wait "$TRAIN_PID"
sleep 4          # let tail flush the freeze-verification lines into the shot
