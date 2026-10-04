"""Keeps experiments from wedging the Strix Halo box.

On this APU the GPU allocates from system RAM, and memory the GPU holds (GTT) is invisible
to Linux's OOM killer: when several model servers fill it, the amdgpu driver ends up
waiting forever for pages and the box hangs (2026-10-04). This watchdog checks free RAM and
GTT every few seconds and, before that happens, stops the newest model server that an
experiment started. It never touches Lemonade, Home Assistant or the Vecta server.

    tmux new-session -d -s guard "python3 ~/git/vecta/agent/scripts/gpu_guard.py"

Logs go to data/gpu_guard.log; warnings and actions also to data/gpu_guard.alerts.
"""

import argparse
import os
import re
import subprocess
import time
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data"
GIB = 2**30
WARN_AVAIL, KILL_AVAIL = 20 * GIB, 12 * GIB  # MemAvailable
WARN_GTT, KILL_GTT = 0.85, 0.90  # share of the GTT pool in use
# model servers started by experiments; Lemonade's own servers live under /var/cache/lemonade
MODEL_EXE = re.compile(r"(^|/)(llama-server|llama-mtmd-cli|llama-omni-cli|python[\d.]*)(\s|$)")
EXPERIMENT_DIR = re.compile(r"/work/|/\.scratch/")  # the container's mount, or a scratch dir
PROTECTED = re.compile(r"lemonade|homeassistant|vecta-server|gpu_guard")


def mem_available() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return 0


def gtt() -> tuple[int, int]:
    for dev in Path("/sys/class/drm").glob("card*/device"):
        used, total = dev / "mem_info_gtt_used", dev / "mem_info_gtt_total"
        if used.exists():
            return int(used.read_text()), int(total.read_text())
    return 0, 1


def experiments() -> list[tuple[float, int, str]]:
    """(start time, pid, command) of experiment processes, oldest first."""
    found = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            cmd = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            # start time in clock ticks: field 22, counted after the ")" that ends the name
            started = float((proc / "stat").read_text().rsplit(")", 1)[1].split()[19])
        except OSError, IndexError, ValueError:
            continue
        try:
            cwd = os.readlink(proc / "cwd")
        except OSError:
            cwd = ""
        ours = EXPERIMENT_DIR.search(f"{cmd} {cwd}") and not PROTECTED.search(cmd)
        if ours and MODEL_EXE.search(cmd):
            found.append((started, int(proc.name), cmd.strip()))
    return sorted(found)


def stop(pid: int, cmd: str) -> str:
    try:
        os.kill(pid, 15)
        return "SIGTERM"
    except PermissionError:
        pass  # started as root inside a container: stop it from inside that container
    except ProcessLookupError:
        return "already gone"
    cgroup = Path(f"/proc/{pid}/cgroup").read_text()
    m = re.search(r"docker[-/]([0-9a-f]{12,64})", cgroup)
    if not m:
        return "no permission"
    exe = cmd.split()[0].rsplit("/", 1)[-1]
    r = subprocess.run(
        ["docker", "exec", m.group(1)[:12], "pkill", "-n", "-f", exe],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return f"docker pkill -n {exe} (rc {r.returncode})"


def log(line: str, alert: bool = False) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    for name in ("gpu_guard.log",) + (("gpu_guard.alerts",) if alert else ()):
        with (DATA / name).open("a") as f:
            f.write(f"{stamp} {line}\n")


def main(interval: float, dry_run: bool) -> None:
    last_status = last_warn = 0.0
    warned_share, warned_avail = 0.0, float("inf")  # levels at the last warning
    while True:
        avail = mem_available()
        used, total = gtt()
        share = used / total
        status = (
            f"avail {avail / GIB:.1f} GiB, gtt {used / GIB:.1f}/{total / GIB:.0f} GiB ({share:.0%})"
        )
        now = time.monotonic()
        if avail < KILL_AVAIL or share > KILL_GTT:
            procs = experiments()
            if procs:
                _, pid, cmd = procs[-1]  # the newest experiment is the one that tipped it over
                action = "dry run" if dry_run else stop(pid, cmd)
                log(f"DANGER {status}: stopping pid {pid} ({action}): {cmd[:200]}", alert=True)
                time.sleep(10)  # let the memory come back before judging again
                continue
            if now - last_warn > 60:
                log(f"DANGER {status}: no experiment process to stop", alert=True)
                last_warn = now
        elif avail < WARN_AVAIL or share > WARN_GTT:
            # warn when it gets worse (or every 15 min while it stays high), not every minute
            worse = share > warned_share + 0.05 or avail < warned_avail - 5 * GIB
            if worse or now - last_warn > 900:
                log(f"WARN {status}", alert=True)
                last_warn, warned_share, warned_avail = now, share, avail
        elif avail > WARN_AVAIL + 5 * GIB and share < WARN_GTT - 0.05:
            warned_share, warned_avail = 0.0, float("inf")  # clearly back to normal
        if now - last_status > 60:
            log(status)
            last_status = now
        time.sleep(interval)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--dry-run", action="store_true", help="log what would be stopped")
    args = ap.parse_args()
    main(args.interval, args.dry_run)
