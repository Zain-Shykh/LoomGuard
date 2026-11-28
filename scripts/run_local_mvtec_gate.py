#!/usr/bin/env python3
"""Run Huzaima's frozen MVTec RD4AD gate as a detached WSL process."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time

SCRIPT = Path(__file__).resolve()
ARCHIVE = SCRIPT.parents[1] / "artifacts/private-training-archive"
RUNTIME = ARCHIVE / "runtime/mvtec-gate"
EXPORT = ARCHIVE / "verified/member-a"
SHARED = Path("artifacts/distributed/gate-shared")
PID_FILE = "training.pid"
EXIT_FILE = "exit-code.txt"
STATUS_FILE = "status.txt"
SOURCE = "67fb7f6ca2dec0365b63a7454ffb57b3138b6875"
PIN_SHA = "702a0913c595aecfbdae73ab89be2f72ba2ccdead54f53b20f526b42c43981a0"
WEIGHTS_SHA = "95faca4d11227dddf8633dbb5ff6c8a9003c1aa5b8945c73834b8007b10950b8"
TEXTURES = ("carpet", "grid", "leather", "tile", "wood")


class Interrupted(Exception):
    pass


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_runtime(name, value):
    fd, temp = tempfile.mkstemp(dir=RUNTIME)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(value)
        os.replace(temp, RUNTIME / name)
    finally:
        Path(temp).unlink(missing_ok=True)


def write_status(state, completed=(), current=None, detail=None):
    rows = [state, f"Updated UTC: {timestamp()}",
            f"Current texture: {current or 'none'}", f"Completed textures: {', '.join(completed) or 'none'}"]
    if detail:
        rows.append(detail)
    write_runtime(STATUS_FILE, "\n".join(rows) + "\n")


def identity(pid):
    stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    if stat[0] in {"Z", "X"}:
        raise ProcessLookupError(pid)
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    return {"pid": pid, "boot_id": boot, "start_ticks": int(stat[19])}


def current_process(stop=False):
    try:
        record = json.loads((RUNTIME / PID_FILE).read_text())
        pid = int(record["pid"])
        if pid <= 1:
            return None
        fd = os.pidfd_open(pid)
        try:
            if identity(pid) != record:
                return None
            if stop:
                signal.pidfd_send_signal(fd, signal.SIGTERM)
            return pid
        finally:
            os.close(fd)
    except (FileNotFoundError, ProcessLookupError):
        return None


def in_frozen_environment(args):
    return ["bash", "-c", 'source "$1" >&2 && shift && exec python -u "$@"',
            "loomguard-mvtec-gate", str(ARCHIVE / "tools/activate-rd4ad.sh"), str(SCRIPT), *args]


def preflight():
    from distributed_study import (PLAN, WORK, check_inputs, hardware, load_pin,
                                   ready_context, training_commit)
    from loomguard.distributed import digest_file, load_plan, read_json

    if training_commit() != SOURCE:
        raise ValueError("Expected the frozen experimental source and clean training files.")
    original = ARCHIVE / "incoming/member-b/pins/gate/datasets.json"
    if any(digest_file(path) != PIN_SHA for path in (original, SHARED / "datasets.json")):
        raise ValueError("The original gate pin and installed bytes must match the accepted SHA-256.")
    pin, jobs = load_pin(load_plan(PLAN), SHARED)
    runs = [run for job in jobs if (job.owner, job.stage, job.queue, job.job_id)
            == ("A", "gate", "gate", "mvtec_rd4ad") for run in job.runs()]
    if pin.get("scope") != "gate" or pin["source_commit"] != SOURCE or tuple(r.profile for r in runs) != TEXTURES:
        raise ValueError("Unexpected pinned gate scope or MVTec RD4AD assignment.")
    machine = hardware("A")
    if machine["gpu"] != "NVIDIA GeForce RTX 5060 Ti":
        raise ValueError(f"Expected the prepared RTX 5060 Ti, found {machine['gpu']}.")
    ready = read_json(WORK / "ready-A.json")
    if (any(ready.get(k) != v for k, v in ready_context(pin, PLAN, "A", machine).items())
            or not ready.get("environment_review") or ready.get("bundle_location") != str(EXPORT)
            or ready.get("version") != 2 or ready.get("verified_configs") != ["configs/mvtec_rd4ad.yaml"]):
        raise ValueError("Gate readiness is stale or incomplete. Repeat gate-scoped preparation.")
    weights = Path(os.environ["TORCH_HOME"]) / "hub/checkpoints/wide_resnet50_2-95faca4d.pth"
    if digest_file(weights) != WEIGHTS_SHA:
        raise ValueError("Approved frozen backbone checksum mismatch.")
    check_inputs([runs[0].job], pin)
    return pin, runs


def profile_status(run):
    from distributed_study import WORK
    return WORK / "profiles" / run.result.relative_to("results").with_suffix("") / "status.json"


def complete(run, pin):
    from loomguard.distributed import identity as run_identity, read_json, verify_bundle, verify_result

    if not run.result.exists() and not run.result.is_symlink():
        return False
    result = verify_result(run, run.result, pin)
    verify_bundle(run, run.bundle, result)
    status = profile_status(run)
    if not status.is_file():
        return False
    evidence = read_json(status)
    return evidence.get("state") == "passed" and evidence.get("identity") == run_identity(run, pin)


def export(run, pin):
    from distributed_study import WORK
    from loomguard.data.manifest import read_manifest
    from loomguard.distributed import BUNDLE_FILES, digest_file, read_json, verify_result
    from loomguard.runs import manifest_digest

    result = verify_result(run, run.result, pin)
    status = profile_status(run)
    entry = read_json(status)
    attempt = entry.get("attempt")
    if entry.get("state") != "passed" or not isinstance(attempt, str) or not re.fullmatch(r"attempt-[\w-]+", attempt):
        raise ValueError(f"{run.profile}: missing passed attempt with local export evidence.")
    logs = WORK / "attempts" / attempt
    manifest_path = logs / "artifacts/manifests/mvtec.csv"
    if manifest_digest(read_manifest(manifest_path)) != pin["manifests"][run.job.config]:
        raise ValueError(f"{run.profile}: attempt manifest differs from the pinned split.")
    files = [run.result, *(run.bundle / name for name in (*BUNDLE_FILES, "provenance.json")),
             status, logs / "fit.log", logs / "evaluate.log", manifest_path]
    skipped = re.search(r"(\d+) fp16-overflow steps skipped", (logs / "fit.log").read_text())
    if not skipped:
        raise ValueError(f"{run.profile}: missing GradScaler evidence in fit log.")
    curve = read_json(run.bundle / "metrics.json")["loss_curve"]
    if not curve or not all(math.isfinite(loss) for row in curve for loss in row[1:] if loss is not None):
        raise ValueError(f"{run.profile}: missing or non-finite loss curve.")
    checksums = "".join(f"{digest_file(path)}  {path.as_posix()}\n" for path in files)
    manifest = EXPORT / "checksums" / f"{run.profile}.sha256"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    if manifest.exists() and manifest.read_text() != checksums:
        raise ValueError(f"{manifest}: existing transfer manifest differs from the canonical files.")

    # Use the private receipt helper's atomic, no-overwrite copy for each file.
    sys.path.insert(0, str(ARCHIVE / "tools"))
    from verify_patchcore_exports import copy_once

    if not manifest.exists():
        for path in files:
            copy_once(path, EXPORT / path)
        with tempfile.NamedTemporaryFile(dir=manifest.parent) as pending:
            pending.write(checksums.encode())
            pending.flush()
            copy_once(pending.name, manifest)
    subprocess.run(["sha256sum", "--check", "--strict", str(manifest)], cwd=EXPORT, check=True)
    print(f"{run.profile}: fit {result['fit_minutes']:.2f} min, "
          f"fit/eval {result['distributed']['elapsed_seconds']:.1f} s, "
          f"peak VRAM {result['peak_vram_gb']:.2f} GB, skipped updates {skipped.group(1)}", flush=True)
    if result["peak_vram_gb"] >= 11:
        raise ValueError(f"{run.profile}: peak VRAM reached the 11 GB gate limit.")


def run_profile(profile, locks, interrupted, monitor):
    command = [sys.executable, "-u", "scripts/distributed_study.py", "--shared", str(SHARED),
               "--owner", "A", "--stage", "gate", "--job", "mvtec_rd4ad", "--profile", profile]
    with subprocess.Popen(command, start_new_session=True, pass_fds=locks) as process:
        requested = None
        while process.poll() is None:
            if interrupted() or monitor.poll() is not None:
                if requested is None:
                    requested = time.monotonic()
                    try:
                        os.killpg(process.pid, signal.SIGINT)
                    except ProcessLookupError:
                        pass
                elif time.monotonic() - requested > 30:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
            try:
                process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                continue
        if interrupted():
            raise Interrupted
        if monitor.poll() is not None:
            raise RuntimeError("GPU monitor exited during fitting.")
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command)


def worker(locks, startup):
    interrupted = False
    completed = []
    current = None
    monitor = None

    def request_stop(_signum, _frame):
        nonlocal interrupted
        interrupted = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, request_stop)
    with os.fdopen(startup, "w") as pipe:
        write_runtime(PID_FILE, json.dumps(identity(os.getpid())) + "\n")
        write_status("RUNNING", detail="Verifying source, gate pin and readiness.")
        pipe.write(f"{os.getpid()}\n")
    started = time.monotonic()
    exit_code = 1
    try:
        pin, runs = preflight()
        with (RUNTIME / "gpu.csv").open("w") as csv:
            monitor = subprocess.Popen(["nvidia-smi",
                "--query-gpu=timestamp,memory.used,memory.total,utilization.gpu", "--format=csv", "-l", "2"],
                stdout=csv)
            try:
                for run in runs:
                    if interrupted:
                        raise Interrupted
                    current = run.profile
                    write_status("RUNNING", completed, current)
                    profile_start = time.monotonic()
                    print(f"{timestamp()} {current} started", flush=True)
                    if complete(run, pin):
                        print(f"{current}: verified existing outputs, skipping fit", flush=True)
                    else:
                        run_profile(current, locks, lambda: interrupted, monitor)
                    export(run, pin)
                    completed.append(current)
                    print(f"{timestamp()} {current} verified and exported in {time.monotonic()-profile_start:.1f} s",
                          flush=True)
                if interrupted:
                    raise Interrupted
                if monitor.poll() is not None:
                    raise RuntimeError("GPU monitor exited.")
            finally:
                if monitor.poll() is None:
                    monitor.terminate()
                    try:
                        monitor.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        monitor.kill()
                        monitor.wait()
        write_status("COMPLETED", completed)
        exit_code = 0
    except Interrupted:
        write_status("INTERRUPTED", completed, current, "Previous exports and the current attempt were preserved.")
        exit_code = 130
    except Exception as exc:
        write_status("FAILED", completed, current, str(exc))
        print(f"{timestamp()} FAILED: {exc}", file=sys.stderr)
        if isinstance(exc, subprocess.CalledProcessError):
            exit_code = exc.returncode if exc.returncode > 0 else 128 - exc.returncode
    finally:
        write_runtime(EXIT_FILE, f"{exit_code}\n")
        print(f"{timestamp()} exit {exit_code} after {time.monotonic()-started:.1f} s", flush=True)
        for fd in locks:
            os.close(fd)
    return exit_code


def start():
    RUNTIME.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        locks = []
        for path in (RUNTIME / "training.lock", ARCHIVE / "logs/member-a-mvtec-gate.lock"):
            stream = stack.enter_context(path.open("a"))
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError("A local MVTec gate process already holds the training lock.") from exc
            locks.append(stream.fileno())
        if current_process() is not None:
            raise ValueError("A local MVTec gate worker is already running.")
        for name in (EXIT_FILE, PID_FILE, "gpu.csv"):
            (RUNTIME / name).unlink(missing_ok=True)
        write_status("RUNNING", detail="Starting detached worker.")
        reader, writer = os.pipe()
        with os.fdopen(reader) as receipt, (RUNTIME / "training.log").open("a") as log:
            try:
                print(f"\n{timestamp()} starting local MVTec RD4AD gate", file=log, flush=True)
                process = subprocess.Popen(in_frozen_environment(["_worker", *map(str, locks), str(writer)]),
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True, pass_fds=(*locks, writer))
            except OSError as exc:
                write_status("FAILED", detail=str(exc))
                write_runtime(EXIT_FILE, "1\n")
                raise
            finally:
                os.close(writer)
            if receipt.readline().strip() != str(process.pid):
                rc = process.wait()
                write_status("FAILED", detail="Worker startup failed. See training.log.")
                write_runtime(EXIT_FILE, f"{rc if rc >= 0 else 128-rc}\n")
                raise ValueError("Worker startup failed. See training.log.")
            print(f"Detached MVTec gate worker PID {process.pid}")
    return 0


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "start", "status", "stop"))
    if len(sys.argv) == 5 and sys.argv[1] == "_worker":
        return worker(tuple(map(int, sys.argv[2:4])), int(sys.argv[4]))
    if sys.argv[1:] == ["_check"]:
        pin, runs = preflight()
        for run in runs:
            print(f"{run.profile}: {'verified existing outputs' if complete(run, pin) else 'pending'}")
        return 0
    args = parser.parse_args()
    if args.command == "check":
        os.execvp("bash", in_frozen_environment(["_check"]))
    if args.command == "start":
        return start()
    pid = current_process(stop=args.command == "stop")
    if args.command == "stop":
        if pid is None:
            print("No live local MVTec gate worker.")
            return 1
        print(f"Sent stop request to MVTec gate worker PID {pid}.")
        return 0
    status = RUNTIME / STATUS_FILE
    text = status.read_text().rstrip() if status.exists() else "NOT STARTED"
    if pid is None and text.startswith("RUNNING"):
        print(f"FAILED\nWorker exited before recording a final status.\n{text}")
        return 1
    print(text)
    if pid is not None:
        print(f"PID: {pid}")
    code = RUNTIME / EXIT_FILE
    if code.exists():
        print(f"Exit status: {code.read_text().strip()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        sys.exit(f"Local MVTec gate: {exc}")
