"""Run one member's LoomGuard queue, including shared readiness and integration checks."""
import argparse
from contextlib import contextmanager
import importlib.metadata
import os
from pathlib import Path
import platform
import re
from statistics import mean
import subprocess
import sys
import tempfile
import time

import torch
import yaml

# This script is run from the repository root, as are the existing study entry points.
from loomguard.distributed import (
    FABRICS, GATE_TARGETS, OWNERS, QUEUES, STAGES, bundle_hashes, digest_file, digest_obj, digest_tree, expected_fabrics,
    expected_profiles, finite, identity, load_plan, read_json, verify_bundle, verify_result, write_json,
)
from loomguard.runs import build_rows, manifest_digest, profiles
from summarize import load_results, missing_cells

PLAN = Path("configs/studies/distributed/assignments.yaml")
WORK = Path("artifacts/distributed")
COMMON_ERRORS = (ValueError, KeyError, TypeError, OSError, subprocess.SubprocessError, yaml.YAMLError)


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def training_commit():
    changed = git("status", "--porcelain", "--untracked-files=all", "--", "configs", "loomguard", "scripts", "pyproject.toml")
    if changed:
        raise ValueError("training requires a pinned checkout with clean code and configs")
    return git("rev-parse", "HEAD")


def configs_for(jobs):
    return {job.config: job.cfg for job in jobs}


def dataset_roots(jobs):
    roots = {}
    for cfg in configs_for(jobs).values():
        old = roots.setdefault(cfg["dataset"], cfg["root"])
        if old != cfg["root"]:
            raise ValueError(f"{cfg['dataset']}: configs must point at the same dataset")
    return roots


def dataset_hashes(jobs):
    return {ds: digest_tree(root) for ds, root in dataset_roots(jobs).items()}


def inspected_manifest(cfg):
    rows = build_rows(cfg)
    available = set(expected_fabrics(cfg if cfg["regime"] != "held_out" else {**cfg, "fabrics": None}))
    found = {row.fabric for row in rows}
    if found != available:
        raise ValueError(f"{cfg['config']}: source fabric inventory differs: missing {sorted(available - found)}, "
                         f"unexpected {sorted(found - available)}")
    complete = profiles(rows, cfg)
    if set(complete) != set(expected_profiles(cfg)):
        raise ValueError(f"{cfg['config']}: dataset is missing canonical profiles")
    for name, selected in complete.items():
        if {r.split for r in selected} != {"train", "val", "test"}:
            raise ValueError(f"{cfg['config']} {name}: train, validation and test pools must be present")
    return manifest_digest(rows)


def pin_datasets(jobs, source, review, aitex_approval, path):
    if path.exists():
        raise ValueError(f"{path} already exists. Use a new shared directory for a revised dataset pin.")
    active = [j for j in jobs if j.queue != "aitex" or aitex_approval]
    datasets = dataset_hashes(active)
    manifests = {}
    configs = configs_for(active)
    for name, cfg in configs.items():
        print(f"Verifying split manifest for {name}", flush=True)
        manifests[name] = inspected_manifest(cfg)
    write_json(path, {"version": 1, "source_commit": source, "verification_review": review,
                      "aitex_approval": aitex_approval, "datasets": datasets, "manifests": manifests,
                      "configs": {name: digest_obj(cfg) for name, cfg in configs.items()}})


def load_pin(jobs, shared):
    pin = read_json(shared / "datasets.json")
    source = pin["source_commit"]
    if (pin["version"] != 1 or not pin["verification_review"] or not re.fullmatch(r"[0-9a-f]{40}", source)
            or git("rev-parse", "--verify", f"{source}^{{commit}}") != source):
        raise ValueError("invalid checkpoint 0 dataset pin")
    active = [j for j in jobs if j.queue != "aitex" or pin.get("aitex_approval")]
    configs = configs_for(active)
    if (set(pin["datasets"]) != set(dataset_roots(active)) or set(pin["manifests"]) != set(configs)
            or pin["configs"] != {name: digest_obj(cfg) for name, cfg in configs.items()}):
        raise ValueError("dataset pin does not match the active plan's configs and scope")
    for digest in (*pin["datasets"].values(), *pin["manifests"].values()):
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("invalid checksum in dataset pin")
    return pin, active


def hardware(owner):
    if not torch.cuda.is_available():
        raise ValueError("CUDA is required for distributed fitting")
    if owner == "A" and "sm_120" not in torch.cuda.get_arch_list():
        raise ValueError("Member A requires a PyTorch build with sm_120")
    torch.ones(1, device="cuda").add_(1)
    driver = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).splitlines()[0].strip()
    return {"machine": platform.node(), "platform": platform.platform(), "python": platform.python_version(),
            "gpu": torch.cuda.get_device_name(), "driver": driver, "torch": str(torch.__version__), "cuda": torch.version.cuda,
            "native_bf16": torch.cuda.is_bf16_supported(including_emulation=False),
            "packages": {name: importlib.metadata.version(name) for name in ("torchvision", "numpy", "pillow", "pyyaml")}}


def check_inputs(jobs, pin):
    for ds, actual in dataset_hashes(jobs).items():
        if actual != pin["datasets"][ds]:
            raise ValueError(f"{ds}: dataset contents differ from checkpoint 0")


def ready_context(pin, plan, owner, machine):
    return {"pin_sha256": digest_obj(pin), "plan_sha256": digest_file(plan), "owner": owner, "hardware": machine}


def prepare(jobs, pin, plan, owner, review, location):
    machine = hardware(owner)
    check_inputs(jobs, pin)
    for name, cfg in configs_for(jobs).items():
        print(f"Checking local split manifest for {name}", flush=True)
        if inspected_manifest(cfg) != pin["manifests"][name]:
            raise ValueError(f"{name}: split manifest differs from checkpoint 0")
    ready = ready_context(pin, plan, owner, machine) | {"environment_review": review, "bundle_location": location.rstrip("/")}
    write_json(WORK / f"ready-{owner}.json", ready, replace=True)
    print(f"Checkpoint 0 verified for member {owner} on {machine['gpu']}")


def checkpoint_runs(jobs, queue):
    return [run for job in jobs if job.queue == queue and (queue != "mini" or job.stage == "study") for run in job.runs()]


def benchmark_means(results):
    means = {}
    for method, targets in GATE_TARGETS.items():
        category = [r for r in results if r["method"] == method]
        if len(category) != 5 or {r["fabric"] for r in category} != set(FABRICS["mvtec"]):
            raise ValueError(f"{method}: checkpoint requires all five MVTec textures")
        means[method] = {}
        for metric, target in targets.items():
            # One tile per resized MVTec image. Use the image metric specified by D-33.
            key = "image_auroc" if metric == "tile_auroc" else metric
            if any(not finite(r.get(key), 0, 1) for r in category):
                raise ValueError(f"{method}: undefined MVTec {key}")
            value = mean(r[key] for r in category)
            means[method][key] = value
    return means


def gate_passes(means):
    for method, targets in GATE_TARGETS.items():
        for metric, target in targets.items():
            key = "image_auroc" if metric == "tile_auroc" else metric
            value = means[method][key]
            if not finite(value, target / 100, 1):
                raise ValueError(f"STOP: {method} MVTec {key} mean {100 * value:.4f}% missed {target:.1f}%.")


def checkpoint_context(jobs, pin, queue):
    return {"source_commit": pin["source_commit"],
            "inputs": [identity(run, pin) for run in sorted(checkpoint_runs(jobs, queue), key=lambda r: str(r.result))]}


def check_approval(jobs, pin, shared, queue):
    path = shared / f"{queue}.json"
    if not path.is_file():
        raise ValueError(f"waiting for shared checkpoint approval {path}")
    approval = read_json(path)
    expected = checkpoint_context(jobs, pin, queue)
    if approval.get("context") != expected or not approval.get("review"):
        raise ValueError(f"{path}: approval is stale or incomplete")
    evidence = approval.get("result_sha256", {})
    if set(evidence) != {i["result_path"] for i in expected["inputs"]} or not all(
        isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v) for v in evidence.values()
    ):
        raise ValueError(f"{path}: approval lacks the complete result inventory")
    if queue == "gate":
        gate_passes(approval["means"])
        if not finite(approval.get("rd4ad_peak_vram_gb"), 0) or approval["rd4ad_peak_vram_gb"] >= 11:
            raise ValueError(f"{path}: RD4AD MVTec peak VRAM must be under 11 GB")
    return approval


def approve(jobs, pin, shared, queue, review):
    if queue == "mini":
        check_approval(jobs, pin, shared, "gate")
    evidence = {}
    results = []
    for run in checkpoint_runs(jobs, queue):
        res = verify_result(run, run.result, pin)
        verify_bundle(run, run.bundle, res)
        evidence[run.result.as_posix()] = digest_file(run.result)
        results.append(res)
    approval = {"context": checkpoint_context(jobs, pin, queue), "review": review, "result_sha256": evidence}
    if queue == "gate":
        approval["means"] = benchmark_means(results)
        for method, metrics in approval["means"].items():
            print(method, " ".join(f"{k}={100 * v:.4f}%" for k, v in metrics.items()), flush=True)
        gate_passes(approval["means"])
        peak = max(r["peak_vram_gb"] for r in results if r["method"] == "rd4ad")
        if peak >= 11:
            raise ValueError(f"STOP: RD4AD MVTec peak VRAM {peak:.2f} GB must be under 11 GB")
        approval["rd4ad_peak_vram_gb"] = peak
    write_json(shared / f"{queue}.json", approval)
    print(f"Shared {queue} checkpoint approved")


@contextmanager
def profile_lock(path):
    """An OS lock is released on process exit, including a terminated runner."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as f:
        try:
            if os.name == "nt":
                import msvcrt
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ValueError(f"profile already being run: {path.parent}") from exc
        yield


def train_and_evaluate(run, attempt, manifest):
    args = ["--config", run.job.config, "--profiles", run.profile, "--output-root", str(attempt),
            "--manifest-sha256", manifest]
    for name in ("fit", "evaluate"):
        with (attempt / f"{name}.log").open("wb") as log:
            subprocess.run([sys.executable, "-u", f"scripts/{name}.py", *args], stdout=log, stderr=subprocess.STDOUT, check=True)


def publish(run, attempt, pin):
    result = attempt / run.result
    res = verify_result(run, result, pin)
    staged = attempt / run.bundle
    if staged.exists():
        if run.bundle.exists():
            raise ValueError(f"{run.bundle} already exists. Review it before publishing.")
        verify_bundle(run, staged, res)
        run.bundle.parent.mkdir(parents=True, exist_ok=True)
        staged.rename(run.bundle)
    else:
        # A previous process published the bundle and was interrupted before linking the result.
        verify_bundle(run, run.bundle, res)
    run.result.parent.mkdir(parents=True, exist_ok=True)
    os.link(result, run.result)


def run_profile(run, pin, location, machine, prerequisite):
    state_dir = WORK / "profiles" / run.result.relative_to("results").with_suffix("")
    attempts = WORK / "attempts"
    status = state_dir / "status.json"
    with profile_lock(state_dir / "lock"):
        info = identity(run, pin)
        previous = read_json(status) if status.exists() else {}
        state = {"identity": info}

        def record(value, **extra):
            write_json(status, {**state, **extra, "state": value}, replace=True)

        phase = "blocked"
        try:
            prerequisite()
            if run.result.exists() or run.result.is_symlink():
                res = verify_result(run, run.result, pin)
                verify_bundle(run, run.bundle, res)
                record("passed", attempt=previous.get("attempt"), message="verified existing outputs")
                return "verified, skipped"
            if previous.get("identity") == info and previous.get("ready_to_publish"):
                old = previous["attempt"]
                if not isinstance(old, str) or Path(old).name != old or not old.startswith("attempt-"):
                    raise ValueError(f"{status}: invalid staged attempt")
                state.update(attempt=old, ready_to_publish=True)
                publish(run, attempts / old, pin)
                record("passed", message="published staged outputs")
                return "recovered publication"
            if previous.get("ready_to_publish"):
                raise ValueError(f"{status}: staged result has conflicting provenance. Review before retrying.")
            if run.bundle.exists() or run.bundle.is_symlink():
                raise ValueError(f"{run.bundle} contains an unverified bundle. Review before retrying.")
            if run.cfg["method"] == "rd4ad" and not machine["native_bf16"]:
                raise ValueError("this accelerator lacks native CUDA bfloat16 support for the fixed RD4AD training mode")
            phase = "failed"
            attempts.mkdir(parents=True, exist_ok=True)
            attempt = Path(tempfile.mkdtemp(prefix="attempt-", dir=attempts))
            state["attempt"] = attempt.name
            record("running")
            start = time.perf_counter()
            train_and_evaluate(run, attempt, info["manifest_sha256"])
            res = verify_result(run, attempt / run.result, pin, provenance=False)
            staged = attempt / run.bundle
            proof = {**info, "hardware": machine, "bundle_location": f"{location}/{run.bundle.as_posix()}",
                     "elapsed_seconds": time.perf_counter() - start, "bundle_sha256": bundle_hashes(run, staged, res)}
            write_json(staged / "provenance.json", proof)
            write_json(attempt / run.result, {**res, "distributed": proof}, replace=True)
            state["ready_to_publish"] = True
            record("running")
            publish(run, attempt, pin)
            record("passed")
            return f"published {run.result}"
        except (*COMMON_ERRORS, KeyboardInterrupt) as exc:
            # Keep the last recoverable attempt pointer when a prerequisite is temporarily unavailable.
            if previous.get("ready_to_publish") and previous.get("identity") != info:
                state.update(previous)
            if "attempt" not in state and previous.get("identity") == info and previous.get("attempt"):
                state.update(attempt=previous["attempt"], ready_to_publish=previous.get("ready_to_publish", False))
            record(phase, message=str(exc) or type(exc).__name__)
            raise


def check_integration(all_jobs, active, pin, shared, roots):
    approvals = [check_approval(active, pin, shared, queue) for queue in ("gate", "mini")]
    expected = {r.result: r for job in active for r in job.runs()}
    approved_hashes = {p: h for a in approvals for p, h in a["result_sha256"].items()}
    seen = {}
    issues = []
    for root in roots:
        for path in sorted((root / "results").rglob("*.json")):
            relative = Path("results") / path.relative_to(root / "results")
            if relative not in expected:
                issues.append(f"unexpected result {path}")
                continue
            if relative in seen:
                issues.append(f"duplicate result {relative}: {seen[relative]} and {path}")
                continue
            seen[relative] = path
            run = expected[relative]
            try:
                res = verify_result(run, path, pin)
                verify_bundle(run, root / run.bundle, res)
                if run.job.job_id in ("raw_fabrid_patchcore", "raw_fabrid_rd4ad_s0"):
                    if not finite(res["false_alarms_per_1000"], 5, 20):
                        issues.append(f"{path}: RAW-FABRID headline false alarms outside 5 to 20 per 1,000")
                    if run.cfg["method"] == "patchcore" and res["coreset_seconds"] >= 600:
                        issues.append(f"{path}: RAW-FABRID headline PatchCore coreset must complete in under 600 seconds")
                if relative.as_posix() in approved_hashes and digest_file(path) != approved_hashes[relative.as_posix()]:
                    raise ValueError(f"{path}: differs from the approved checkpoint result")
            except COMMON_ERRORS as exc:
                issues.append(str(exc))
    missing = expected.keys() - seen.keys()
    issues.extend(f"missing {path.as_posix()}" for path in sorted(missing))
    # The accepted full-study acceptance criteria still include AITEX.
    omitted = [r.result for job in all_jobs if job not in active for r in job.runs()]
    if omitted:
        issues.append(f"full-study completion pending: {len(omitted)} AITEX results await team scope approval")
    if issues:
        print("\n".join(issues), file=sys.stderr)
        return 1
    merged = {}
    for root in roots:
        merged.update(load_results(root / "results"))
    coverage = missing_cells(merged)
    if coverage:
        print("\n".join(coverage), file=sys.stderr)
        return 1
    print(f"Verified {len(seen)} canonical results with their bundles and full-study coverage")
    return 0


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--shared", type=Path, default=WORK / "shared", help="directory for checkpoint metadata")
    parser.add_argument("--owner", choices=OWNERS)
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--queue", choices=QUEUES)
    parser.add_argument("--job", help="one assigned job ID")
    parser.add_argument("--profile", help="one profile within --job")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="validate the entire plan and show selected outputs")
    mode.add_argument("--validate", action="store_true", help="validate the entire plan without datasets")
    mode.add_argument("--pin-datasets", action="store_true", help="create the shared checkpoint 0 dataset pin")
    mode.add_argument("--prepare", action="store_true", help="verify this member's checkpoint 0")
    mode.add_argument("--approve-gate", action="store_true", help="verify and approve both complete MVTec benchmarks")
    mode.add_argument("--approve-mini", action="store_true", help="verify and approve both complete ZJU-mini headline runs")
    mode.add_argument("--check-integration", action="store_true", help="check all canonical results and their model bundles")
    parser.add_argument("--review", help="reference to the recorded verification or approval")
    parser.add_argument("--aitex-approval", help="explicit team approval reference for AITEX when pinning datasets")
    parser.add_argument("--bundle-location", help="private durable storage prefix for this member's model bundles")
    parser.add_argument("--incoming", action="append", type=Path, help="isolated export root for integration, repeat for each member")
    args = parser.parse_args(argv)
    if args.profile and not args.job:
        parser.error("--profile requires --job")
    if args.aitex_approval and not args.pin_datasets:
        parser.error("--aitex-approval requires --pin-datasets")
    if args.incoming and not args.check_integration:
        parser.error("--incoming requires --check-integration")
    if (args.pin_datasets or args.prepare or args.approve_gate or args.approve_mini) and not (args.review or "").strip():
        parser.error("this checkpoint requires --review with a recorded verification or approval reference")
    if args.prepare and (not args.owner or not (args.bundle_location or "").strip().rstrip("/")):
        parser.error("--prepare requires --owner and --bundle-location")
    return args


def selected_runs(jobs, args):
    runs = [r for j in jobs if (not args.owner or args.owner == j.owner) and (not args.stage or args.stage == j.stage)
            and (not args.queue or args.queue == j.queue) and (not args.job or args.job == j.job_id) for r in j.runs()
            if not args.profile or args.profile == r.profile]
    if args.job and not runs:
        raise ValueError("no matching assigned job/profile for the specified owner, stage, and queue")
    return runs


def dispatch(args):
    jobs = load_plan(args.plan)
    if args.validate:
        print(f"Validated {len(jobs)} jobs and {sum(len(j.profiles) for j in jobs)} unique canonical result paths")
        return 0
    if args.dry_run:
        runs = selected_runs(jobs, args)
        for run in runs:
            j = run.job
            optional = " (requires recorded AITEX approval)" if j.queue == "aitex" else ""
            print(f"{j.owner} {j.stage}/{j.queue} {j.job_id}/{run.profile}: {run.result.as_posix()}{optional}")
        print(f"{len(runs)} profiles selected, {sum(r.job.queue == 'aitex' for r in runs)} conditional")
        return 0
    if args.pin_datasets:
        pin_datasets(jobs, training_commit(), args.review, args.aitex_approval, args.shared / "datasets.json")
        return 0
    pin, active = load_pin(jobs, args.shared)
    if args.approve_gate or args.approve_mini:
        approve(active, pin, args.shared, "gate" if args.approve_gate else "mini", args.review)
        return 0
    if args.check_integration:
        return check_integration(jobs, active, pin, args.shared, args.incoming or [Path(".")])
    if not args.owner or not (args.prepare or args.stage or args.job or args.queue):
        raise ValueError("execution requires --owner plus --stage, --queue or --job")
    if training_commit() != pin["source_commit"]:
        raise ValueError("checkout differs from checkpoint 0's pinned source commit")
    if args.prepare:
        prepare([j for j in active if j.owner == args.owner], pin, args.plan, args.owner, args.review, args.bundle_location)
        return 0
    runs = selected_runs(active, args)
    if not runs:
        if args.queue == "aitex" and not pin.get("aitex_approval"):
            print("AITEX is pending a recorded team inclusion decision")
        else:
            print("No active profiles in this queue")
        return 0
    machine = hardware(args.owner)
    ready = read_json(WORK / f"ready-{args.owner}.json")
    if (any(ready.get(k) != v for k, v in ready_context(pin, args.plan, args.owner, machine).items())
            or not ready.get("environment_review") or not ready.get("bundle_location")):
        raise ValueError("checkpoint 0 readiness is incomplete or stale. Rerun --prepare for this member.")
    check_inputs([r.job for r in runs], pin)
    failed = False
    for run in runs:
        def prerequisite():
            if run.job.stage != "gate":
                check_approval(active, pin, args.shared, "gate")
            if run.job.queue == "zju_full" or (run.job.queue == "mini" and run.job.stage == "ablation"):
                check_approval(active, pin, args.shared, "mini")

        label = f"{run.job.job_id}/{run.profile}"
        print(f"Checking {label}", flush=True)
        try:
            print(f"{label}: {run_profile(run, pin, ready['bundle_location'], machine, prerequisite)}", flush=True)
        except COMMON_ERRORS as exc:
            status = WORK / "profiles" / run.result.relative_to("results").with_suffix("") / "status.json"
            print(f"{label}: {exc}. See {status} for the attempt ID and logs.", file=sys.stderr)
            failed = True
    return int(failed)


def main(argv=None):
    args = parse_args(argv)
    try:
        return dispatch(args)
    except COMMON_ERRORS as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted. Rerun the same profile after checking its saved logs.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
