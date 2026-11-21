"""Distributed queues and checkpoint handling without real datasets or CUDA."""
import copy
from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from loomguard.distributed import GATE_TARGETS, METRICS, digest_file, digest_tree, expected_fabrics, expected_profiles
from loomguard.distributed import load_plan, read_json, write_json
from loomguard.runs import build_rows, load_config, manifest_digest, profiles, result_path, selected_profiles
from loomguard.data.manifest import zju_group

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import distributed_study as runner  # noqa: E402
import evaluate  # noqa: E402
import summarize  # noqa: E402


@pytest.fixture
def jobs():
    return load_plan(ROOT / runner.PLAN)


def test_initial_assignment_and_complete_canonical_coverage(jobs):
    ownership = {
        "A": """mvtec_rd4ad raw_fabrid_per_fabric_rd4ad raw_fabrid_rd4ad_s0 raw_fabrid_rd4ad_s1
                raw_fabrid_rd4ad_s2 raw_fabrid_held_out_rd4ad""",
        "B": """mvtec_patchcore zju_per_fabric_mini_patchcore raw_fabrid_per_fabric_patchcore raw_fabrid_patchcore
                raw_fabrid_held_out_patchcore zju_per_fabric_rest_patchcore zju_per_group_patchcore zju_unified_patchcore
                zju_held_out_patchcore aitex_per_fabric_patchcore aitex_per_fabric_rd4ad zju_mini_patchcore_cr1
                zju_mini_patchcore_cr25 zju_res256_patchcore raw_fabrid_per_fabric_patchcore_cr1
                raw_fabrid_per_fabric_patchcore_cr25 raw_fabrid_per_fabric_patchcore_n50
                raw_fabrid_per_fabric_patchcore_n200 raw_fabrid_per_fabric_patchcore_ov32
                zju_per_fabric_mini_rd4ad zju_per_fabric_rest_rd4ad""",
        "C": """zju_per_group_rd4ad zju_unified_rd4ad zju_held_out_rd4ad raw_fabrid_per_fabric_rd4ad_it1k
                raw_fabrid_per_fabric_rd4ad_ov32 zju_res256_rd4ad""",
    }
    for owner, names in ownership.items():
        assert {j.job_id for j in jobs if j.owner == owner} == set(names.split())
    results = [r.result for j in jobs for r in j.runs()]
    assert len(results) == len(set(results)) == 185
    assert len([r for j in jobs if j.queue != "aitex" for r in j.runs()]) == 171


@pytest.mark.parametrize("invalid,match", [
    ("unknown_profile", "invalid or duplicate profiles"),
    ("expected_path", "expected_result_paths must match"),
    ("duplicate_path", "duplicate path"),
    ("duplicate_id", "duplicate job ID"),
    ("wrong_gate_owner", "conflicting owner"),
    ("missing", "missing config/profile pairs"),
    ("invalid_cfg", "unknown config or incorrect stage"),
])
def test_invalid_plans(tmp_path, invalid, match):
    plan = yaml.safe_load((ROOT / runner.PLAN).read_text())
    entry = plan["jobs"][0]
    if invalid == "unknown_profile":
        entry["profiles"][0] = "wrong"
    elif invalid == "expected_path":
        entry["expected_result_paths"][0] = "results/wrong.json"
    elif invalid == "duplicate_path":
        old = next(j for j in plan["jobs"] if j["job_id"] == "zju_per_group_rd4ad")
        plan["jobs"].append(copy.deepcopy(old) | {"job_id": "duplicate", "owner": "B"})
    elif invalid == "duplicate_id":
        plan["jobs"].append(copy.deepcopy(entry))
    elif invalid == "wrong_gate_owner":
        entry["owner"] = "C"
    elif invalid == "invalid_cfg":
        entry["config"] = "configs/unknown.yaml"
    else:
        plan["jobs"].pop()
    path = tmp_path / "plan.yaml"
    path.write_text(yaml.safe_dump(plan))
    with pytest.raises(ValueError, match=match):
        load_plan(path)


def test_rebalanced_shards_match_unsharded_paths(tmp_path):
    plan = yaml.safe_load((ROOT / runner.PLAN).read_text())
    job = next(j for j in plan["jobs"] if j["job_id"] == "raw_fabrid_held_out_rd4ad")
    old_paths = set(job["expected_result_paths"])
    moved = dict(job, job_id="moved_held_out", owner="C",
                 profiles=job["profiles"][:2], expected_result_paths=job["expected_result_paths"][:2])
    for key in ("profiles", "expected_result_paths"):
        job[key] = job[key][2:]
    plan["jobs"].append(moved)
    path = tmp_path / "sharded.yaml"
    path.write_text(yaml.safe_dump(plan))
    shards = [j for j in load_plan(path) if j.config == job["config"]]
    assert {r.result.as_posix() for j in shards for r in j.runs()} == old_paths
    cfg = load_config(job["config"])
    assert {result_path(cfg, n).as_posix() for n in expected_profiles(cfg)} == old_paths
    assert {j.owner for j in shards} == {"A", "C"}


def test_runner_filters_owner_stage_and_profile(capsys):
    args = ["--dry-run", "--owner", "C", "--stage", "study", "--job", "zju_per_group_rd4ad"]
    assert runner.main([*args, "--profile", "group1"]) == 0
    assert "C study/zju_full zju_per_group_rd4ad/group1" in capsys.readouterr().out
    assert runner.main([*args, "--profile", "group6"]) == 1
    assert runner.main(["--dry-run", "--owner", "A", "--job", "zju_per_group_rd4ad"]) == 1


@pytest.mark.parametrize(("dataset", "regime", "profile", "train_fabrics"), [
    ("raw_fabrid", "held_out", "RolloA_held_out", {"RolloB"}),
    ("zju_leaper", "held_out", "pattern2_held_out", {"pattern1"}),
    ("zju_leaper", "per_group", "group1", {"pattern1", "pattern2"}),
    ("zju_leaper", "unified", "unified", {"pattern1", "pattern2", "pattern4", "pattern5"}),
])
def test_complete_training_pools(request, tmp_path, dataset, regime, profile, train_fabrics):
    root = request.getfixturevalue("raw_root" if dataset == "raw_fabrid" else "zju_root")
    f = tmp_path / "experiment.yaml"
    cfg = {"dataset": dataset, "root": str(root), "regime": regime, "method": "patchcore"}
    if dataset == "zju_leaper" and regime == "held_out":
        cfg.update(held_out_scope="group", fabrics=["pattern2", "pattern5"])
    f.write_text(yaml.safe_dump(cfg))
    cfg = load_config(f)
    rows = build_rows(cfg)
    all_profiles = profiles(rows, cfg)
    shard = selected_profiles(rows, cfg, profile)
    assert selected_profiles(rows, cfg, None) == all_profiles
    assert list(shard) == [profile] and shard[profile] == all_profiles[profile]
    assert {r.fabric for r in shard[profile] if r.split in ("train", "val")} == train_fabrics
    assert {result_path(cfg, p) for n in all_profiles for p in selected_profiles(rows, cfg, n)} == {
        result_path(cfg, n) for n in all_profiles
    }
    for bad in (f"{profile},{profile}", f"{profile},", "missing", ""):
        with pytest.raises(ValueError):
            selected_profiles(rows, cfg, bad)


def test_fit_and_evaluate_cli_can_run_a_single_complete_profile(tmp_path, mvtec_root):
    path = tmp_path / "tiny.yaml"
    path.write_text(yaml.safe_dump({"dataset": "mvtec", "root": str(mvtec_root), "regime": "per_fabric",
                                   "method": "patchcore", "backbone_weights": None, "device": "cpu",
                                   "patchcore": {"max_tiles": 4, "coreset_ratio": 0.005},
                                   "targets": {"tile_auroc": 0}}))
    rows = build_rows(load_config(path))
    base = ["--config", str(path), "--profiles", "wood", "--output-root", str(tmp_path / "stage")]
    env = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
    bad = subprocess.run([sys.executable, "scripts/fit.py", *base, "--manifest-sha256", "wrong"],
                         cwd=ROOT, env=env, capture_output=True, text=True)
    assert bad.returncode and "manifest differs" in bad.stderr and not (tmp_path / "stage").exists()
    for script in ("fit", "evaluate"):
        got = subprocess.run([sys.executable, f"scripts/{script}.py", *base, "--manifest-sha256", manifest_digest(rows)],
                             cwd=ROOT, env=env, check=True, capture_output=True, text=True)
        assert "PASS" not in got.stdout and "FAIL" not in got.stdout
    stage = tmp_path / "stage"
    assert sorted(p.stem for p in (stage / "results").rglob("*.json")) == ["wood"]
    assert sorted(p.parent.parent.name for p in (stage / "artifacts").rglob("model.pt")) == ["wood"]
    assert read_json(stage / "results/mvtec/per_fabric/patchcore/wood.json")["config"] == str(path)


def test_only_unsharded_five_texture_evaluation_prints_benchmark(tmp_path, mvtec_root, monkeypatch, capsys):
    config = tmp_path / "tiny.yaml"
    config.write_text(yaml.safe_dump({"dataset": "mvtec", "root": str(mvtec_root), "regime": "per_fabric",
                                     "method": "patchcore", "targets": {"tile_auroc": 97.9}}))
    monkeypatch.setattr(evaluate, "evaluate_profile", lambda *args: {k: 1.0 for k in evaluate.KEYS})
    base = ["--config", str(config)]
    assert len(evaluate.main([*base, "--profiles", "tile,wood"])) == 2
    assert "PASS" not in capsys.readouterr().out
    assert len(evaluate.main(base)) == 5
    assert "PASS" in capsys.readouterr().out


def pin_for(jobs):
    return {"source_commit": "a" * 40, "datasets": {j.cfg["dataset"]: "b" * 64 for j in jobs},
            "manifests": {j.config: "c" * 64 for j in jobs}}


MACHINE = {"machine": "cpu-test", "gpu": "simulated", "driver": "test", "torch": "test", "cuda": "test", "native_bf16": True}


def fake_job_outputs(run, root, manifest):
    c = run.cfg
    res = {k: c[k] for k in ("dataset", "regime", "method", "tag", "seed", "config", "hist_bins")}
    res.update({k: 0.99 for k in METRICS})
    res.update(fabric=run.profile, git_commit="aaaaaaa", tau=0.1, n_train_tiles=100, n_val_tiles=50, n_test_good=3,
               n_test_defect=4, peak_vram_gb=10, peak_ram_gb=2, fit_minutes=1, bundle_mb=0.1, ms_per_tile_cuda=3,
               false_alarms_per_1000=10, device="cuda")
    if c["method"] == "rd4ad":
        res.update(iterations=c["rd4ad"]["iterations"], iters_per_second=4, final_train_loss=0.2, final_val_loss=0.3)
    else:
        res.update(coreset_ratio=c["patchcore"]["coreset_ratio"], bank_size=10, coreset_seconds=20)
    if c["regime"] in ("unified", "per_group"):
        res["by_fabric"] = {f: {k: res[k] for k in METRICS} for f in expected_fabrics(c)
                            if c["regime"] == "unified" or zju_group(f) == run.profile}
    if c["grade_rolls"]:
        names = expected_fabrics(c) if run.profile == "unified" else (run.profile.removesuffix("_held_out"),)
        roll = {"frames": 1, "defects_pred": 0, "defects_gt": 0, "points_pred": 0, "points_gt": 0,
                "points_per_100_pred": 0, "points_per_100_gt": 0, "grade_pred": "A", "grade_gt": "A"}
        res.update(rolls={f: roll for f in names}, grade_agreement=1, points_error_mean=0, defect_count_error_mean=0)
    write_json(root / run.result, res)
    out = root / run.bundle
    write_json(out / "metrics.json", res)
    write_json(out / "calibration.json", {"score": "heatmap_max", "tau": res["tau"], "n_val_tiles": res["n_val_tiles"],
                                          "target_far": c["target_far"], "low_confidence": True})
    (out / "config.yaml").write_text(yaml.safe_dump({**c, "profile": run.profile, "git_commit": res["git_commit"]}))
    (out / "model.pt").write_bytes(b"simulated model")


def no_prerequisites():
    pass


def status_path(run):
    return runner.WORK / "profiles" / run.result.relative_to("results").with_suffix("") / "status.json"


def test_verified_completion_skip_and_conflicts(tmp_path, monkeypatch, jobs):
    monkeypatch.chdir(tmp_path)
    run = next(jobs[0].runs())
    pin = pin_for(jobs)
    calls = []

    def train(*args):
        calls.append(args)
        fake_job_outputs(*args)

    monkeypatch.setattr(runner, "train_and_evaluate", train)
    execute = lambda current_pin=pin: runner.run_profile(run, current_pin, "private://store", MACHINE, no_prerequisites)
    # Existing JSON alone must block without training or replacing anything.
    write_json(run.result, {})
    with pytest.raises(ValueError, match="mismatch"):
        execute()
    assert read_json(run.result) == {} and not calls
    run.result.unlink()
    assert execute().startswith("published") and execute() == "verified, skipped" and len(calls) == 1
    res = read_json(run.result)
    assert res["distributed"]["fit_seed"] == run.cfg["rd4ad"]["seed"]
    assert res["distributed"]["dataset_sha256"] == pin["datasets"][run.cfg["dataset"]]
    wrong_split = copy.deepcopy(pin)
    wrong_split["manifests"][run.job.config] = "d" * 64
    with pytest.raises(ValueError, match="conflicting distributed provenance"):
        execute(wrong_split)
    (run.bundle / "model.pt").write_bytes(b"corrupted model")
    with pytest.raises(ValueError, match="checksum"):
        execute()
    assert len(calls) == 1 and read_json(status_path(run))["state"] == "blocked"


def test_failed_job_retry_and_recover_partial_publication(tmp_path, monkeypatch, jobs):
    monkeypatch.chdir(tmp_path)
    run = next(jobs[0].runs())
    pin = pin_for(jobs)

    def fail(*args):
        raise subprocess.CalledProcessError(1, "fit")

    monkeypatch.setattr(runner, "train_and_evaluate", fail)
    execute = lambda: runner.run_profile(run, pin, "private://store", MACHINE, no_prerequisites)
    with pytest.raises(subprocess.CalledProcessError):
        execute()
    old = read_json(status_path(run))
    assert old["state"] == "failed" and not run.result.exists() and not run.bundle.exists()
    monkeypatch.setattr(runner, "train_and_evaluate", fake_job_outputs)
    link = os.link

    def interrupted_link(src, dest):
        if Path(dest) == run.result:
            raise OSError("interrupted publish")
        link(src, dest)

    monkeypatch.setattr(os, "link", interrupted_link)
    with pytest.raises(OSError, match="interrupted publish"):
        execute()
    attempt = read_json(status_path(run))
    assert attempt["attempt"] != old["attempt"] and attempt["ready_to_publish"]
    assert run.bundle.exists() and not run.result.exists()
    monkeypatch.setattr(os, "link", link)
    monkeypatch.setattr(runner, "train_and_evaluate", fail)
    assert execute() == "recovered publication" and read_json(status_path(run))["state"] == "passed"


def test_dataset_digest_is_content_based(tmp_path):
    for name in ("first", "second"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "sample").write_bytes(b"same data")
    a, b = (tmp_path / name for name in ("first", "second"))
    assert digest_tree(a) == digest_tree(b)
    (b / "sample").write_bytes(b"changed data")
    assert digest_tree(a) != digest_tree(b)


def test_checkpoint_zero_pins_and_compares_real_synthetic_manifest(tmp_path, monkeypatch, mvtec_root, jobs):
    gate = [replace(j, cfg={**j.cfg, "root": str(mvtec_root)}) for j in jobs if j.queue == "gate"]
    monkeypatch.setattr(runner, "WORK", tmp_path / "work")
    monkeypatch.setattr(runner, "hardware", lambda owner: MACHINE)
    monkeypatch.setattr(runner, "git", lambda *args: "a" * 40)
    shared = tmp_path / "shared"
    runner.pin_datasets(gate, "a" * 40, "verified layouts", None, shared / "datasets.json")
    pin, active = runner.load_pin(gate, shared)
    assert active == gate and pin["datasets"]["mvtec"] == digest_tree(mvtec_root)
    assert pin["manifests"][gate[0].config] == manifest_digest(build_rows(gate[0].cfg))
    runner.prepare([gate[0]], pin, ROOT / runner.PLAN, "A", "passed CPU tests", "private://store")
    assert read_json(runner.WORK / "ready-A.json")["hardware"] == MACHINE
    (mvtec_root / "different-file").write_bytes(b"changed")
    with pytest.raises(ValueError, match="dataset contents differ"):
        runner.prepare([gate[0]], pin, ROOT / runner.PLAN, "A", "passed CPU tests", "private://store")


def test_gate_requires_every_texture_and_accepted_thresholds(jobs):
    results = []
    for run in runner.checkpoint_runs(jobs, "gate"):
        method = run.cfg["method"]
        metrics = {k: v / 100 for k, v in GATE_TARGETS[method].items()}
        results.append({"method": method, "fabric": run.profile, **metrics, "image_auroc": metrics["tile_auroc"]})
    runner.gate_passes(runner.benchmark_means(results))
    with pytest.raises(ValueError, match="five MVTec"):
        runner.benchmark_means(results[:-1])
    results[-1]["image_auroc"] -= 0.01
    with pytest.raises(ValueError, match="missed"):
        runner.gate_passes(runner.benchmark_means(results))


def test_shared_sequence_enforced(tmp_path, monkeypatch, jobs):
    monkeypatch.chdir(tmp_path)
    pin = pin_for(jobs)
    with pytest.raises(ValueError, match="gate.json"):
        runner.approve(jobs, pin, tmp_path, "mini", "team decision")
    assert not (tmp_path / "mini.json").exists()
    with pytest.raises(FileNotFoundError):
        runner.approve(jobs, pin, tmp_path, "gate", "team decision")
    assert not (tmp_path / "gate.json").exists()

    # Exercise the runner's actual launch selection and prerequisites with synthetic setup receipts.
    monkeypatch.setattr(runner, "load_plan", lambda path: jobs)
    monkeypatch.setattr(runner, "load_pin", lambda jobs, shared: (pin, jobs))
    monkeypatch.setattr(runner, "training_commit", lambda: pin["source_commit"])
    monkeypatch.setattr(runner, "hardware", lambda owner: MACHINE)
    monkeypatch.setattr(runner, "ready_context", lambda *args: {})
    monkeypatch.setattr(runner, "check_inputs", lambda *args: None)
    monkeypatch.setattr(runner, "train_and_evaluate", fake_job_outputs)
    write_json(runner.WORK / "ready-C.json", {"bundle_location": "private://store", "environment_review": "tests passed"})
    args = ["--owner", "C", "--job", "zju_per_group_rd4ad", "--profile", "group1", "--shared", str(tmp_path)]
    assert runner.main(args) == 1 and not Path("results").exists()
    for run in runner.checkpoint_runs(jobs, "gate"):
        runner.run_profile(run, pin, "private://store", MACHINE, no_prerequisites)
    runner.approve(jobs, pin, tmp_path, "gate", "team decision")
    assert runner.main(args) == 1
    full = next(j for j in jobs if j.job_id == "zju_per_group_rd4ad")
    assert not next(full.runs()).result.exists()
    for run in runner.checkpoint_runs(jobs, "mini"):
        runner.run_profile(run, pin, "private://store", MACHINE, no_prerequisites)
    runner.approve(jobs, pin, tmp_path, "mini", "team decision")
    assert runner.main(args) == 0 and next(full.runs()).result.exists()


def test_consolidation_reports_missing_and_duplicate_results(tmp_path, monkeypatch, jobs, capsys):
    monkeypatch.chdir(tmp_path)
    pin = pin_for(jobs)
    monkeypatch.setattr(runner, "train_and_evaluate", fake_job_outputs)
    for queue in ("gate", "mini"):
        for run in runner.checkpoint_runs(jobs, queue):
            runner.run_profile(run, pin, "private://store", MACHINE, no_prerequisites)
        runner.approve(jobs, pin, tmp_path, queue, "team decision")
    duplicate = tmp_path / "copy"
    run = next(jobs[0].runs())
    write_json(duplicate / run.result, read_json(run.result))
    assert runner.check_integration(jobs, jobs, pin, tmp_path, [tmp_path, duplicate]) == 1
    err = capsys.readouterr().err
    assert f"duplicate result {run.result}" in err and "missing results/raw_fabrid" in err
    assert digest_file(run.result) == digest_file(duplicate / run.result)
    for job in jobs:
        for run in job.runs():
            if not run.result.exists():
                runner.run_profile(run, pin, "private://store", MACHINE, no_prerequisites)
    assert runner.check_integration(jobs, jobs, pin, tmp_path, [tmp_path]) == 0
    assert summarize.main(["--check"]) == 0
    assert summarize.main([]) == 0 and (tmp_path / "results/report/all_results.csv").is_file()
