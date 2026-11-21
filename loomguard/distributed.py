"""Plan validation and evidence for LoomGuard's distributed study."""
import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import yaml

from loomguard.data.manifest import MVTEC_TEXTURES, ZJU_GROUPS, zju_group
from loomguard.runs import bundle_dir, load_config, result_path

OWNERS = ("A", "B", "C")
STAGES = ("gate", "study", "ablation")
QUEUES = ("gate", "mini", "standard", "zju_full", "aitex")
FABRICS = {
    "mvtec": MVTEC_TEXTURES,
    "raw_fabrid": tuple(f"Rollo{i}A" for i in range(1, 8)),
    "zju_leaper": tuple(f"pattern{i}" for group in ZJU_GROUPS.values() for i in group),
    "aitex": tuple(f"fabric{i:02}" for i in range(7)),
}
GATE_CONFIGS = {"configs/mvtec_rd4ad.yaml": "A", "configs/mvtec_patchcore.yaml": "B"}
GATE_TARGETS = {
    "patchcore": {"tile_auroc": 97.9, "pixel_auroc": 96.6, "aupro_30": 90.7},
    "rd4ad": {"tile_auroc": 98.5, "pixel_auroc": 96.7, "aupro_30": 92.0},
}
BUNDLE_FILES = ("config.yaml", "calibration.json", "metrics.json", "model.pt")
METRICS = ("tile_auroc", "tile_ap", "image_auroc", "image_ap", "pixel_auroc", "pixel_ap", "aupro_30",
           "false_alarms_per_1000", "tile_recall", "dice", "iou")


@dataclass(frozen=True)
class Job:
    job_id: str
    owner: str
    config: str
    stage: str
    queue: str
    cfg: dict
    profiles: tuple[str, ...]

    def runs(self):
        return (Run(self, name) for name in self.profiles)


@dataclass(frozen=True)
class Run:
    job: Job
    profile: str

    @property
    def cfg(self):
        return self.job.cfg

    @property
    def result(self):
        return result_path(self.cfg, self.profile)

    @property
    def bundle(self):
        return bundle_dir(self.cfg, self.profile)


def expected_fabrics(cfg):
    full = FABRICS.get(cfg["dataset"])
    if full is None:
        raise ValueError(f"unknown dataset {cfg['dataset']!r}")
    fabrics = cfg["fabrics"]
    if fabrics is not None and (
        not isinstance(fabrics, list) or not fabrics or any(f not in full for f in fabrics)
        or len(set(fabrics)) != len(fabrics)
    ):
        raise ValueError(f"{cfg['config']}: invalid fabrics {fabrics!r}")
    return tuple(fabrics or full)


def expected_profiles(cfg):
    """Profile names from the documented inventory, without reading or filtering image rows."""
    fabrics = expected_fabrics(cfg)
    regime = cfg["regime"]
    if regime == "unified":
        return ("unified",)
    if regime == "held_out":
        return tuple(f"{f}_held_out" for f in fabrics)
    if regime == "per_group":
        if cfg["dataset"] != "zju_leaper":
            raise ValueError("per_group is defined for zju_leaper only")
        return tuple(dict.fromkeys(map(zju_group, fabrics)))
    return fabrics


def study_configs():
    """The legacy lists remain the source of truth for study coverage and phases."""
    specs = {p: ("gate", "gate") for p in GATE_CONFIGS}
    lists = {"01_zju_mini": ("study", "mini"), "02_raw_fabrid": ("study", "standard"),
             "03_aitex": ("study", "aitex"), "04_zju_full": ("study", "zju_full"),
             "05_ablations": ("ablation", "standard")}
    for name, phase in lists.items():
        for line in Path(f"configs/studies/{name}.txt").read_text().splitlines():
            config = line.partition("#")[0].strip()
            if config:
                if name == "05_ablations" and load_config(config)["dataset"] == "zju_leaper":
                    phase_for_config = ("ablation", "mini")
                else:
                    phase_for_config = phase
                if config in specs:
                    raise ValueError(f"duplicate config in study lists: {config}")
                specs[config] = phase_for_config
    return specs


def validate_config(cfg):
    c = cfg[cfg["method"]]
    integers = (cfg["hist_bins"], c["batch_size"], cfg["seed"], cfg["num_workers"])
    positive = ((c["max_tiles"], c["proj_dim"]) if cfg["method"] == "patchcore"
                else (c["iterations"], c["val_tiles"], c["val_every"], c["log_every"]))
    ratio = c["coreset_ratio"] if cfg["method"] == "patchcore" else c["lr"]
    if (any(type(v) is not int or v < 0 for v in integers) or min(integers[:2]) < 1
            or any(type(v) is not int or v < 1 for v in positive) or not finite(ratio, 0) or ratio == 0
            or (cfg["method"] == "patchcore" and ratio > 1)
            or not finite(cfg["target_far"], 0, 1) or cfg["target_far"] in (0, 1)
            or cfg["device"] not in ("auto", "cuda")):
        raise ValueError(f"{cfg['config']}: invalid training or calibration settings")


def load_plan(path):
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or doc.get("version") != 1 or not isinstance(doc.get("jobs"), list):
        raise ValueError("expected assignment plan version 1 with a jobs list")
    specs = study_configs()
    configs = {path: load_config(path) for path in specs}
    for cfg in configs.values():
        validate_config(cfg)
        if (cfg["results"] != "results" or cfg["artifacts"] != "artifacts"
                or not isinstance(cfg["tag"], str) or not re.fullmatch(r"[a-z0-9_]*", cfg["tag"])
                or cfg["held_out_scope"] not in ("all", "group")
                or (cfg["held_out_scope"] == "group" and cfg["dataset"] != "zju_leaper")):
            raise ValueError(f"{cfg['config']}: invalid study paths, tag or held-out scope")
        if cfg["config"] in GATE_CONFIGS and (
            cfg["dataset"] != "mvtec" or cfg["config"] != f"configs/mvtec_{cfg['method']}.yaml"
            or set(expected_profiles(cfg)) != set(MVTEC_TEXTURES) or cfg["regime"] != "per_fabric"
            or cfg["tag"] or cfg["targets"] != GATE_TARGETS[cfg["method"]]
        ):
            raise ValueError(f"{cfg['config']}: invalid MVTec gate configuration")
    jobs, ids, results, bundles = [], set(), {}, {}
    covered = set()
    for entry in doc["jobs"]:
        if not isinstance(entry, dict):
            raise ValueError("job must be a mapping")
        job_id, owner, path = (entry.get(k) for k in ("job_id", "owner", "config"))
        if not isinstance(job_id, str) or not re.fullmatch(r"[a-z0-9_-]+", job_id) or job_id in ids:
            raise ValueError(f"invalid or duplicate job ID: {job_id!r}")
        if not isinstance(path, str):
            raise ValueError(f"{job_id}: config must be a canonical path")
        if owner not in OWNERS or (path in GATE_CONFIGS and owner != GATE_CONFIGS[path]):
            raise ValueError(f"{job_id}: invalid or conflicting owner {owner!r}")
        if path not in specs or (entry.get("stage"), entry.get("queue")) != specs[path]:
            raise ValueError(f"{job_id}: unknown config or incorrect stage/queue: {path!r}")
        cfg = configs[path]
        full = expected_profiles(cfg)
        selection = entry.get("profiles")
        names = full if selection == "all" else selection
        if (not isinstance(names, (tuple, list)) or not names or any(n not in full for n in names)
                or len(set(names)) != len(names)):
            raise ValueError(f"{job_id}: invalid or duplicate profiles {names!r}; expected names from {full}")
        stage, queue = specs[path]
        job = Job(job_id, owner, path, stage, queue, cfg, tuple(names))
        paths = [r.result.as_posix() for r in job.runs()]
        if entry.get("expected_result_paths") != paths:
            raise ValueError(f"{job_id}: expected_result_paths must match {paths}")
        for run in job.runs():
            for resolved, seen in ((run.result.resolve(), results), (run.bundle.resolve(), bundles)):
                if resolved in seen:
                    raise ValueError(f"duplicate path {resolved}: {seen[resolved]} and {job_id} ({owner})")
                seen[resolved] = f"{job_id} ({owner})"
            covered.add((path, run.profile))
        ids.add(job_id)
        jobs.append(job)
    missing = {(path, n) for path, cfg in configs.items() for n in expected_profiles(cfg)} - covered
    if missing:
        raise ValueError(f"assignment plan is missing config/profile pairs: {sorted(missing)}")
    return jobs


def read_json(path):
    def object_pairs(pairs):
        out = dict(pairs)
        if len(out) != len(pairs):
            raise ValueError(f"{path}: repeated JSON key")
        return out

    def invalid_constant(value):
        raise ValueError(f"{path}: invalid JSON number {value}")

    obj = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=object_pairs, parse_constant=invalid_constant)
    if not isinstance(obj, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return obj


def write_json(path, value, *, replace=False):
    """Publish complete JSON atomically, requiring explicit permission to replace a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        if replace:
            os.replace(temp, path)
        else:
            os.link(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


def digest_obj(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def digest_file(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def digest_tree(root):
    root = Path(root)
    files = sorted((p for p in root.rglob("*") if p.is_file()), key=lambda p: p.relative_to(root).as_posix())
    if not files:
        raise ValueError(f"{root}: dataset directory is empty or missing")
    return digest_obj([(p.relative_to(root).as_posix(), digest_file(p)) for p in files])


def finite(value, low=None, high=None):
    return (type(value) in (int, float) and math.isfinite(value) and (low is None or value >= low)
            and (high is None or value <= high))


def identity(run, pin):
    c = run.cfg
    return {"job_id": run.job.job_id, "owner": run.job.owner, "profile": run.profile, "config": run.job.config,
            "result_path": run.result.as_posix(), "bundle_path": run.bundle.as_posix(), "method": c["method"],
            "dataset": c["dataset"], "source_commit": pin["source_commit"], "config_sha256": digest_obj(c),
            "dataset_sha256": pin["datasets"][c["dataset"]], "manifest_sha256": pin["manifests"][run.job.config],
            "split_seed": c["seed"], "fit_seed": c["rd4ad"]["seed"] if c["method"] == "rd4ad" else c["seed"]}


def verify_metrics(values, path):
    if not isinstance(values, dict):
        raise ValueError(f"{path}: expected a metrics object")
    for key in METRICS:
        upper = 1000 if key == "false_alarms_per_1000" else 1
        if key not in values or (values[key] is not None and not finite(values[key], 0, upper)):
            raise ValueError(f"{path}: invalid or missing {key}")


def verify_result(run, path, pin, *, provenance=True):
    res = read_json(path)
    c = run.cfg
    expected = {k: c[k] for k in ("dataset", "regime", "method", "tag", "seed", "config", "hist_bins")}
    expected["fabric"] = run.profile
    if any(res.get(k) != v for k, v in expected.items()):
        raise ValueError(f"{path}: result config, seed or profile mismatch")
    short = res.get("git_commit")
    if not isinstance(short, str) or len(short) < 7 or not pin["source_commit"].startswith(short):
        raise ValueError(f"{path}: result source commit mismatch")
    for k in ("n_train_tiles", "n_val_tiles", "n_test_good", "n_test_defect"):
        if type(res.get(k)) is not int or res[k] < (1 if k in ("n_train_tiles", "n_val_tiles") else 0):
            raise ValueError(f"{path}: invalid {k}")
    if not res["n_test_good"] + res["n_test_defect"]:
        raise ValueError(f"{path}: empty test pool")
    verify_metrics(res, path)
    if any(res[k] is None for k in ("tile_auroc", "pixel_auroc", "aupro_30")):
        raise ValueError(f"{path}: undefined headline metrics")
    for k in ("tau", "peak_vram_gb", "peak_ram_gb", "fit_minutes", "bundle_mb", "ms_per_tile_cuda"):
        if not finite(res.get(k), 0):
            raise ValueError(f"{path}: invalid {k}")
    if res.get("device") != "cuda":
        raise ValueError(f"{path}: expected a CUDA result")
    if c["method"] == "rd4ad":
        if (res.get("iterations") != c["rd4ad"]["iterations"] or not finite(res.get("iters_per_second"), 0)
                or not finite(res.get("final_train_loss")) or not finite(res.get("final_val_loss"))):
            raise ValueError(f"{path}: invalid RD4AD training evidence")
    elif (res.get("coreset_ratio") != c["patchcore"]["coreset_ratio"] or not finite(res.get("coreset_seconds"), 0)
          or type(res.get("bank_size")) is not int or res["bank_size"] < 1):
        raise ValueError(f"{path}: invalid PatchCore fit evidence")
    if c["regime"] in ("unified", "per_group"):
        expected_test = {f for f in expected_fabrics(c) if c["regime"] == "unified" or zju_group(f) == run.profile}
        by_fabric = res.get("by_fabric", {})
        if not isinstance(by_fabric, dict) or set(by_fabric) != expected_test:
            raise ValueError(f"{path}: incomplete per-fabric results")
        for fabric, metrics in by_fabric.items():
            verify_metrics(metrics, f"{path} ({fabric})")
    if c["grade_rolls"]:
        expected_rolls = expected_fabrics(c) if run.profile == "unified" else (run.profile.removesuffix("_held_out"),)
        rolls = res.get("rolls", {})
        if not isinstance(rolls, dict) or set(rolls) != set(expected_rolls) or any(
            not finite(res.get(k), 0) for k in ("grade_agreement", "points_error_mean", "defect_count_error_mean")
        ) or res["grade_agreement"] > 1:
            raise ValueError(f"{path}: missing roll grading")
        for fabric, roll in rolls.items():
            quantities = ("frames", "defects_pred", "defects_gt", "points_pred", "points_gt",
                          "points_per_100_pred", "points_per_100_gt")
            if (not isinstance(roll, dict) or not all(finite(roll.get(k), 0) for k in quantities)
                    or roll["frames"] < 1
                    or any(roll.get(k) not in ("A", "B", "C", "Reject") for k in ("grade_pred", "grade_gt"))):
                raise ValueError(f"{path}: invalid roll grading for {fabric}")
    if provenance:
        evidence = res.get("distributed")
        if not isinstance(evidence, dict) or any(evidence.get(k) != v for k, v in identity(run, pin).items()):
            raise ValueError(f"{path}: missing or conflicting distributed provenance")
        hardware = evidence.get("hardware")
        files = evidence.get("bundle_sha256")
        if (not isinstance(hardware, dict) or not all(hardware.get(k) for k in ("machine", "gpu", "driver", "torch", "cuda"))
                or not evidence.get("bundle_location") or not isinstance(files, dict) or set(files) != set(BUNDLE_FILES)
                or any(not isinstance(h, str) or not re.fullmatch(r"[0-9a-f]{64}", h) for h in files.values())):
            raise ValueError(f"{path}: incomplete hardware or bundle manifest")
    return res


def bundle_hashes(run, bundle, res):
    for name in BUNDLE_FILES:
        if not (bundle / name).is_file() or (bundle / name).stat().st_size == 0:
            raise ValueError(f"{bundle}: missing or empty {name}")
    saved = yaml.safe_load((bundle / "config.yaml").read_text(encoding="utf-8"))
    if not isinstance(saved, dict) or any(saved.get(k) != v for k, v in run.cfg.items()):
        raise ValueError(f"{bundle}: fitted configuration mismatch")
    if saved.get("profile") != run.profile or saved.get("git_commit") != res["git_commit"]:
        raise ValueError(f"{bundle}: fitted profile or source commit mismatch")
    cal = read_json(bundle / "calibration.json")
    if (cal.get("score") != "heatmap_max" or cal.get("tau") != res["tau"]
            or cal.get("n_val_tiles") != res["n_val_tiles"] or cal.get("target_far") != run.cfg["target_far"]
            or type(cal.get("low_confidence")) is not bool):
        raise ValueError(f"{bundle}: invalid calibration")
    fit = read_json(bundle / "metrics.json")
    keys = ["n_train_tiles", "peak_vram_gb", "peak_ram_gb", "fit_minutes", "bundle_mb"]
    keys += ["iterations", "final_train_loss", "final_val_loss"] if run.cfg["method"] == "rd4ad" else ["bank_size", "coreset_ratio"]
    if any(fit.get(k) != res[k] for k in keys):
        raise ValueError(f"{bundle}: fit metrics mismatch")
    return {name: digest_file(bundle / name) for name in BUNDLE_FILES}


def verify_bundle(run, bundle, res):
    evidence = res["distributed"]
    if (read_json(bundle / "provenance.json") != evidence
            or bundle_hashes(run, bundle, res) != evidence["bundle_sha256"]):
        raise ValueError(f"{bundle}: bundle checksum or provenance mismatch")
