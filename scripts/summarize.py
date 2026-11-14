"""Turn results/**/*.json into the tables and plots of the study (markdown, CSV and plain SVG, no plotting library).

    python scripts/summarize.py [--results results] [--out results/report] [--check]

--check only lists the cells of EVALUATION 4.3 that have no result file for a method and exits 1 if there are any.
Writes regimes.md, ablations.md, all_results.csv and figures/*.svg under --out.
"""
import argparse
import csv
import json
import sys
from pathlib import Path
from xml.sax.saxutils import escape

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # also runnable without the package being installed
from loomguard.data.manifest import zju_group  # noqa: E402

METHODS = ("patchcore", "rd4ad")
ROLLS = [f"Rollo{i}A" for i in range(1, 8)]
PATTERNS = [f"pattern{n}" for n in range(1, 20)]
HELD_ZJU = ["pattern3", "pattern7", "pattern12", "pattern15", "pattern19"]
# EVALUATION 4.3: (dataset, regime, expected result names; an int means "at least that many files")
CELLS = [("raw_fabrid", "per_fabric", ROLLS), ("raw_fabrid", "unified", ["unified"]),
         ("raw_fabrid", "held_out", [f"{r}_held_out" for r in ROLLS]),
         ("zju_leaper", "per_fabric", PATTERNS), ("zju_leaper", "per_group", [f"group{g}" for g in range(1, 6)]),
         ("zju_leaper", "unified", ["unified"]), ("zju_leaper", "held_out", [f"{p}_held_out" for p in HELD_ZJU]),
         ("aitex", "per_fabric", 7)]
# EVALUATION 4.4: name -> (dataset, regime, {method: [(variant label, method directory)]})
PC_SIZE = [("50 tiles", "patchcore_n50"), ("200 tiles", "patchcore_n200"), ("600 tiles (default)", "patchcore")]
CORESET = {"patchcore": [("1%", "patchcore_cr1"), ("10% (default)", "patchcore"), ("25%", "patchcore_cr25")]}
ABLATIONS = {
    "resolution": ("zju_leaper", "per_fabric", {m: [("native 256 tiles (default)", m), ("whole image resized to 256", f"{m}_res256")]
                                                for m in METHODS}),
    "coreset_zju_mini": ("zju_leaper", "per_fabric", CORESET),
    "coreset_raw_fabrid": ("raw_fabrid", "per_fabric", CORESET),
    "train_size": ("raw_fabrid", "per_fabric", {"patchcore": PC_SIZE, "rd4ad": [("1k iterations", "rd4ad_it1k"),
                                                                                  ("5k iterations (default)", "rd4ad")]}),
    "overlap": ("raw_fabrid", "per_fabric", {m: [("0 px (default)", m), ("32 px", f"{m}_ov32")] for m in METHODS}),
    "seeds": ("raw_fabrid", "unified", {"rd4ad": [(f"seed {i}", f"rd4ad_s{i}") for i in range(3)]}),
}
COLORS = {"patchcore": "#2a6fdb", "rd4ad": "#e8833a"}


def headline(dataset: str, regime: str, method: str) -> str:
    """Method directory of the headline run: the RD4AD RAW-FABRID unified headline is seed 0 (the other seeds are an ablation)."""
    return "rd4ad_s0" if (dataset, regime, method) == ("raw_fabrid", "unified", "rd4ad") else method


def det_metric(dataset: str) -> str:
    return "image_auroc" if dataset == "zju_leaper" else "tile_auroc"  # ZJU-Leaper detects per image (EVALUATION 4.2)


def load_results(results: Path) -> dict:
    out = {}
    for p in sorted(Path(results).rglob("*.json")):
        res = json.loads(p.read_text())
        if isinstance(res, dict) and {"dataset", "regime", "method", "fabric"} <= set(res):
            out[(res["dataset"], res["regime"], p.parent.name, p.stem)] = res
    return out


def missing_cells(R: dict) -> list[str]:
    out = []
    for ds, regime, names in CELLS:
        for m in METHODS:
            d = headline(ds, regime, m)
            have = {k[3] for k in R if k[:3] == (ds, regime, d)}
            lack = sorted(set(names) - have) if isinstance(names, list) else ([f"only {len(have)} of {names} files"] if len(have) < names else [])
            if lack:
                out.append(f"{ds}/{regime}/{d}: missing {', '.join(lack)}")
    return out


def _f(v) -> str:
    return "-" if v is None else f"{v:.3f}"


def _mean(vals) -> float | None:
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def _table(header: list[str], rows: list[list[str]]) -> str:
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in rows])


def regime_tables(R: dict) -> str:
    out = ["# Regimes (headline settings)\n"]
    for ds, regime in sorted({k[:2] for k in R}):
        det = det_metric(ds)
        dirs = {m: headline(ds, regime, m) for m in METHODS}
        names = sorted({k[3] for k in R if k[0] == ds and k[1] == regime and k[2] in dirs.values()})
        if not names:
            continue
        header, rows, cols = ["fabric"], [], {m: [] for m in METHODS}
        for m in METHODS:
            header += [f"{m} {det}", f"{m} aupro_30", f"{m} recall", f"{m} FA/1000"]
        for n in names:
            row = [n]
            for m in METHODS:
                r = R.get((ds, regime, dirs[m], n))
                vals = [r and r.get(k) for k in (det, "aupro_30", "tile_recall", "false_alarms_per_1000")]
                cols[m].append(vals)
                row += [_f(v) if r else "missing" for v in vals]
            rows.append(row)
        rows.append(["**mean**"] + [_f(_mean(c[i] for c in cols[m])) for m in METHODS for i in range(4)])
        out.append(f"## {ds}, {regime}\n\n" + _table(header, rows) + "\n")
    out.append(cross_table(R))
    return "\n".join(out)


def cross_value(R: dict, ds: str, regime: str, method: str, f: str, metric: str):
    d = headline(ds, regime, method)
    if regime == "per_fabric":
        r = R.get((ds, regime, d, f))
    elif regime == "held_out":
        r = R.get((ds, regime, d, f"{f}_held_out"))
    else:
        r = R.get((ds, regime, d, "unified" if regime == "unified" else zju_group(f) if ds == "zju_leaper" else ""))
        r = (r or {}).get("by_fabric", {}).get(f)
    return None if r is None else r.get(metric)


def cross_table(R: dict) -> str:
    """Q2 to Q4: the same fabric under every regime (shared bundles through their by_fabric breakdown)."""
    out = ["# Same fabric under every regime (Q2, Q3, Q4)\n\nValues are the detection metric of the dataset on the test pool of that fabric; "
           "held-out values come from a model that never saw the fabric.\n"]
    for ds in sorted({k[0] for k in R}):
        fabrics = sorted({k[3].removesuffix("_held_out") for k in R if k[0] == ds and k[1] in ("per_fabric", "held_out")})
        regimes = [g for g in ("per_fabric", "per_group", "unified", "held_out") if any(k[:2] == (ds, g) for k in R)]
        if not fabrics or len(regimes) < 2:
            continue
        header = ["fabric"] + [f"{m} {g}" for m in METHODS for g in regimes]
        rows = [[f] + [_f(cross_value(R, ds, g, m, f, det_metric(ds))) for m in METHODS for g in regimes] for f in fabrics]
        out.append(f"## {ds} ({det_metric(ds)})\n\n" + _table(header, rows) + "\n")
    return "\n".join(out)


def ablation_data(R: dict, name: str) -> tuple[list[str], dict]:
    """Variants -> {method: {metric: mean over the fabrics present in every variant of that method}}."""
    ds, regime, spec = ABLATIONS[name]
    det, data, order = det_metric(ds), {}, []
    for m, variants in spec.items():
        found = {lab: {k[3]: R[k] for k in R if k[:3] == (ds, regime, d)} for lab, d in variants}
        common = set.intersection(*(set(v) for v in found.values() if v)) if any(found.values()) else set()
        for lab, _ in variants:
            if lab not in order:
                order.append(lab)
            rs = [found[lab][n] for n in sorted(common)] if found[lab] else []
            data.setdefault(lab, {})[m] = {"n": len(rs), det: _mean(r[det] for r in rs) if rs else None,
                                           "aupro_30": _mean(r["aupro_30"] for r in rs) if rs else None}
    return order, data


def bar_svg(path: Path, title: str, groups: dict[str, dict[str, float | None]]) -> None:
    """Grouped bar chart, y axis 0 to 1, plain SVG (decision A-21: no plotting dependency)."""
    series = sorted({s for v in groups.values() for s in v})
    W, H, L, B, T = 140 + 130 * len(groups), 300, 50, 70, 40
    gw = (W - L - 10) / max(len(groups), 1)
    bw = gw * 0.8 / max(len(series), 1)
    el = [f'<text x="{L}" y="22" font-size="14" font-family="sans-serif">{escape(title)}</text>']
    for i in range(0, 11, 2):
        y = H - B - (H - B - T) * i / 10
        el.append(f'<line x1="{L}" x2="{W}" y1="{y}" y2="{y}" stroke="#ddd"/><text x="{L - 6}" y="{y + 4}" font-size="10" '
                  f'text-anchor="end" font-family="sans-serif">{i / 10:.1f}</text>')
    for gi, (label, vals) in enumerate(groups.items()):
        x0 = L + gi * gw + gw * 0.1
        el.append(f'<text x="{x0 + gw * 0.4}" y="{H - B + 16}" font-size="10" text-anchor="middle" font-family="sans-serif">{escape(label)}</text>')
        for si, s in enumerate(series):
            v = vals.get(s)
            if v is None:
                continue
            h = (H - B - T) * max(0.0, min(v, 1.0))
            x = x0 + si * bw
            el.append(f'<rect x="{x}" y="{H - B - h}" width="{bw * 0.9}" height="{h}" fill="{COLORS.get(s, "#777")}"/>'
                      f'<text x="{x + bw * 0.45}" y="{H - B - h - 3}" font-size="9" text-anchor="middle" font-family="sans-serif">{v:.2f}</text>')
    for si, s in enumerate(series):
        el.append(f'<rect x="{L + si * 90}" y="{H - 22}" width="10" height="10" fill="{COLORS.get(s, "#777")}"/>'
                  f'<text x="{L + si * 90 + 14}" y="{H - 13}" font-size="10" font-family="sans-serif">{escape(s)}</text>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">'
                    f'<rect width="100%" height="100%" fill="white"/>' + "".join(el) + "</svg>")


def ablation_report(R: dict, out: Path) -> str:
    md = ["# Ablations (reported as ablations; never used to tune the headline, D-35)\n\nEach table compares the variants on the "
          "fabrics present in every variant of a method. Pixel metrics of the resolution ablation live on different pixel grids "
          "(512 against 256): compare the image-level metric.\n"]
    for name, (ds, regime, _) in ABLATIONS.items():
        order, data = ablation_data(R, name)
        det = det_metric(ds)
        used = [m for m in METHODS if any(m in data[lab] for lab in order)]
        rows = [[lab] + [c for m in used for c in ((str(data[lab][m]["n"]), _f(data[lab][m][det]), _f(data[lab][m]["aupro_30"]))
                                                   if m in data[lab] else ("-", "-", "-"))] for lab in order]
        header = ["variant"] + [c for m in used for c in (f"{m} fabrics", f"{m} {det}", f"{m} aupro_30")]
        md.append(f"## {name} ({ds}, {regime})\n\n" + _table(header, rows) + "\n")
        for metric in (det, "aupro_30"):
            bar_svg(out / "figures" / f"ablation_{name}_{metric}.svg", f"{name}: {metric}",
                    {lab: {m: data[lab][m][metric] for m in data[lab]} for lab in order})
    return "\n".join(md)


def write_csv(R: dict, path: Path) -> None:
    flat = [{"method_dir": k[2], **{a: b for a, b in v.items() if not isinstance(b, (dict, list))}} for k, v in R.items()]
    cols = sorted({c for r in flat for c in r}, key=lambda c: (("dataset", "regime", "method_dir", "fabric").index(c) if c in ("dataset", "regime", "method_dir", "fabric") else 9, c))
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        wr.writerows(flat)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/report")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    R = load_results(Path(a.results))
    missing = missing_cells(R)
    print("\n".join(missing) if missing else "every cell of EVALUATION 4.3 has a result file for both methods")
    if a.check:
        return 1 if missing else 0
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "regimes.md").write_text(regime_tables(R))
    (out / "ablations.md").write_text(ablation_report(R, out))
    write_csv(R, out / "all_results.csv")
    print(f"wrote {out}/regimes.md, ablations.md, all_results.csv and figures/ from {len(R)} result files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
