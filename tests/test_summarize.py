import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import summarize  # noqa: E402


def put(results, ds, regime, mdir, name, method=None, **kw):
    res = {"dataset": ds, "regime": regime, "method": method or mdir.split("_")[0], "fabric": name,
           "tile_auroc": 0.8, "image_auroc": 0.8, "aupro_30": 0.5, "tile_recall": 0.5, "false_alarms_per_1000": 10.0, **kw}
    p = Path(results) / ds / regime / mdir / f"{name}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res))


def all_cells(results, skip=None):
    for ds, regime, names in summarize.CELLS:
        for m in summarize.METHODS:
            for n in names if isinstance(names, list) else [f"fabric0{i}" for i in range(names)]:
                if (ds, regime, m, n) != skip:
                    put(results, ds, regime, summarize.headline(ds, regime, m), n, m)


def test_check_reports_missing(tmp_path, capsys):
    all_cells(tmp_path, skip=("zju_leaper", "held_out", "rd4ad", "pattern12_held_out"))
    (tmp_path / "notes.json").write_text("[1]")  # not a result file: ignored
    assert summarize.main(["--results", str(tmp_path), "--check"]) == 1
    out = capsys.readouterr().out
    assert out.strip() == "zju_leaper/held_out/rd4ad: missing pattern12_held_out"
    put(tmp_path, "zju_leaper", "held_out", "rd4ad", "pattern12_held_out", "rd4ad")
    assert summarize.main(["--results", str(tmp_path), "--check"]) == 0
    # RAW-FABRID's RD4AD unified headline is seed 0
    (tmp_path / "raw_fabrid" / "unified" / "rd4ad_s0" / "unified.json").rename(tmp_path / "raw_fabrid" / "unified" / "rd4ad_s0" / "x.json")
    assert "raw_fabrid/unified/rd4ad_s0: missing unified" in summarize.missing_cells(summarize.load_results(tmp_path))
    # AITEX is counted, not named
    for f in (tmp_path / "aitex" / "per_fabric" / "patchcore").glob("fabric0[0-2].json"):
        f.unlink()
    assert "aitex/per_fabric/patchcore: missing 4 of 7 files" in summarize.missing_cells(summarize.load_results(tmp_path))


def test_ablation_tables_and_plots(tmp_path, capsys):
    res, out = tmp_path / "results", tmp_path / "report"
    # overlap ablation: variants compared on the rolls they share (Rollo1A, Rollo2A), headline also has Rollo3A
    for roll, v in (("Rollo1A", 0.9), ("Rollo2A", 0.7), ("Rollo3A", 0.1)):
        put(res, "raw_fabrid", "per_fabric", "patchcore", roll, tile_auroc=v, aupro_30=v / 2)
    for roll, v in (("Rollo1A", 0.8), ("Rollo2A", 0.6)):
        put(res, "raw_fabrid", "per_fabric", "patchcore_ov32", roll, "patchcore", tile_auroc=v, aupro_30=v / 2)
    # a shared unified bundle with a per-fabric breakdown, to appear in the cross-regime table
    put(res, "raw_fabrid", "unified", "rd4ad_s0", "unified", "rd4ad", by_fabric={"Rollo1A": {"tile_auroc": 0.55}})
    put(res, "raw_fabrid", "per_fabric", "rd4ad", "Rollo1A", tile_auroc=0.95)
    put(res, "raw_fabrid", "held_out", "rd4ad", "Rollo1A_held_out", tile_auroc=0.33)
    assert summarize.main(["--results", str(res), "--out", str(out)]) == 0
    order, data = summarize.ablation_data(summarize.load_results(res), "overlap")
    assert order == ["0 px (default)", "32 px"]
    assert data["0 px (default)"]["patchcore"] == {"n": 2, "tile_auroc": pytest.approx(0.8), "aupro_30": pytest.approx(0.4)}
    assert data["32 px"]["patchcore"]["tile_auroc"] == pytest.approx(0.7)
    md = (out / "ablations.md").read_text()
    assert "## overlap (raw_fabrid, per_fabric)" in md and "| 32 px | 2 | 0.700 | 0.350 |" in md
    for name in summarize.ABLATIONS:  # every ablation gets a table and two charts, even with nothing measured yet
        assert f"## {name} " in md
        for metric in ("tile_auroc", "image_auroc", "aupro_30"):
            f = out / "figures" / f"ablation_{name}_{metric}.svg"
            if f.exists():
                t = f.read_text()
                assert t.startswith('<svg') and t.endswith('</svg>')
    assert "<rect" in (out / "figures" / "ablation_overlap_tile_auroc.svg").read_text()
    reg = (out / "regimes.md").read_text()
    assert "## raw_fabrid, per_fabric" in reg and "| Rollo1A | 0.900 |" in reg and "**mean**" in reg
    row = next(ln for ln in reg.split("\n") if ln.startswith("| Rollo1A |") and "0.550" in ln)
    assert "0.950" in row and "0.330" in row  # per-fabric, unified (from by_fabric) and held-out values of one fabric
    assert (out / "all_results.csv").read_text().startswith("dataset,regime,method_dir,fabric")
