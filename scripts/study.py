"""Run a study: every config of a list file, fit then evaluate. Resumable: configs whose result files all exist are skipped.

    python scripts/study.py configs/studies/02_raw_fabrid.txt [--force] [--dry-run]

The list has one config path per line ('#' starts a comment). A config that fails does not stop the study; the table at
the end says which failed and why, and the exit code is 1.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))  # fit.py and evaluate.py sit next to this file
import evaluate  # noqa: E402
import fit  # noqa: E402

from loomguard.runs import build_rows, load_config, profiles, result_path  # noqa: E402


def read_list(path: Path | str) -> list[str]:
    lines = (ln.split("#")[0].strip() for ln in Path(path).read_text().splitlines())
    return [ln for ln in lines if ln]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("list")
    ap.add_argument("--force", action="store_true", help="rerun configs whose results already exist")
    ap.add_argument("--dry-run", action="store_true", help="print the configs and their result files only")
    a = ap.parse_args(argv)
    status = []
    for path in read_list(a.list):
        try:
            cfg = load_config(path)
            expected = [result_path(cfg, n) for n in profiles(build_rows(cfg), cfg)]
            if a.dry_run:
                print(f"{path}: {len(expected)} results, e.g. {expected[0]}")
                continue
            if not a.force and all(p.is_file() for p in expected):
                status.append((path, "skipped", "results exist"))
                continue
            fit.main(["--config", path])
            evaluate.main(["--config", path])
            status.append((path, "ok", f"{len(expected)} results"))
        except (Exception, SystemExit) as e:  # one broken config must not lose the rest of a day-long study
            status.append((path, "FAILED", f"{type(e).__name__}: {e}"))
            print(f"FAILED {path}: {e}", file=sys.stderr)
    for path, st, note in status:
        print(f"{st:8} {path}  {note}")
    return 1 if any(st == "FAILED" for _, st, _ in status) else 0


if __name__ == "__main__":
    sys.exit(main())
