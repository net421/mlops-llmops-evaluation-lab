from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from mlops_eval.shared_evaluation import run_shared_evaluation

FILES = ["shared_release_report.json", "shared_checks.jsonl", "shared_evidence_manifest.json"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=Path("config/shared_evaluation_contract.json"))
    parser.add_argument("--semantic-root", type=Path, required=True)
    parser.add_argument("--decision-twin-root", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        first = root / "first"
        second = root / "second"
        run_shared_evaluation(args.contract, args.semantic_root, args.decision_twin_root, first)
        run_shared_evaluation(args.contract, args.semantic_root, args.decision_twin_root, second)
        mismatches = [name for name in FILES if (first / name).read_bytes() != (second / name).read_bytes()]
        if mismatches:
            raise SystemExit(f"Shared evidence is not reproducible: {mismatches}")
    print(f"Shared evidence reproducibility passed: {len(FILES)} files")


if __name__ == "__main__":
    main()
