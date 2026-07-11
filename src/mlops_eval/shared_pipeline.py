"""CLI for the shared Semantic Agent and Decision Twin release gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .shared_evaluation import run_shared_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=Path("config/shared_evaluation_contract.json"))
    parser.add_argument("--semantic-root", type=Path, required=True)
    parser.add_argument("--decision-twin-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/shared"))
    args = parser.parse_args()
    report = run_shared_evaluation(
        args.contract.resolve(),
        args.semantic_root.resolve(),
        args.decision_twin_root.resolve(),
        args.output_dir.resolve(),
    )
    print(
        json.dumps(
            {
                "status": report["overall_release_decision"],
                "systems": {
                    name: value["release_decision"] for name, value in report["systems"].items()
                },
            },
            sort_keys=True,
        )
    )
    if report["overall_release_decision"] != "pass":
        sys.exit(1)


if __name__ == "__main__":
    main()
