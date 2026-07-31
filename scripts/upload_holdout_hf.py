#!/usr/bin/env python3
"""Upload Mac Phase-1 eval holdout to HF (ops follow-up from ticket 14).

Run on the Mac that has the immutable holdout artifact:

  python scripts/upload_holdout_hf.py \\
    --dir data/phase1/processed/eval-holdout-v1 \\
    --expect-sha256 5809a9fc1ed42b2fda29abdd77e37980f895ab52e7c5787023fce2fc7ccd1bfc

Uploads to AlCampbell/catan-llm-phase1 under processed/eval-holdout-v1/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPO = "AlCampbell/catan-llm-phase1"
DEFAULT_SHA = "5809a9fc1ed42b2fda29abdd77e37980f895ab52e7c5787023fce2fc7ccd1bfc"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dir",
        type=Path,
        default=Path("data/phase1/processed/eval-holdout-v1"),
    )
    ap.add_argument("--expect-sha256", default=DEFAULT_SHA)
    ap.add_argument("--repo", default=REPO)
    args = ap.parse_args()

    holdout = args.dir / "holdout.jsonl"
    manifest = args.dir / "manifest.json"
    if not holdout.is_file():
        print(f"missing {holdout}", file=sys.stderr)
        return 1
    got = sha256_file(holdout)
    print(f"holdout.jsonl sha256={got}")
    if args.expect_sha256 and got != args.expect_sha256:
        print(
            f"ERROR: checksum mismatch (expected {args.expect_sha256})",
            file=sys.stderr,
        )
        return 2
    if manifest.is_file():
        meta = json.loads(manifest.read_text(encoding="utf-8"))
        if not meta.get("immutable"):
            print("WARNING: manifest.immutable is not true", file=sys.stderr)

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN required", file=sys.stderr)
        return 1

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    for path in sorted(args.dir.iterdir()):
        if path.is_file():
            dest = f"processed/eval-holdout-v1/{path.name}"
            print(f"upload {path} → {args.repo}:{dest}")
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=dest,
                repo_id=args.repo,
                repo_type="dataset",
                commit_message=f"upload immutable eval holdout ({path.name})",
            )
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
