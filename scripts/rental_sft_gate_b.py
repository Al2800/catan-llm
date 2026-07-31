#!/usr/bin/env python3
"""Ticket 17 rental driver: QLoRA SFT on expert-v1 + optional Gate B.

Usage on HF Jobs L40S (see scripts/rental_sft_gate_b_job.sh):
  python scripts/rental_sft_gate_b.py
  python scripts/rental_sft_gate_b.py --max-steps 200 --gate-games 4   # smoke
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "outputs" / "sft" / "qwen3.5-9b-qlora"
DATA_DIR = REPO_ROOT / "data" / "phase1" / "processed" / "expert-v1"
HF_DATASET = "AlCampbell/catan-llm-phase1"


def _download_expert_v1() -> None:
    from huggingface_hub import hf_hub_download

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("train.jsonl", "val.jsonl", "test.jsonl", "manifest.json", "quality.json"):
        dest = DATA_DIR / name
        if dest.is_file() and dest.stat().st_size > 0:
            print(f"keep local {dest}", flush=True)
            continue
        print(f"download {name} from {HF_DATASET}", flush=True)
        local = hf_hub_download(
            repo_id=HF_DATASET,
            repo_type="dataset",
            filename=f"processed/expert-v1/{name}",
            token=os.environ.get("HF_TOKEN"),
        )
        shutil.copy2(local, dest)


def _download_hub_checkpoint(repo_id: str, checkpoint: str, dest: Path) -> Path:
    from catan_llm.training.hub_checkpoints import download_hub_checkpoint

    print(f"download resume checkpoint {checkpoint} from {repo_id}", flush=True)
    path = download_hub_checkpoint(repo_id, checkpoint, dest)
    print(f"resume checkpoint ready at {path}", flush=True)
    return path


def _upload_outputs(report: dict) -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        return
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        repo = "AlCampbell/catan-llm-sft-v1"
        api.create_repo(repo_id=repo, repo_type="model", private=True, exist_ok=True)
        adapter = OUT_DIR / "adapter"
        if adapter.is_dir():
            api.upload_folder(
                folder_path=str(adapter),
                path_in_repo="adapter",
                repo_id=repo,
                repo_type="model",
                commit_message="ticket 17: QLoRA adapter",
            )
        report_path = OUT_DIR / "ticket17_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        for name in (
            "ticket17_report.json",
            "train_report.json",
            "train_history.json",
            "train_history.md",
            "resolved_config.yaml",
        ):
            path = OUT_DIR / name
            if path.is_file():
                api.upload_file(
                    path_or_fileobj=str(path),
                    path_in_repo=name,
                    repo_id=repo,
                    repo_type="model",
                    commit_message=f"ticket 17: {name}",
                )
        gate_path = REPO_ROOT / "outputs" / "arena" / "gate_b_ladder4p.json"
        if gate_path.is_file():
            api.upload_file(
                path_or_fileobj=str(gate_path),
                path_in_repo="gate_b_ladder4p.json",
                repo_id=repo,
                repo_type="model",
                commit_message="ticket 17: Gate B report",
            )
        print(f"uploaded artifacts → hf://{repo}", flush=True)
    except Exception as exc:
        print(f"upload skipped/failed: {exc}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-gate", action="store_true")
    ap.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip mid-train val eval (faster chunks; Hub checkpoints still upload)",
    )
    ap.add_argument("--gate-games", type=int, default=200)
    ap.add_argument(
        "--adapter",
        type=Path,
        default=None,
        help="Existing adapter (skip train)",
    )
    ap.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Local Trainer checkpoint dir (e.g. outputs/.../checkpoint-400)",
    )
    ap.add_argument(
        "--resume-from-hub",
        type=str,
        default=None,
        help="Hub checkpoint name or repo:name (default repo AlCampbell/catan-llm-sft-v1)",
    )
    ap.add_argument(
        "--hub-checkpoint-repo",
        type=str,
        default=None,
        help="Override mid-run checkpoint upload repo (empty string disables)",
    )
    args = ap.parse_args()

    sys.path.insert(0, str(REPO_ROOT / "src"))
    report: dict = {
        "ticket": "17",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "train": None,
        "gate_b": None,
        "error": None,
    }

    try:
        import torch

        report["hardware"] = {
            "cuda": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required")

        _download_expert_v1()

        if args.hub_checkpoint_repo is not None:
            os.environ["CATAN_HUB_CHECKPOINT_REPO"] = args.hub_checkpoint_repo

        resume_from = args.resume_from
        if args.resume_from_hub:
            raw = args.resume_from_hub
            if ":" in raw and not raw.startswith("checkpoint-"):
                hub_repo, ckpt_name = raw.split(":", 1)
            else:
                hub_repo, ckpt_name = "AlCampbell/catan-llm-sft-v1", raw
            resume_from = _download_hub_checkpoint(hub_repo, ckpt_name, OUT_DIR)

        adapter = args.adapter
        if not args.skip_train and adapter is None:
            from catan_llm.training.qlora import run_qlora_sft

            train_report = run_qlora_sft(
                REPO_ROOT / "configs" / "qwen3.5-9b-qlora.yaml",
                train_file=DATA_DIR / "train.jsonl",
                val_file=None if args.skip_eval else (DATA_DIR / "val.jsonl"),
                output_dir=OUT_DIR,
                max_steps=args.max_steps,
                max_samples=args.max_samples,
                resume_from=resume_from,
                repo_root=REPO_ROOT,
            )
            report["train"] = train_report.as_dict()
            adapter = Path(train_report.checkpoint or (OUT_DIR / "adapter"))
        elif adapter is None:
            adapter = OUT_DIR / "adapter"

        if not args.skip_gate:
            # Prefer installed console script when present.
            cmd = [
                sys.executable,
                "-m",
                "catan_llm.scripts.run_gate_b",
                "--adapter",
                str(adapter),
                "--games",
                str(args.gate_games),
                "--out",
                str(REPO_ROOT / "outputs" / "arena" / "gate_b_ladder4p.json"),
                "--config",
                str(REPO_ROOT / "configs" / "qwen3.5-9b-qlora.yaml"),
            ]
            print("running", " ".join(cmd), flush=True)
            proc = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
            gate_path = REPO_ROOT / "outputs" / "arena" / "gate_b_ladder4p.json"
            if gate_path.is_file():
                report["gate_b"] = json.loads(gate_path.read_text(encoding="utf-8"))
            report["gate_exit_code"] = proc.returncode

        report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "ticket17_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        _upload_outputs(report)
        print(json.dumps(report.get("train") or {"gate_only": True}, indent=2))
        return int(report.get("gate_exit_code") or 0)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "ticket17_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(report["error"], file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
