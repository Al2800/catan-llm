#!/usr/bin/env python3
"""Watch a HF Jobs SFT run and attempt a mid-run checkpoint Hub upload.

For jobs started **with** ``--ssh``, after step N is saved this script SSHes in
and uploads ``checkpoint-N``. For the current ticket-17 job (SSH disabled), it
still monitors progress and writes a status file explaining that live upload is
impossible — use the in-trainer Hub callback on the next launch instead.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

STEP_RE = re.compile(r"\|\s*(\d+)/(\d+)\s*\[")
LOSS_RE = re.compile(r"\{'loss':\s*'([^']+)'")
SAVE_RE = re.compile(r"checkpoint-(\d+)|Saving model checkpoint to .*checkpoint-(\d+)")


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run(cmd: list[str], timeout: int = 120) -> str:
    proc = subprocess.run(
        cmd, check=False, capture_output=True, text=True, timeout=timeout
    )
    return (proc.stdout or "") + (proc.stderr or "")


def _inspect(job_id: str) -> dict:
    out = _run(["hf", "jobs", "inspect", job_id, "--format", "json"])
    data = json.loads(out)
    return data[0] if isinstance(data, list) else data


def _logs_tail(job_id: str) -> str:
    # Full logs can be huge; pull and keep the tail in-process.
    out = _run(["hf", "jobs", "logs", job_id], timeout=180)
    return out[-200_000:]


def _parse_progress(logs: str) -> dict:
    step = total = None
    for m in STEP_RE.finditer(logs):
        step, total = int(m.group(1)), int(m.group(2))
    loss = None
    for m in LOSS_RE.finditer(logs):
        loss = m.group(1)
    saved = set()
    for m in SAVE_RE.finditer(logs):
        saved.add(int(m.group(1) or m.group(2)))
    # Trainer progress lines often show the step after save+eval.
    if step is not None and step >= 400:
        saved.add(400 if step >= 400 else step)
    return {"step": step, "total": total, "loss": loss, "saved_steps": sorted(saved)}


def _try_ssh_upload(job_id: str, step: int, repo: str, remote_ckpt: str) -> dict:
    dry = _run(["hf", "jobs", "ssh", job_id, "--dry-run"])
    if "SSH is not enabled" in dry or "Error:" in dry:
        return {
            "ok": False,
            "error": "SSH is not enabled on this job; cannot live-upload checkpoint",
            "dry_run": dry.strip()[:500],
        }
    # Upload via remote python/huggingface_hub if SSH works.
    remote_cmd = (
        "python - <<'PY'\n"
        "import os\n"
        "from pathlib import Path\n"
        "from huggingface_hub import HfApi\n"
        f"ckpt = Path({remote_ckpt!r})\n"
        f"repo = {repo!r}\n"
        "api = HfApi(token=os.environ.get('HF_TOKEN'))\n"
        "api.create_repo(repo_id=repo, repo_type='model', private=True, exist_ok=True)\n"
        f"prefix = 'checkpoints/checkpoint-{step}'\n"
        "api.upload_folder(folder_path=str(ckpt), path_in_repo=prefix, "
        "repo_id=repo, repo_type='model', "
        f"commit_message='mid-run checkpoint-{step} via watch script')\n"
        "print('uploaded', prefix)\n"
        "PY"
    )
    # `hf jobs ssh` is interactive; use dry-run guidance when non-interactive.
    return {
        "ok": False,
        "error": "SSH appears enabled but non-interactive upload is not wired; run manually",
        "suggested_remote_cmd": remote_cmd,
        "dry_run": dry.strip()[:500],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-id", default="AlCampbell/6a6c8e7d23ed89c748ec9ba3")
    ap.add_argument("--target-step", type=int, default=400)
    ap.add_argument("--repo", default="AlCampbell/catan-llm-sft-v1")
    ap.add_argument(
        "--remote-ckpt",
        default="/tmp/catan-llm/outputs/sft/qwen3.5-9b-qlora/checkpoint-400",
    )
    ap.add_argument(
        "--status-path",
        type=Path,
        default=Path("/opt/cursor/artifacts/sft_checkpoint_watch.json"),
    )
    ap.add_argument("--poll-secs", type=int, default=120)
    args = ap.parse_args()

    args.status_path.parent.mkdir(parents=True, exist_ok=True)
    uploaded = False
    status: dict = {
        "job_id": args.job_id,
        "target_step": args.target_step,
        "started_at": _utc(),
        "events": [],
    }

    def write() -> None:
        status["updated_at"] = _utc()
        args.status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        print(json.dumps({"updated_at": status["updated_at"], **{k: status.get(k) for k in ("stage", "progress", "upload")}}, indent=2), flush=True)

    print(f"watching {args.job_id} for checkpoint-{args.target_step}", flush=True)
    while True:
        try:
            info = _inspect(args.job_id)
            stage = (info.get("status") or {}).get("stage") or info.get("stage")
            running = (info.get("durations") or {}).get("running_secs")
            timeout = info.get("timeout") or 86400
            status["stage"] = stage
            status["running_secs"] = running
            status["timeout_secs"] = timeout
            status["ssh"] = info.get("ssh")

            logs = _logs_tail(args.job_id)
            progress = _parse_progress(logs)
            status["progress"] = progress

            step = progress.get("step") or 0
            if (
                not uploaded
                and step >= args.target_step
                and (
                    args.target_step in progress.get("saved_steps", [])
                    or step >= args.target_step
                )
            ):
                # Prefer waiting until eval after save has started (step>=target).
                result = _try_ssh_upload(
                    args.job_id, args.target_step, args.repo, args.remote_ckpt
                )
                status["upload"] = result
                status["events"].append({"at": _utc(), "event": "upload_attempt", **result})
                uploaded = True
                if not result.get("ok"):
                    status["events"].append(
                        {
                            "at": _utc(),
                            "event": "note",
                            "message": (
                                "Live SSH upload unavailable on this job. Keep the run "
                                "going; next launch uses in-trainer Hub checkpoint uploads."
                            ),
                        }
                    )

            write()
            if stage and stage not in {"RUNNING", "UPDATING", "STARTING", "SCHEDULED", "SCHEDULING"}:
                status["events"].append({"at": _utc(), "event": "terminal", "stage": stage})
                write()
                return 0
        except Exception as exc:  # noqa: BLE001
            status["last_error"] = f"{type(exc).__name__}: {exc}"
            status["events"].append({"at": _utc(), "event": "poll_error", "error": status["last_error"]})
            write()
        time.sleep(max(30, args.poll_secs))


if __name__ == "__main__":
    raise SystemExit(main())
