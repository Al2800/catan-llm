"""Upload Trainer checkpoints to the Hub mid-run (resume insurance).

HF Jobs disks are ephemeral and SSH is optional. Pushing each ``checkpoint-*``
folder to a model repo lets a later job resume without relying on the first
machine staying alive.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable


DEFAULT_HUB_CHECKPOINT_REPO = "AlCampbell/catan-llm-sft-v1"


def resolve_hub_checkpoint_repo(
    config_value: str | None = None,
    *,
    env: dict[str, str] | None = None,
) -> str | None:
    """Return Hub model repo for mid-run checkpoints, or None to disable.

    Precedence: ``CATAN_HUB_CHECKPOINT_REPO`` env (empty disables) → config →
    default repo when ``HF_TOKEN`` is set.
    """
    environ = env if env is not None else os.environ
    if "CATAN_HUB_CHECKPOINT_REPO" in environ:
        raw = (environ.get("CATAN_HUB_CHECKPOINT_REPO") or "").strip()
        return raw or None
    if config_value is not None:
        raw = str(config_value).strip()
        return raw or None
    if environ.get("HF_TOKEN"):
        return DEFAULT_HUB_CHECKPOINT_REPO
    return None


def upload_checkpoint_folder(
    checkpoint_dir: Path | str,
    *,
    repo_id: str,
    token: str | None = None,
    path_in_repo: str | None = None,
    api: Any | None = None,
) -> str:
    """Upload one checkpoint directory; return the Hub path prefix used."""
    ckpt = Path(checkpoint_dir)
    if not ckpt.is_dir():
        raise FileNotFoundError(f"checkpoint dir missing: {ckpt}")
    name = ckpt.name
    prefix = path_in_repo or f"checkpoints/{name}"
    tok = token if token is not None else os.environ.get("HF_TOKEN")
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi(token=tok)
    api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
    api.upload_folder(
        folder_path=str(ckpt),
        path_in_repo=prefix,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"mid-run checkpoint: {name}",
    )
    return prefix


def download_hub_checkpoint(
    repo_id: str,
    checkpoint_name: str,
    dest_dir: Path | str,
    *,
    token: str | None = None,
    path_in_repo: str | None = None,
) -> Path:
    """Download ``checkpoints/<name>`` (or custom prefix) into ``dest_dir/<name>``."""
    from huggingface_hub import snapshot_download

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    name = checkpoint_name if checkpoint_name.startswith("checkpoint-") else (
        f"checkpoint-{checkpoint_name}"
    )
    prefix = path_in_repo or f"checkpoints/{name}"
    local = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        allow_patterns=[f"{prefix}/*", f"{prefix}/**"],
        token=token if token is not None else os.environ.get("HF_TOKEN"),
    )
    src = Path(local) / prefix
    if not src.is_dir():
        raise FileNotFoundError(f"Hub checkpoint not found at {prefix} in {repo_id}")
    out = dest / name
    if out.exists():
        return out
    # snapshot_download already materializes files; symlink/copy into trainer path.
    import shutil

    shutil.copytree(src, out)
    return out


class HubCheckpointUploadCallback:
    """transformers ``TrainerCallback`` that uploads each saved checkpoint.

    Constructed without importing transformers at module import time so CPU unit
    tests stay light. Call :meth:`as_trainer_callback` for a real callback
    instance, or invoke :meth:`on_save` directly in tests.
    """

    def __init__(
        self,
        repo_id: str,
        *,
        token: str | None = None,
        uploader: Callable[..., str] | None = None,
        enabled: bool = True,
    ) -> None:
        self.repo_id = repo_id
        self.token = token
        self.uploader = uploader or upload_checkpoint_folder
        self.enabled = enabled
        self.uploaded: list[dict[str, Any]] = []

    def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        if not self.enabled:
            return control
        out_dir = Path(getattr(args, "output_dir", "") or ".")
        step = int(getattr(state, "global_step", 0) or 0)
        ckpt = out_dir / f"checkpoint-{step}"
        if not ckpt.is_dir():
            # Trainer may still be flushing; brief retry.
            time.sleep(1.0)
        if not ckpt.is_dir():
            print(f"hub-checkpoint: skip missing {ckpt}", flush=True)
            return control
        try:
            prefix = self.uploader(
                ckpt, repo_id=self.repo_id, token=self.token
            )
            record = {
                "step": step,
                "local": str(ckpt),
                "repo_id": self.repo_id,
                "path_in_repo": prefix,
                "ok": True,
            }
            self.uploaded.append(record)
            print(
                f"hub-checkpoint: uploaded {ckpt.name} → hf://{self.repo_id}/{prefix}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 — never kill training for Hub I/O
            record = {
                "step": step,
                "local": str(ckpt),
                "repo_id": self.repo_id,
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
            self.uploaded.append(record)
            print(f"hub-checkpoint: upload failed (continuing): {record['error']}", flush=True)
        return control

    def as_trainer_callback(self) -> Any:
        from transformers import TrainerCallback

        outer = self

        class _CB(TrainerCallback):
            def on_save(self, args, state, control, **kwargs):  # type: ignore[no-untyped-def]
                return outer.on_save(args, state, control, **kwargs)

        return _CB()
