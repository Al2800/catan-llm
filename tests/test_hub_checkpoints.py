"""Mid-run Hub checkpoint upload helpers (CPU-safe)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from catan_llm.training.hub_checkpoints import (
    DEFAULT_HUB_CHECKPOINT_REPO,
    HubCheckpointUploadCallback,
    resolve_hub_checkpoint_repo,
    upload_checkpoint_folder,
)


def test_resolve_hub_repo_env_override_and_disable():
    assert (
        resolve_hub_checkpoint_repo(
            "cfg/repo", env={"CATAN_HUB_CHECKPOINT_REPO": "env/repo", "HF_TOKEN": "x"}
        )
        == "env/repo"
    )
    assert (
        resolve_hub_checkpoint_repo(
            "cfg/repo", env={"CATAN_HUB_CHECKPOINT_REPO": "", "HF_TOKEN": "x"}
        )
        is None
    )
    assert resolve_hub_checkpoint_repo("cfg/repo", env={}) == "cfg/repo"
    assert (
        resolve_hub_checkpoint_repo(None, env={"HF_TOKEN": "x"})
        == DEFAULT_HUB_CHECKPOINT_REPO
    )
    assert resolve_hub_checkpoint_repo(None, env={}) is None


def test_upload_checkpoint_folder_calls_api(tmp_path: Path):
    ckpt = tmp_path / "checkpoint-400"
    ckpt.mkdir()
    (ckpt / "adapter_model.safetensors").write_bytes(b"x")

    calls: list[dict] = []

    class FakeApi:
        def create_repo(self, **kwargs):
            calls.append({"op": "create", **kwargs})

        def upload_folder(self, **kwargs):
            calls.append({"op": "upload", **kwargs})

    prefix = upload_checkpoint_folder(
        ckpt, repo_id="AlCampbell/catan-llm-sft-v1", api=FakeApi()
    )
    assert prefix == "checkpoints/checkpoint-400"
    assert calls[0]["op"] == "create"
    assert calls[1]["path_in_repo"] == "checkpoints/checkpoint-400"


def test_callback_on_save_records_success(tmp_path: Path):
    ckpt = tmp_path / "checkpoint-400"
    ckpt.mkdir()
    (ckpt / "x.bin").write_text("ok", encoding="utf-8")

    def fake_upload(path, *, repo_id, token=None):
        assert Path(path) == ckpt
        assert repo_id == "r/test"
        return "checkpoints/checkpoint-400"

    cb = HubCheckpointUploadCallback("r/test", uploader=fake_upload)
    args = SimpleNamespace(output_dir=str(tmp_path))
    state = SimpleNamespace(global_step=400)
    control = SimpleNamespace()
    cb.on_save(args, state, control)
    assert cb.uploaded == [
        {
            "step": 400,
            "local": str(ckpt),
            "repo_id": "r/test",
            "path_in_repo": "checkpoints/checkpoint-400",
            "ok": True,
        }
    ]


def test_callback_on_save_swallows_errors(tmp_path: Path):
    ckpt = tmp_path / "checkpoint-200"
    ckpt.mkdir()

    def boom(*_a, **_k):
        raise RuntimeError("hub down")

    cb = HubCheckpointUploadCallback("r/test", uploader=boom)
    cb.on_save(
        SimpleNamespace(output_dir=str(tmp_path)),
        SimpleNamespace(global_step=200),
        SimpleNamespace(),
    )
    assert cb.uploaded[0]["ok"] is False
    assert "RuntimeError" in cb.uploaded[0]["error"]
