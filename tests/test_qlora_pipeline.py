"""Ticket 15 — production QLoRA pipeline wiring (CPU-safe)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from catan_llm.training.masking import qwen_max_seq_length, qwen_model_name, qwen_revision
from catan_llm.training.qlora import (
    resolve_data_path,
    run_qlora_sft,
    verify_assistant_only_mask,
)


def test_qwen_config_pins_revision_and_seq_floor():
    assert qwen_model_name() == "Qwen/Qwen3.5-9B"
    assert qwen_revision() == "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    assert qwen_max_seq_length() >= 4096


def test_config_points_at_phase1_expert_v1():
    cfg = yaml.safe_load(Path("configs/qwen3.5-9b-qlora.yaml").read_text(encoding="utf-8"))
    assert "phase1" in cfg["data"]["train_file"]
    assert cfg["data"]["train_file"].endswith("train.jsonl")
    assert cfg["train"]["assistant_only_loss"] is True
    assert cfg["train"]["gradient_checkpointing"] is True


def test_resolve_data_path_phase1_alias(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path
    target = root / "data" / "phase1" / "processed" / "expert-v1" / "train.jsonl"
    target.parent.mkdir(parents=True)
    target.write_text('{"messages":[]}\n', encoding="utf-8")
    monkeypatch.chdir(root)
    # Old alias path in early config drafts.
    got = resolve_data_path("data/processed/expert-v1/train.jsonl", repo_root=root)
    assert got == target.resolve()


def test_verify_assistant_only_mask_with_smol():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        "HuggingFaceTB/SmolLM2-135M-Instruct", trust_remote_code=True
    )
    result = verify_assistant_only_mask(tok, max_seq_length=4096)
    assert result["ok"] is True
    assert result["assistant_token_count"] > 0
    assert result["prompt_masked"] is True


def test_dry_run_writes_report(tmp_path: Path):
    train = tmp_path / "train.jsonl"
    rows = []
    for i in range(4):
        rows.append(
            {
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": f"user {i}"},
                    {
                        "role": "assistant",
                        "content": '{"action": 0, "reasoning": "roll"}',
                    },
                ],
                "prompt_version": "2026-07-30.1",
            }
        )
    train.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    out = tmp_path / "out"
    report = run_qlora_sft(
        Path("configs/qwen3.5-9b-qlora.yaml"),
        train_file=train,
        output_dir=out,
        dry_run=True,
        max_samples=4,
        repo_root=Path.cwd(),
    )
    assert report.dry_run is True
    assert report.assistant_only_loss is True
    assert report.max_seq_length >= 4096
    assert report.model == "Qwen/Qwen3.5-9B"
    assert (out / "dry_run_report.json").is_file()
    payload = json.loads((out / "dry_run_report.json").read_text(encoding="utf-8"))
    assert payload["ticket"] == "15"
    assert payload["mask_check"]["ok"] is True


def test_refuses_low_max_seq_length(tmp_path: Path):
    cfg_path = tmp_path / "bad.yaml"
    base = yaml.safe_load(Path("configs/qwen3.5-9b-qlora.yaml").read_text(encoding="utf-8"))
    base["data"]["max_seq_length"] = 2048
    train = tmp_path / "train.jsonl"
    train.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "system", "content": "s"},
                    {"role": "user", "content": "u"},
                    {"role": "assistant", "content": '{"action": 0, "reasoning": "x"}'},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    cfg_path.write_text(yaml.safe_dump(base), encoding="utf-8")
    with pytest.raises(ValueError, match="4096"):
        run_qlora_sft(
            cfg_path,
            train_file=train,
            output_dir=tmp_path / "out",
            dry_run=True,
            repo_root=tmp_path,
        )
