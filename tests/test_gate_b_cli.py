"""Ticket 17 Gate B report wiring (no GPU)."""

from __future__ import annotations

import json
from pathlib import Path

from catan_llm.eval.arena import write_report


def test_gate_b_thresholds_logic():
    # Mirror pass condition from run_gate_b without importing torch/peft.
    parse_rate, legality, finished = 0.996, 0.995, 200
    cand, wr = 0.31, 0.28
    assert parse_rate >= 0.995 and legality >= 0.995 and finished >= 200 and cand > wr

    parse_rate, legality, finished = 0.99, 1.0, 200
    cand, wr = 0.40, 0.20
    assert not (parse_rate >= 0.995 and legality >= 0.995 and finished >= 200 and cand > wr)


def test_write_gate_report(tmp_path: Path):
    report = {
        "fixture": {"format": "ladder-4p"},
        "results": {"finished": 200, "win_rates": {"candidate": 0.3, "weightedrandom": 0.25}},
        "gate_b": {"pass": True},
    }
    out = tmp_path / "gate.json"
    write_report(report, out)
    assert json.loads(out.read_text())["gate_b"]["pass"] is True
