"""Ticket 18 taxonomy builder (CPU)."""

from __future__ import annotations

import json
from pathlib import Path

from catan_llm.eval.taxonomy import build_taxonomy, taxonomy_to_markdown, write_taxonomy


def test_build_taxonomy_ranks_errors(tmp_path: Path):
    report = {
        "fixture": {"format": "ladder-4p"},
        "results": {
            "games": 2,
            "finished": 2,
            "parse_rate_model": 0.5,
            "legality_rate_model": 0.5,
            "fallback_rate": 0.5,
            "action_error_hist": {"json_parse_failed": 3, "action_out_of_range": 1},
            "phase_error_hist": {"BUILD_ROAD:json_parse_failed": 2},
        },
        "gate_b": {"pass": False},
    }
    tax = build_taxonomy(report)
    assert tax["top_failure_modes"][0]["error"] == "json_parse_failed"
    assert tax["top_failure_modes"][0]["count"] == 3
    assert "decoding" in tax["top_failure_modes"][0]["proposed_fix"]
    md = taxonomy_to_markdown(tax)
    assert "json_parse_failed" in md

    src = tmp_path / "gate.json"
    src.write_text(json.dumps(report), encoding="utf-8")
    out_j = tmp_path / "tax.json"
    out_m = tmp_path / "tax.md"
    write_taxonomy(src, out_json=out_j, out_md=out_m)
    assert out_j.is_file() and out_m.is_file()
