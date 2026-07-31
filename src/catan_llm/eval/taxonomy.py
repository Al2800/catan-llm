"""Failure taxonomy builder (ticket 18) — analyze Gate B / arena JSON reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Ordered hints: error code → proposed fix class.
FIX_HINTS: dict[str, str] = {
    "json_parse_failed": (
        "decoding: tighten structured JSON / lower temperature; more SFT on assistant JSON"
    ),
    "missing_action": "decoding: enforce schema key `action`; add constrained decoding",
    "action_not_int": "decoding: coerce/validate action as int in schema",
    "action_out_of_range": "data+decoding: oversample long action lists; guided_json index range",
    "request_failed": "serving: timeouts / OOM / connectivity — check serve logs",
    "unknown": "inspect raw samples; extend parser error codes",
}


def load_arena_report(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _results_block(report: dict[str, Any]) -> dict[str, Any]:
    if "results" in report and isinstance(report["results"], dict):
        return report["results"]
    return report


def build_taxonomy(report: dict[str, Any]) -> dict[str, Any]:
    """Derive top failure modes + proposed fixes from an arena/Gate B report."""
    results = _results_block(report)
    gate = report.get("gate_b") or {}
    err = results.get("action_error_hist") or {}
    phase = results.get("phase_error_hist") or {}
    total_err = sum(int(v) for v in err.values()) or 0
    ranked = sorted(err.items(), key=lambda kv: (-int(kv[1]), kv[0]))
    modes = []
    for code, count in ranked:
        count_i = int(count)
        modes.append(
            {
                "error": code,
                "count": count_i,
                "share_of_errors": (count_i / total_err) if total_err else 0.0,
                "proposed_fix": FIX_HINTS.get(code, FIX_HINTS["unknown"]),
            }
        )
    phase_ranked = sorted(phase.items(), key=lambda kv: (-int(kv[1]), kv[0]))[:20]
    return {
        "ticket": "18",
        "fixture": (report.get("fixture") or {}).get("format"),
        "games": results.get("games"),
        "finished": results.get("finished"),
        "parse_rate_model": results.get("parse_rate_model"),
        "legality_rate_model": results.get("legality_rate_model"),
        "fallback_rate": results.get("fallback_rate"),
        "gate_b_pass": gate.get("pass"),
        "error_total": total_err,
        "top_failure_modes": modes,
        "phase_error_top": [
            {"bucket": k, "count": int(v)} for k, v in phase_ranked
        ],
        "stage1_notes": [
            "Gate B requires parse/legality ≥ 0.995 and candidate WR > weightedrandom.",
            "Taxonomy drives the next data/decoding iteration — not a skill claim by itself.",
        ],
    }


def taxonomy_to_markdown(tax: dict[str, Any]) -> str:
    lines = [
        "# Failure taxonomy v1",
        "",
        f"- fixture: `{tax.get('fixture')}`",
        f"- games / finished: {tax.get('games')} / {tax.get('finished')}",
        f"- parse_rate_model: {tax.get('parse_rate_model')}",
        f"- legality_rate_model: {tax.get('legality_rate_model')}",
        f"- fallback_rate: {tax.get('fallback_rate')}",
        f"- gate_b_pass: {tax.get('gate_b_pass')}",
        f"- error_total: {tax.get('error_total')}",
        "",
        "## Top failure modes",
        "",
        "| error | count | share | proposed fix |",
        "|---|---:|---:|---|",
    ]
    for mode in tax.get("top_failure_modes") or []:
        lines.append(
            f"| `{mode['error']}` | {mode['count']} | {mode['share_of_errors']:.3f} | "
            f"{mode['proposed_fix']} |"
        )
    if not tax.get("top_failure_modes"):
        lines.append("| _(none)_ | 0 | 0 | — |")
    lines.extend(
        [
            "",
            "## Phase × error (top)",
            "",
            "| bucket | count |",
            "|---|---:|",
        ]
    )
    for row in tax.get("phase_error_top") or []:
        lines.append(f"| `{row['bucket']}` | {row['count']} |")
    if not tax.get("phase_error_top"):
        lines.append("| _(none)_ | 0 |")
    lines.extend(["", "## Notes", ""])
    for note in tax.get("stage1_notes") or []:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def write_taxonomy(
    report_path: Path,
    *,
    out_json: Path,
    out_md: Path | None = None,
) -> dict[str, Any]:
    report = load_arena_report(report_path)
    tax = build_taxonomy(report)
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(tax, indent=2), encoding="utf-8")
    if out_md is not None:
        Path(out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(out_md).write_text(taxonomy_to_markdown(tax), encoding="utf-8")
    return tax
