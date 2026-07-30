"""Stable identity helpers for schema v2 (game_key, hashes, commits)."""

from __future__ import annotations

import hashlib
import json
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

# Pinned in pyproject.toml; keep in sync when bumping Catanatron.
CATANATRON_COMMIT = "82aae93ab1f7c267218be0566df573ce477ec3d8"

# Bump whenever canonical prompt text changes (DATA_CONTRACT §2 / §4).
PROMPT_VERSION = "2026-07-30.1"
SCHEMA_VERSION = "v2"
KNOWN_PROMPT_VERSIONS = frozenset({PROMPT_VERSION})


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def bot_config_hash(bot_config: list[dict[str, Any]]) -> str:
    return sha256_hex(canonical_json(bot_config))


def make_game_key(seed: int, map_hash: str, bot_cfg_hash: str) -> str:
    return sha256_hex(f"{seed}:{map_hash}:{bot_cfg_hash}")


def hash_catan_map(catan_map) -> str:
    """Deterministic hash of land tiles / numbers / ports after construction."""
    tiles = []
    for coord, tile in sorted(catan_map.land_tiles.items(), key=lambda item: item[1].id):
        tiles.append(
            {
                "id": tile.id,
                "coord": list(coord),
                "resource": None if tile.resource is None else str(tile.resource),
                "number": tile.number,
                "nodes": sorted(tile.nodes.values()),
            }
        )
    ports = []
    for resource, node_ids in sorted(
        catan_map.port_nodes.items(),
        key=lambda item: (str(item[0]), sorted(item[1])),
    ):
        ports.append(
            {
                "resource": None if resource is None else str(resource),
                "nodes": sorted(node_ids),
            }
        )
    return sha256_hex(canonical_json({"tiles": tiles, "ports": ports}))


def compact_board_dict(game) -> dict[str, Any]:
    """Static board layout for trajectory records (renderer reconstruction)."""
    board = game.state.board
    catan_map = board.map
    tiles = []
    for coord, tile in sorted(catan_map.land_tiles.items(), key=lambda item: item[1].id):
        tiles.append(
            {
                "id": tile.id,
                "coord": list(coord),
                "resource": None if tile.resource is None else str(tile.resource),
                "number": tile.number,
                "nodes": sorted(tile.nodes.values()),
            }
        )
    ports = []
    for resource, node_ids in catan_map.port_nodes.items():
        ports.append(
            {
                "resource": None if resource is None else str(resource),
                "nodes": sorted(node_ids),
            }
        )
    robber = board.robber_coordinate
    return {
        "tiles": tiles,
        "ports": ports,
        "robber_start": list(robber) if isinstance(robber, tuple) else robber,
    }


@lru_cache(maxsize=1)
def resolve_source_commit() -> str:
    """Best-effort git SHA of this repo; 'unknown' if unavailable."""
    try:
        root = Path(__file__).resolve().parents[3]
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"
