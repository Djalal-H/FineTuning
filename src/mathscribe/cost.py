"""GPU hours and $ cost of a run, from the public prices in ``configs/gpu_prices.yaml``."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

PRICES_FILE = Path(__file__).resolve().parents[2] / "configs" / "gpu_prices.yaml"


def load_prices(path: str | Path = PRICES_FILE) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def gpu_price(gpu_name: str | None, prices: dict[str, Any]) -> dict[str, Any] | None:
    """The first price entry whose ``match`` regex is in ``gpu_name``. None if there is none."""
    if not gpu_name:
        return None
    for entry in prices.get("gpus", []):
        if re.search(entry["match"], gpu_name):
            return entry
    return None


def run_cost(
    gpu_name: str | None, wall_clock_s: float | None, num_gpus: int = 1, prices: Any = None
) -> dict[str, Any]:
    """GPU hours and the derived $ cost. ``usd`` is None when the GPU has no public price."""
    gpu_hours = None if wall_clock_s is None else round(num_gpus * wall_clock_s / 3600, 4)
    if prices is None:
        prices = load_prices() if PRICES_FILE.is_file() else {}
    entry = gpu_price(gpu_name, prices)
    out: dict[str, Any] = {"gpu_hours": gpu_hours, "usd_per_hour": None, "usd": None}
    if entry is not None:
        out["usd_per_hour"] = entry["usd_per_hour"]
        out["usd"] = None if gpu_hours is None else round(gpu_hours * entry["usd_per_hour"], 4)
        out["price_source"] = entry.get("source")
        out["prices_retrieved"] = str(prices.get("retrieved"))
    return out
