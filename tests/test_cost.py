import pytest

from mathscribe.cost import PRICES_FILE, gpu_price, load_prices, run_cost

PRICES = {
    "retrieved": "2026-11-07",
    "gpus": [
        {"match": r"\bT4\b", "usd_per_hour": 0.35, "source": "a"},
        {"match": r"\bL4\b", "usd_per_hour": 0.71, "source": "b"},
        {"match": "A100.*40GB", "usd_per_hour": 2.95, "source": "c"},
    ],
}


@pytest.mark.parametrize(
    ("gpu", "price"),
    [
        ("Tesla T4", 0.35),
        ("NVIDIA L4", 0.71),
        ("NVIDIA L40S", None),  # \bL4\b must not match L40S
        ("NVIDIA A100-SXM4-40GB", 2.95),
        ("NVIDIA A100-SXM4-80GB", None),
        ("NVIDIA GeForce RTX 3060 Laptop GPU", None),
        ("API", None),
        (None, None),
    ],
)
def test_gpu_price_matches_device_names(gpu, price):
    entry = gpu_price(gpu, PRICES)
    assert (entry["usd_per_hour"] if entry else None) == price


def test_run_cost_on_a_priced_gpu():
    cost = run_cost("Tesla T4", 7200, prices=PRICES)
    assert cost["gpu_hours"] == 2.0
    assert cost["usd"] == pytest.approx(0.70)
    assert cost["price_source"] == "a"
    assert cost["prices_retrieved"] == "2026-11-07"


def test_run_cost_without_price_keeps_gpu_hours():
    cost = run_cost("NVIDIA GeForce RTX 3060 Laptop GPU", 1800, prices=PRICES)
    assert cost == {"gpu_hours": 0.5, "usd_per_hour": None, "usd": None}
    assert run_cost("Tesla T4", None, prices=PRICES)["usd"] is None


def test_repo_price_file_has_a_t4_price():
    prices = load_prices(PRICES_FILE)
    assert gpu_price("Tesla T4", prices)["usd_per_hour"] > 0
    for entry in prices["gpus"]:
        assert entry["source"].startswith("https://")
