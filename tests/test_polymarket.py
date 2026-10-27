from __future__ import annotations

import copy
import json

import httpx
import pytest

from f1predict.data.polymarket import PolymarketClient, select_snapshot, yes_token

CUTOFF = "2026-10-25T18:00:00Z"
STAMP = 1792951200


def fixture():
    event = {
        "id": "example",
        "slug": "f1-example",
        "title": "Example race winner",
        "endDate": "2026-11-01T20:00:00Z",
        "markets": [
            {
                "slug": slug,
                "outcomes": json.dumps(["No", "Yes"]),
                "clobTokenIds": json.dumps([str(token + 1), str(token)]),
                "outcomePrices": '["1", "0"]',
            }
            for slug, token in [("driver-a", 10), ("driver-b", 20), ("other", 30)]
        ],
    }
    histories = {
        "10": {"history": [{"t": STAMP - 60, "p": 0.6}, {"t": STAMP + 60, "p": 1}]},
        "20": {"history": [{"t": STAMP - 90, "p": 0.35}]},
        "30": {"history": [{"t": STAMP - 40, "p": 0.05}]},
    }
    return event, histories


def select(event, histories, **kwargs):
    return select_snapshot(
        event,
        histories,
        driver_markets={"a": "driver-a", "b": "driver-b"},
        expected_driver_ids=["a", "b"],
        cutoff=CUTOFF,
        **kwargs,
    )


def test_snapshot_ignores_future_settlement_and_audits_excluded_mass():
    event, histories = fixture()
    snapshot = select(event, histories)
    assert snapshot.quotes["market_probability"].tolist() == [0.6, 0.35]
    assert snapshot.quotes["decimal_odds"].tolist() == pytest.approx([1 / 0.6, 1 / 0.35])
    assert snapshot.metadata["unmatched_observed_probability_sum"] == 0.05
    assert snapshot.metadata["matched_probability_sum"] == pytest.approx(0.95)
    assert not snapshot.metadata["probabilities_normalized"]
    assert snapshot.metadata["cutoff_at"] == "2026-10-25T18:00:00+00:00"
    assert snapshot.raw["event"] == event


@pytest.mark.parametrize(
    ("history", "message"),
    [
        ({"history": []}, "No observation"),
        ({"history": [{"t": STAMP + 1, "p": 1}]}, "No observation"),
        ({"history": [{"t": STAMP - 3601, "p": 0.5}]}, "Stale"),
        ({"history": [{"t": STAMP - 1, "p": 0}]}, "Zero probability"),
        ({"history": [{"t": STAMP - 1, "p": float("nan")}]}, "finite"),
        ({"history": [{"t": STAMP - 1, "p": 1.1}]}, "between 0 and 1"),
        ({"history": [{"t": True, "p": 0.5}]}, "Unix integer"),
        ({"history": [{"t": STAMP, "p": 0.3}, {"t": STAMP, "p": 0.6}]}, "Conflicting"),
        ({"error": "not found"}, "Malformed"),
    ],
)
def test_rejects_unusable_mapped_driver_history(history, message):
    event, histories = fixture()
    histories["10"] = history
    with pytest.raises(ValueError, match=message):
        select(event, histories)


def test_requires_complete_explicit_grid_mapping_and_unique_markets():
    event, histories = fixture()
    with pytest.raises(ValueError, match="actual entry list"):
        select_snapshot(
            event,
            histories,
            driver_markets={"a": "driver-a"},
            expected_driver_ids=["a", "b"],
            cutoff=CUTOFF,
        )
    event["markets"].append(copy.deepcopy(event["markets"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        select(event, histories)


def test_reports_unknown_unmatched_outcomes_and_rejects_naive_cutoff():
    event, histories = fixture()
    histories["30"] = {"history": []}
    snapshot = select(event, histories)
    assert snapshot.metadata["unmatched_missing_histories"] == 1
    assert snapshot.metadata["unmatched_markets"][0]["probability"] is None
    with pytest.raises(ValueError, match="UTC offset"):
        select_snapshot(
            event,
            histories,
            driver_markets={"a": "driver-a", "b": "driver-b"},
            expected_driver_ids=["a", "b"],
            cutoff="2026-10-25T18:00:00",
        )


def test_client_uses_only_public_gets_and_exact_slug():
    event, histories = fixture()
    calls = []

    def handler(request):
        calls.append(request)
        assert request.method == "GET"
        assert "Authorization" not in request.headers
        if request.url.host == "gamma-api.polymarket.com":
            return httpx.Response(200, json=[event])
        assert int(request.url.params["endTs"]) == STAMP
        return httpx.Response(200, json=histories[request.url.params["market"]])

    with PolymarketClient(transport=httpx.MockTransport(handler)) as client:
        snapshot = client.fetch_snapshot(
            "f1-example",
            driver_markets={"a": "driver-a", "b": "driver-b"},
            expected_driver_ids=["a", "b"],
            cutoff=CUTOFF,
        )
        with pytest.raises(ValueError, match="exact requested slug"):
            client.event("other-event")
    assert len(calls) == 5
    assert snapshot.metadata["mapping_coverage"] == 1
    assert yes_token(event["markets"][0]) == "10"
