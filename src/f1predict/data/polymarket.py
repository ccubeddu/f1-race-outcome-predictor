"""Read-only, timestamped prediction-market snapshots from Polymarket.

The caller supplies the race's real entry list and an explicit market-slug mapping.
Market ``endDate`` can be a cancellation deadline and is never used as race time.
These are prediction-market prices, not historical sportsbook closing odds.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pandas as pd


@dataclass
class PolymarketSnapshot:
    quotes: pd.DataFrame
    metadata: dict[str, Any]
    raw: dict[str, Any]


def _utc(value: str | datetime) -> datetime:
    stamp = (
        datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    )
    if not isinstance(stamp, datetime) or stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError("cutoff must include an explicit UTC offset")
    return stamp.astimezone(UTC)


def _array(value: Any, name: str) -> list[Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid {name}") from exc
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON array")
    return value


def yes_token(market: Mapping[str, Any]) -> str:
    """Resolve YES by outcome label; outcome ordering is not assumed."""
    outcomes = _array(market.get("outcomes"), "outcomes")
    tokens = _array(market.get("clobTokenIds"), "clobTokenIds")
    labels = [str(outcome).lower() for outcome in outcomes]
    if labels.count("yes") != 1 or len(tokens) != len(outcomes):
        raise ValueError(f"Market {market.get('slug')} has no unique YES token")
    token = str(tokens[labels.index("yes")])
    if not token.isdigit():
        raise ValueError("Invalid outcome token ID")
    return token


def _latest(history: Any, cutoff: datetime) -> dict[str, Any] | None:
    if not isinstance(history, dict) or not isinstance(history.get("history"), list):
        raise ValueError("Malformed Polymarket history response")
    points: dict[int, float] = {}
    for point in history["history"]:
        if not isinstance(point, dict):
            raise ValueError("Malformed history observation")
        stamp, price = point.get("t"), point.get("p")
        if isinstance(stamp, bool) or not isinstance(stamp, int) or stamp < 0:
            raise ValueError("History timestamp must be a nonnegative Unix integer")
        if isinstance(price, bool):
            raise ValueError("Invalid history probability")
        try:
            price = float(price)
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid history probability") from exc
        if not math.isfinite(price) or not 0 <= price <= 1:
            raise ValueError("History probability must be finite and between 0 and 1")
        if stamp in points and points[stamp] != price:
            raise ValueError("Conflicting observations at the same timestamp")
        points[stamp] = price
    eligible = [stamp for stamp in points if stamp <= cutoff.timestamp()]
    if not eligible:
        return None
    stamp = max(eligible)
    return {"t": stamp, "p": points[stamp]}


def select_snapshot(
    event: dict[str, Any],
    histories: Mapping[str, Any],
    *,
    driver_markets: Mapping[str, str],
    expected_driver_ids: Sequence[str],
    cutoff: str | datetime,
    max_age_seconds: float = 3600,
) -> PolymarketSnapshot:
    """Select only observations at or before cutoff, without normalizing prices.

    Extra event outcomes are audited separately. Consumers must examine unmatched
    mass and unknown histories before interpreting a normalized grid distribution.
    No resolved ``outcomePrices`` or future history points enter the snapshot.
    """
    cutoff_time = _utc(cutoff)
    if not math.isfinite(max_age_seconds) or max_age_seconds <= 0:
        raise ValueError("max_age_seconds must be positive and finite")
    expected = list(expected_driver_ids)
    if not expected or len(expected) != len(set(expected)):
        raise ValueError("Expected driver IDs must be nonempty and unique")
    if set(driver_markets) != set(expected):
        raise ValueError("Explicit driver mapping must exactly cover the actual entry list")
    if len(set(driver_markets.values())) != len(driver_markets):
        raise ValueError("Each driver must map to a different market slug")
    markets = event.get("markets")
    if not isinstance(markets, list) or not markets:
        raise ValueError("Event has no markets")
    by_slug: dict[str, dict[str, Any]] = {}
    by_token: dict[str, str] = {}
    for market in markets:
        if not isinstance(market, dict) or not isinstance(market.get("slug"), str):
            raise ValueError("Malformed event market")
        slug = market["slug"]
        token = yes_token(market)
        if slug in by_slug or token in by_token:
            raise ValueError("Event contains duplicate market slugs or YES tokens")
        by_slug[slug] = market
        by_token[token] = slug
    missing = set(driver_markets.values()) - set(by_slug)
    if missing:
        raise ValueError(f"Mapped markets absent from event: {sorted(missing)}")

    observations = {token: _latest(histories.get(token), cutoff_time) for token in by_token}
    rows: list[dict[str, Any]] = []
    for driver in expected:
        market = by_slug[driver_markets[driver]]
        token = yes_token(market)
        point = observations[token]
        if point is None:
            raise ValueError(f"No observation at or before cutoff for {driver}")
        age = cutoff_time.timestamp() - point["t"]
        if age > max_age_seconds:
            raise ValueError(f"Stale quote for {driver}: {age:.0f}s before cutoff")
        if point["p"] <= 0:
            raise ValueError(f"Zero probability for {driver}; cannot convert to finite odds")
        rows.append(
            {
                "driver_id": driver,
                "decimal_odds": 1 / point["p"],
                "market_probability": point["p"],
                "quote_at": datetime.fromtimestamp(point["t"], UTC).isoformat(),
                "quote_age_seconds": age,
                "market_slug": market["slug"],
                "token_id": token,
            }
        )

    mapped = set(driver_markets.values())
    unmatched = []
    for slug, market in by_slug.items():
        if slug in mapped:
            continue
        token = yes_token(market)
        point = observations[token]
        age = cutoff_time.timestamp() - point["t"] if point else None
        unmatched.append(
            {
                "market_slug": slug,
                "label": market.get("groupItemTitle", market.get("question")),
                "token_id": token,
                "probability": point["p"] if point else None,
                "quote_at": datetime.fromtimestamp(point["t"], UTC).isoformat() if point else None,
                "quote_age_seconds": age,
                "fresh": age is not None and age <= max_age_seconds,
            }
        )
    quoted = [item for item in unmatched if item["probability"] is not None]
    metadata = {
        "source": "polymarket",
        "source_kind": "prediction_market",
        "source_url": f"https://polymarket.com/event/{event.get('slug', '')}",
        "event_id": event.get("id"),
        "event_slug": event.get("slug"),
        "event_title": event.get("title"),
        "cutoff_at": cutoff_time.isoformat(),
        "max_age_seconds": max_age_seconds,
        "grid_size": len(expected),
        "mapping_coverage": len(rows) / len(expected),
        "matched_probability_sum": sum(row["market_probability"] for row in rows),
        "unmatched_observed_probability_sum": sum(item["probability"] for item in quoted),
        "unmatched_missing_histories": len(unmatched) - len(quoted),
        "unmatched_stale_histories": sum(not item["fresh"] for item in quoted),
        "unmatched_markets": unmatched,
        "probabilities_normalized": False,
        "price_type": "historical CLOB price; executable bid/ask not available from this endpoint",
    }
    return PolymarketSnapshot(
        pd.DataFrame(rows), metadata, {"event": event, "histories": dict(histories)}
    )


class PolymarketClient:
    """Public GET endpoints only. No API keys, account, order, or trade operations."""

    def __init__(
        self, *, timeout_seconds: float = 30, transport: httpx.BaseTransport | None = None
    ) -> None:
        self._client = httpx.Client(
            timeout=timeout_seconds, transport=transport, follow_redirects=True
        )

    def __enter__(self) -> PolymarketClient:
        return self

    def __exit__(self, *_: object) -> None:
        self._client.close()

    def event(self, slug: str) -> dict[str, Any]:
        response = self._client.get(
            "https://gamma-api.polymarket.com/events", params={"slug": slug}
        )
        response.raise_for_status()
        payload = response.json()
        if (
            not isinstance(payload, list)
            or len(payload) != 1
            or not isinstance(payload[0], dict)
            or payload[0].get("slug") != slug
        ):
            raise ValueError("Expected one event matching the exact requested slug")
        return payload[0]

    def history(self, token_id: str, *, start: datetime, end: datetime) -> dict[str, Any]:
        response = self._client.get(
            "https://clob.polymarket.com/prices-history",
            params={
                "market": token_id,
                "startTs": int(start.timestamp()),
                "endTs": int(end.timestamp()),
                "fidelity": 5,
            },
        )
        response.raise_for_status()
        return response.json()

    def fetch_snapshot(
        self,
        event_slug: str,
        *,
        driver_markets: Mapping[str, str],
        expected_driver_ids: Sequence[str],
        cutoff: str | datetime,
        max_age_seconds: float = 3600,
    ) -> PolymarketSnapshot:
        cutoff_time = _utc(cutoff)
        if not math.isfinite(max_age_seconds) or max_age_seconds <= 0:
            raise ValueError("max_age_seconds must be positive and finite")
        event = self.event(event_slug)
        histories = {}
        # Fetch extras too: normalizing an incomplete field could otherwise hide
        # an unlisted driver or material Other outcome at the requested cutoff.
        for market in event.get("markets", []):
            token = yes_token(market)
            histories[token] = self.history(
                token,
                start=cutoff_time - timedelta(seconds=max_age_seconds + 600),
                end=cutoff_time,
            )
        snapshot = select_snapshot(
            event,
            histories,
            driver_markets=driver_markets,
            expected_driver_ids=expected_driver_ids,
            cutoff=cutoff_time,
            max_age_seconds=max_age_seconds,
        )
        snapshot.metadata["retrieved_at"] = datetime.now(UTC).isoformat()
        return snapshot
