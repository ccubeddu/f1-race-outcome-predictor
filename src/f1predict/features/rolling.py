from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def shifted_rolling_mean(
    frame: pd.DataFrame,
    groups: str | Sequence[str],
    value: str,
    window: int,
) -> pd.Series:
    """Rolling mean that can never see the current row."""
    keys = [groups] if isinstance(groups, str) else list(groups)
    return frame.groupby(keys, sort=False, observed=True)[value].transform(
        lambda series: series.shift(1).rolling(window, min_periods=1).mean()
    )


def shifted_rolling_std(
    frame: pd.DataFrame,
    groups: str | Sequence[str],
    value: str,
    window: int,
) -> pd.Series:
    """Rolling standard deviation that can never see the current row."""
    keys = [groups] if isinstance(groups, str) else list(groups)
    return frame.groupby(keys, sort=False, observed=True)[value].transform(
        lambda series: series.shift(1).rolling(window, min_periods=2).std()
    )


def shifted_expanding_mean(
    frame: pd.DataFrame,
    groups: str | Sequence[str],
    value: str,
) -> pd.Series:
    """All-history mean through the preceding row only."""
    keys = [groups] if isinstance(groups, str) else list(groups)
    return frame.groupby(keys, sort=False, observed=True)[value].transform(
        lambda series: series.shift(1).expanding(min_periods=1).mean()
    )


def prior_count(frame: pd.DataFrame, groups: str | Sequence[str]) -> pd.Series:
    keys = [groups] if isinstance(groups, str) else list(groups)
    return frame.groupby(keys, sort=False, observed=True).cumcount()
