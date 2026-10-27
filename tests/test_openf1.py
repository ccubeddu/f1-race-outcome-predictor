from __future__ import annotations

import pandas as pd

from f1predict.data.openf1 import normalize_sessions


def test_normalize_sessions_maps_practice_to_race_round() -> None:
    races = pd.DataFrame([{"season": 2024, "round": 1, "race_date": "2024-03-02"}])
    sessions = [
        {
            "year": 2024,
            "meeting_key": 10,
            "session_key": 11,
            "session_name": "Practice 1",
            "session_type": "Practice",
            "date_start": "2024-02-29T11:30:00+00:00",
            "date_end": "2024-02-29T12:30:00+00:00",
            "is_cancelled": False,
            "circuit_key": 1,
            "circuit_short_name": "Sakhir",
            "country_name": "Bahrain",
            "location": "Sakhir",
        },
        {
            "year": 2024,
            "meeting_key": 9,
            "session_key": 8,
            "session_name": "Day 1",
            "session_type": "Practice",
            "date_start": "2024-02-21T07:00:00+00:00",
            "date_end": "2024-02-21T15:00:00+00:00",
            "is_cancelled": False,
        },
    ]
    frame = normalize_sessions(sessions, races)
    assert len(frame) == 1
    assert frame.loc[0, "round"] == 1
    assert frame.loc[0, "session"] == "FP1"
