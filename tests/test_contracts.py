from __future__ import annotations

import pandas as pd
import pytest

from f1predict.data.contracts import validate_table


def test_contract_rejects_duplicate_driver_race() -> None:
    frame = pd.DataFrame(
        [
            {
                "season": 2024,
                "round": 1,
                "driver_id": "same",
                "constructor_id": "team",
                "finish_position": 1,
                "status": "Finished",
            },
            {
                "season": 2024,
                "round": 1,
                "driver_id": "same",
                "constructor_id": "team",
                "finish_position": 2,
                "status": "Finished",
            },
        ]
    )
    with pytest.raises(ValueError, match="duplicate primary keys"):
        validate_table("results", frame)
