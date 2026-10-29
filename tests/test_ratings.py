from dataclasses import replace

import numpy as np
import pandas as pd

from f1predict.features.ratings import (
    RatingConfig,
    Reliability,
    build_rating_features,
    constructor_lineage,
)


def fixture():
    rows = []
    for race in (1, 2, 3):
        for i in range(4):
            rows.append(
                dict(
                    season=2024,
                    round=race,
                    driver_id=f"d{i}",
                    constructor_id=f"t{i // 2}",
                    race_date=pd.Timestamp(f"2024-03-{race:02d}"),
                    circuit_id="track",
                    label_quali_position=i + 1,
                    target_finish_position=i + 1,
                    target_classified=1,
                    target_dnf=0,
                    target_dns=0,
                    target_dsq=0,
                    label_status="Finished",
                )
            )
    layouts = pd.DataFrame(
        [
            dict(
                season=2024,
                round=r,
                layout_key="layout1",
                layout_bucket="RACE:medium",
                rating_layout_length_km=5.0,
                rating_layout_available=1,
            )
            for r in (1, 2, 3)
        ]
    )
    return pd.DataFrame(rows), layouts


def test_ratings_emit_before_updates_and_ignore_current_and_future_labels():
    data, layouts = fixture()
    original, _ = build_rating_features(data, layouts)
    poisoned = data.copy()
    poisoned.loc[poisoned["round"].ge(2), "label_quali_position"] = (
        5 - poisoned.loc[poisoned["round"].ge(2), "label_quali_position"]
    )
    poisoned.loc[poisoned["round"].ge(2), "target_finish_position"] = (
        5 - poisoned.loc[poisoned["round"].ge(2), "target_finish_position"]
    )
    changed, _ = build_rating_features(poisoned, layouts)
    pd.testing.assert_frame_equal(
        original.loc[original["round"].le(2)], changed.loc[changed["round"].le(2)]
    )
    assert not original.loc[original["round"].eq(3), "rating_quali_driver"].equals(
        changed.loc[changed["round"].eq(3), "rating_quali_driver"]
    )
    assert original.loc[original["round"].eq(1), "asof_race_date"].isna().all()
    assert original.loc[original["round"].eq(1), "rating_race_combined"].eq(0).all()


def test_ratings_are_invariant_to_entrant_row_order():
    data, layouts = fixture()
    a, _ = build_rating_features(data, layouts)
    b, _ = build_rating_features(data.sample(frac=1, random_state=4), layouts)
    pd.testing.assert_frame_equal(a, b)


def test_dnf_does_not_update_race_pace_and_mechanical_is_not_incident():
    data, layouts = fixture()
    mask = data["round"].eq(1) & data.driver_id.eq("d0")
    data.loc[mask, ["target_classified", "target_dnf"]] = [0, 1]
    data.loc[mask, "label_status"] = "Engine"
    ratings, _ = build_rating_features(data, layouts)
    later = ratings.loc[ratings["round"].eq(2) & ratings.driver_id.eq("d0")].iloc[0]
    assert later.rating_race_driver == 0
    assert later.rating_quali_driver > 0
    assert later.rel_driver_dnf_rate > 0.1
    assert later.rel_driver_mechanical_rate > 0.04
    assert later.rel_driver_incident_rate < 0.04


def test_circuit_layout_versions_do_not_share_specific_track_evidence():
    data, layouts = fixture()
    layouts.loc[layouts["round"].eq(3), "layout_key"] = "layout2"
    ratings, _ = build_rating_features(data, layouts)
    r2 = ratings.loc[ratings["round"].eq(2)]
    r3 = ratings.loc[ratings["round"].eq(3)]
    assert r2.rating_context_quali_driver_track_evidence.eq(1).all()
    assert r3.rating_context_quali_driver_track_evidence.eq(0).all()
    assert r3.rating_context_quali_driver_bucket_evidence.eq(2).all()


def test_lineages_are_time_bounded_and_lotus_teams_not_conflated():
    assert constructor_lineage("audi", 2026) == constructor_lineage("sauber", 2025)
    assert constructor_lineage("alfa", 2020) == constructor_lineage("sauber", 2018)
    assert constructor_lineage("lotus_f1", 2013) == constructor_lineage("renault", 2010)
    assert constructor_lineage("lotus_racing", 2011) != constructor_lineage("lotus_f1", 2013)
    assert constructor_lineage("aston_martin", 1959) != constructor_lineage("aston_martin", 2021)
    assert constructor_lineage("cadillac", 2026) == "cadillac"


def test_driver_rating_survives_team_move_while_car_state_does_not_transfer():
    data, layouts = fixture()
    data.loc[data["round"].eq(3) & data.driver_id.eq("d0"), "constructor_id"] = "new_team"
    ratings, _ = build_rating_features(data, layouts)
    mover = ratings.loc[ratings["round"].eq(3) & ratings.driver_id.eq("d0")].iloc[0]
    assert mover.rating_quali_driver > 0
    assert mover.rating_quali_constructor == 0
    assert mover.rating_driver_new_team == 1
    assert mover.rating_driver_team_changes_pre == 1


def test_rebrand_carries_discounted_latent_state_not_championship_points():
    data, layouts = fixture()
    data["season"] = 2020
    layouts["season"] = 2020
    data.loc[data.constructor_id.eq("t0"), "constructor_id"] = "racing_point"
    config = replace(RatingConfig(), constructor_season_retention=1, driver_season_retention=1)
    base, _ = build_rating_features(data, layouts, config)
    expected = base.loc[
        base["round"].eq(3) & base.driver_id.eq("d0"), "rating_quali_constructor"
    ].iloc[0]
    data.loc[data["round"].eq(3), "season"] = 2021
    layouts.loc[layouts["round"].eq(3), "season"] = 2021
    data.loc[data["round"].eq(3) & data.constructor_id.eq("racing_point"), "constructor_id"] = (
        "aston_martin"
    )
    changed, manifest = build_rating_features(data, layouts, config)
    actual = changed.loc[
        changed["round"].eq(3) & changed.driver_id.eq("d0"), "rating_quali_constructor"
    ].iloc[0]
    assert np.isclose(actual, expected * 0.5)
    assert manifest["rebrands"][0]["to"] == "aston_martin"


def test_reliability_cause_probabilities_sum_to_total_and_unknown_is_not_mechanical():
    state = Reliability(starts=10, mechanical=2, incident=1, other=3)
    values = state.features("rel")
    assert np.isclose(
        sum(values[f"rel_{cause}_rate"] for cause in ("mechanical", "incident", "other")),
        values["rel_dnf_rate"],
    )
    data, layouts = fixture()
    mask = data["round"].eq(1) & data.driver_id.eq("d0")
    data.loc[mask, ["target_classified", "target_dnf"]] = [0, 1]
    data.loc[mask, "label_status"] = "Retired"
    ratings, _ = build_rating_features(data, layouts)
    later = ratings.loc[ratings["round"].eq(2) & ratings.driver_id.eq("d0")].iloc[0]
    assert later.rel_driver_other_rate > 0.02
    assert later.rel_driver_mechanical_rate < 0.04


def test_truncating_future_races_keeps_all_earlier_snapshots_identical():
    data, layouts = fixture()
    full, _ = build_rating_features(data, layouts)
    prefix, _ = build_rating_features(
        data.loc[data["round"].le(2)], layouts.loc[layouts["round"].le(2)]
    )
    pd.testing.assert_frame_equal(full.loc[full["round"].le(2)].reset_index(drop=True), prefix)


def test_unlabelled_live_entrants_receive_the_same_pre_race_state():
    data, layouts = fixture()
    historical, _ = build_rating_features(data, layouts)
    labels = [
        "label_quali_position",
        "target_finish_position",
        "target_classified",
        "target_dnf",
        "target_dns",
        "target_dsq",
    ]
    live = data.copy()
    for column in labels:
        live[column] = live[column].astype(float)
    live.loc[live["round"].eq(3), labels] = np.nan
    unknown, _ = build_rating_features(live, layouts)
    pd.testing.assert_frame_equal(historical, unknown)
