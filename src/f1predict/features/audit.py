from __future__ import annotations

import json
from typing import Any

import pandas as pd

from f1predict.config import DataPaths
from f1predict.features.builder import assert_no_leakage


def _family_counts(features: list[str]) -> dict[str, int]:
    families = {
        "driver_form": "driver_",
        "constructor": "constructor_",
        "teammate": "teammate_",
        "track": "track_",
        "practice": "practice_",
        "openf1_detail": "openf1_",
        "static": "static_",
        "post_quali": "post_",
    }
    return {
        label: sum(column.startswith(prefix) for column in features)
        for label, prefix in families.items()
    }


def build_feature_audit(paths: DataPaths) -> dict[str, Any]:
    manifest_path = paths.features / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("Feature matrices do not exist; run `f1predict features build` first.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pre = pd.read_parquet(paths.features / "pre_quali.parquet")
    post = pd.read_parquet(paths.features / "post_quali.parquet")
    assert_no_leakage(pre, post, manifest)

    modes: dict[str, Any] = {}
    for name, frame in (("pre_quali", pre), ("post_quali", post)):
        features = manifest["modes"][name]["features"]
        completeness = frame[features].notna().mean()
        modes[name] = {
            "rows": len(frame),
            "races": int(frame[["season", "round"]].drop_duplicates().shape[0]),
            "features": len(features),
            "duplicate_keys": int(frame.duplicated(["season", "round", "driver_id"]).sum()),
            "all_null_features": completeness[completeness.eq(0)].index.tolist(),
            "median_feature_coverage": float(completeness.median()),
            "families": _family_counts(features),
        }

    practice_by_season = (
        pre.groupby("season", observed=True)[["practice_data_available", "openf1_detail_available"]]
        .mean()
        .round(4)
    )
    return {
        "status": "ok",
        "leakage_checks": {
            "pre_quali_has_post_features": False,
            "targets_exposed_as_features": False,
            "duplicate_race_driver_keys": False,
        },
        "modes": modes,
        "coverage": {
            "practice_by_season": {
                str(season): values
                for season, values in practice_by_season.to_dict("index").items()
            },
            "post_quali_position": float(post["post_quali_position"].notna().mean()),
            "post_grid_position": float(post["post_grid_position"].notna().mean()),
        },
        "policy": manifest["leakage_policy"],
    }
