"""Load the trained model and predict prices for new listings."""
from __future__ import annotations

import json
from functools import lru_cache

import joblib
import pandas as pd

from . import config
from .data import haversine_km
from .train import FEATURES


@lru_cache(maxsize=1)
def load_model():
    return joblib.load(config.MODEL_PATH)


@lru_cache(maxsize=1)
def load_metadata() -> dict:
    return json.loads(config.METADATA_PATH.read_text())


def prepare(listing: dict) -> pd.DataFrame:
    """Turn one listing (as submitted by the web form) into a model-ready row.

    Fields the user doesn't provide are filled with training-set medians,
    and latitude/longitude default to the centre of the chosen neighbourhood.
    """
    meta = load_metadata()
    hood = listing.get("neighbourhood_cleansed")
    if hood not in meta["neighbourhood_coords"]:
        raise ValueError(f"Unknown neighbourhood: {hood!r}")
    for col in config.NUMERIC_FEATURES + config.BOOL_FEATURES:
        value = listing.get(col)
        if value not in (None, ""):
            try:
                float(value)
            except (TypeError, ValueError):
                raise ValueError(f"{col} must be a number, got {value!r}") from None

    row = dict(meta["defaults"])
    row.update({k: v for k, v in listing.items() if v not in (None, "")})

    coords = meta["neighbourhood_coords"].get(hood)
    if coords and ("latitude" not in listing or "longitude" not in listing):
        row["latitude"], row["longitude"] = coords["latitude"], coords["longitude"]

    row["dist_to_centre_km"] = float(haversine_km(float(row["latitude"]), float(row["longitude"])))
    df = pd.DataFrame([row])
    for col in config.NUMERIC_FEATURES + config.BOOL_FEATURES:
        df[col] = pd.to_numeric(df.get(col), errors="coerce")
    for col in config.CATEGORICAL_FEATURES:
        if col not in df:
            df[col] = None
    return df[FEATURES]


def predict_price(listing: dict) -> float:
    return round(float(load_model().predict(prepare(listing))[0]), 2)
