"""Download and clean the Inside Airbnb London listings data."""
from __future__ import annotations

import logging
import re
import urllib.request

import numpy as np
import pandas as pd

from . import config

log = logging.getLogger(__name__)

RAW_COLUMNS = [
    "id",
    "price",
    "neighbourhood_cleansed",
    "latitude",
    "longitude",
    "property_type",
    "room_type",
    "accommodates",
    "bathrooms",
    "bathrooms_text",
    "bedrooms",
    "beds",
    "amenities",
    "minimum_nights",
    "availability_365",
    "number_of_reviews",
    "review_scores_rating",
    "host_is_superhost",
    "host_listings_count",
    "instant_bookable",
]


def download_listings(url: str = config.LISTINGS_URL, force: bool = False):
    """Download the compressed listings file into data/raw."""
    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    if config.RAW_LISTINGS.exists() and not force:
        log.info("Raw data already present at %s", config.RAW_LISTINGS)
        return config.RAW_LISTINGS
    log.info("Downloading %s", url)
    urllib.request.urlretrieve(url, config.RAW_LISTINGS)
    log.info("Saved to %s", config.RAW_LISTINGS)
    return config.RAW_LISTINGS


def load_raw(path=config.RAW_LISTINGS) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Listings file is missing expected columns: {missing}")
    return df[RAW_COLUMNS].copy()


# ---------------------------------------------------------------- helpers

def parse_price(series: pd.Series) -> pd.Series:
    """'$1,234.00' / '£85' -> 1234.0 / 85.0"""
    return pd.to_numeric(
        series.astype(str).str.replace(r"[^0-9.]", "", regex=True), errors="coerce"
    )


def parse_bool(series: pd.Series) -> pd.Series:
    return series.map({"t": 1, "f": 0, True: 1, False: 0}).astype("float")


def bathrooms_from_text(text) -> float:
    """'1.5 shared baths' -> 1.5, 'Half-bath' -> 0.5"""
    if not isinstance(text, str):
        return np.nan
    if "half" in text.lower():
        return 0.5
    match = re.search(r"\d+(\.\d+)?", text)
    return float(match.group()) if match else np.nan


def count_amenities(value) -> int:
    """Amenities are stored as a JSON-like list string."""
    if not isinstance(value, str) or value.strip() in ("", "[]"):
        return 0
    return value.count(",") + 1


def group_property_type(value) -> str:
    """Collapse ~100 property types into a handful of groups."""
    v = str(value).lower()
    if "hotel" in v or "hostel" in v or "aparthotel" in v:
        return "hotel"
    if "condo" in v or "rental unit" in v or "loft" in v or "serviced apartment" in v:
        return "apartment"
    if "townhouse" in v or "home" in v or "house" in v or "villa" in v or "cottage" in v:
        return "house"
    if "guest" in v or "bed and breakfast" in v:
        return "guesthouse"
    if "boat" in v or "houseboat" in v:
        return "boat"
    return "other"


def haversine_km(lat, lon, lat0=config.LONDON_CENTRE[0], lon0=config.LONDON_CENTRE[1]):
    lat, lon = np.radians(lat), np.radians(lon)
    lat0, lon0 = np.radians(lat0), np.radians(lon0)
    a = np.sin((lat - lat0) / 2) ** 2 + np.cos(lat) * np.cos(lat0) * np.sin((lon - lon0) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------- main steps

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering shared by training and the web app."""
    df = df.copy()
    if "bathrooms_text" in df:
        from_text = df["bathrooms_text"].map(bathrooms_from_text)
        base = pd.to_numeric(df.get("bathrooms"), errors="coerce") if "bathrooms" in df else from_text
        df["bathrooms_num"] = base.fillna(from_text)
        df["is_shared_bath"] = (
            df["bathrooms_text"].astype(str).str.contains("shared", case=False).astype(int)
        )
    if "amenities" in df:
        df["amenities_count"] = df["amenities"].map(count_amenities)
    if "property_type" in df:
        df["property_group"] = df["property_type"].map(group_property_type)
    for col in ("host_is_superhost", "instant_bookable"):
        if col in df and not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = parse_bool(df[col])
    df["dist_to_centre_km"] = haversine_km(df["latitude"], df["longitude"])
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["price"] = parse_price(df["price"])
    before = len(df)
    df = df[df["price"].between(config.PRICE_MIN, config.PRICE_MAX)]
    # Very long minimum stays are effectively monthly lets, not nightly rentals.
    df = df[pd.to_numeric(df["minimum_nights"], errors="coerce") <= 365]
    df = add_features(df)
    keep = (
        ["id", config.TARGET]
        + config.NUMERIC_FEATURES
        + config.BOOL_FEATURES
        + config.CATEGORICAL_FEATURES
    )
    df = df[keep].drop_duplicates("id").reset_index(drop=True)
    log.info("Cleaned listings: %d -> %d rows", before, len(df))
    return df


def build_clean_dataset(download: bool = True) -> pd.DataFrame:
    if download:
        download_listings()
    df = clean(load_raw())
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.CLEAN_LISTINGS, index=False)
    return df
