import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from airbnb_price.data import (  # noqa: E402
    bathrooms_from_text, count_amenities, group_property_type, haversine_km, parse_price,
)


def test_parse_price():
    assert parse_price(pd.Series(["$1,234.00", "£85", None])).tolist()[:2] == [1234.0, 85.0]


def test_bathrooms_from_text():
    assert bathrooms_from_text("1.5 shared baths") == 1.5
    assert bathrooms_from_text("Half-bath") == 0.5


def test_count_amenities():
    assert count_amenities('["Wifi", "Kitchen", "Heating"]') == 3
    assert count_amenities("[]") == 0


def test_group_property_type():
    assert group_property_type("Entire rental unit") == "apartment"
    assert group_property_type("Room in boutique hotel") == "hotel"


def test_haversine_centre_is_zero():
    assert haversine_km(51.5074, -0.1278) < 0.01
