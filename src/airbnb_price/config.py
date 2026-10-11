"""Project-wide paths and settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

# Inside Airbnb publishes a new London snapshot roughly every quarter.
# Check https://insideairbnb.com/get-the-data/ and update the date if needed.
SNAPSHOT_DATE = "2026-09-17"
LISTINGS_URL = (
    "https://data.insideairbnb.com/united-kingdom/england/london/"
    f"{SNAPSHOT_DATE}/data/listings.csv.gz"
)
RAW_LISTINGS = DATA_RAW / "listings.csv.gz"
CLEAN_LISTINGS = DATA_PROCESSED / "listings_clean.csv"

MODEL_PATH = MODELS_DIR / "model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
METADATA_PATH = MODELS_DIR / "metadata.json"

# Nightly prices outside this range (GBP) are treated as data errors / outliers.
PRICE_MIN = 10
PRICE_MAX = 1000

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Features the model uses (all available in the Inside Airbnb detailed listings file).
NUMERIC_FEATURES = [
    "accommodates",
    "bedrooms",
    "beds",
    "bathrooms_num",
    "minimum_nights",
    "availability_365",
    "number_of_reviews",
    "review_scores_rating",
    "host_listings_count",
    "latitude",
    "longitude",
    "amenities_count",
    "dist_to_centre_km",
]
BOOL_FEATURES = [
    "host_is_superhost",
    "instant_bookable",
    "is_shared_bath",
]
CATEGORICAL_FEATURES = [
    "neighbourhood_cleansed",
    "room_type",
    "property_group",
]
TARGET = "price"

# Charing Cross, the traditional centre point of London.
LONDON_CENTRE = (51.5074, -0.1278)
