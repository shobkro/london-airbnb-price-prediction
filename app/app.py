"""Flask web app: enter listing details, get a predicted nightly price."""
import json
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from airbnb_price import config  # noqa: E402
from airbnb_price.predict import load_metadata, predict_price  # noqa: E402

app = Flask(__name__)

NUMBER_FIELDS = ["accommodates", "bedrooms", "beds", "bathrooms_num", "minimum_nights",
                 "availability_365", "number_of_reviews", "review_scores_rating",
                 "amenities_count"]
CHECKBOX_FIELDS = ["host_is_superhost", "instant_bookable", "is_shared_bath"]
SELECT_FIELDS = ["neighbourhood_cleansed", "room_type", "property_group"]


def form_to_listing(form) -> dict:
    listing = {f: form.get(f) for f in SELECT_FIELDS}
    for f in NUMBER_FIELDS:
        value = form.get(f)
        listing[f] = float(value) if value not in (None, "") else None
    for f in CHECKBOX_FIELDS:
        listing[f] = 1 if form.get(f) else 0
    return listing


@app.route("/", methods=["GET", "POST"])
def index():
    meta = load_metadata()
    metrics = json.loads(config.METRICS_PATH.read_text())
    price, error, values = None, None, {}
    if request.method == "POST":
        values = request.form.to_dict()
        try:
            price = predict_price(form_to_listing(request.form))
        except (ValueError, KeyError) as exc:
            error = f"Could not make a prediction: {exc}"
    return render_template("index.html", meta=meta, metrics=metrics,
                           price=price, error=error, values=values)


@app.post("/api/predict")
def api_predict():
    """JSON API, e.g. {"neighbourhood_cleansed": "Camden", "room_type": "Entire home/apt", ...}"""
    listing = request.get_json(force=True, silent=True) or {}
    try:
        return jsonify({"predicted_price_gbp": predict_price(listing)})
    except (ValueError, KeyError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
