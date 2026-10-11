# London Airbnb Price Prediction

An end-to-end machine learning project that predicts the **nightly price of an Airbnb listing in London**. It uses real listings data from [Inside Airbnb](https://insideairbnb.com/), compares several regression models, and serves the best one in a Flask web app that ships as a Docker container.

![Python](https://img.shields.io/badge/python-3.12-blue) ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-orange) ![XGBoost](https://img.shields.io/badge/XGBoost-3.4-green) ![Flask](https://img.shields.io/badge/Flask-3.1-lightgrey) ![Docker](https://img.shields.io/badge/Docker-ready-2496ED)

## Why

Hosts want to price competitively without leaving money on the table, and guests want to know whether a listing is fairly priced. This project learns how location, size, room type, reviews and host attributes drive London Airbnb prices, then turns that into a simple pricing tool.

## Pipeline

```
Inside Airbnb listings.csv.gz
        │  download_listings()
        ▼
  Clean: parse £ prices, drop outliers (<£10 / >£1,000), remove monthly lets
        │  clean() / add_features()
        ▼
  Feature engineering
    • distance to central London (haversine to Charing Cross)
    • bathrooms parsed from text ("1.5 shared baths"), shared-bath flag
    • amenities count, property types grouped into 6 categories
        │
        ▼
  Preprocess (impute → scale / one-hot) + log-transformed target
        │  train()
        ▼
  Compare Ridge · Random Forest · XGBoost on a held-out 20% test set
        │  best by MAE
        ▼
  models/model.joblib + metrics.json + metadata.json
        │
        ▼
  Flask app (form + JSON API) → Docker
```

## Project structure

```
├── app/
│   ├── app.py               # Flask app: web form, /api/predict, /health
│   ├── templates/index.html
│   └── static/style.css
├── src/airbnb_price/
│   ├── config.py            # paths, data URL, feature lists
│   ├── data.py              # download, cleaning, feature engineering
│   ├── train.py             # model comparison and saving
│   └── predict.py           # inference for one listing
├── scripts/run_pipeline.py  # download → clean → train
├── tests/                   # unit tests for the cleaning helpers
├── data/                    # raw and processed data (git-ignored)
├── models/                  # trained model and metrics
├── Dockerfile
└── requirements.txt
```

## Quick start

```bash
git clone https://github.com/<your-username>/london-airbnb-price-prediction.git
cd london-airbnb-price-prediction

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1. Download the data and train the models (a few minutes)
python scripts/run_pipeline.py

# 2. Run the web app
python app/app.py                  # open http://localhost:5000
```

> The data snapshot date is set in `src/airbnb_price/config.py` (`SNAPSHOT_DATE`). Inside Airbnb releases new snapshots regularly and removes old ones, so if the download fails, copy the latest London `listings.csv.gz` date from https://insideairbnb.com/get-the-data/.

### API

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"neighbourhood_cleansed": "Camden", "room_type": "Entire home/apt",
       "property_group": "apartment", "accommodates": 4, "bedrooms": 2}'
# {"predicted_price_gbp": ...}
```

You can leave out any field except `neighbourhood_cleansed`. Missing fields fall back to the training-set median.

### Docker

Train first so that `models/` contains the model, then:

```bash
docker build -t london-airbnb .
docker run -p 5000:5000 london-airbnb
```

### Tests

```bash

pytest
```

## Results

After training, the scores for each model are saved to `models/metrics.json`. The scores below are on a held-out 20% test set (11,725 listings, with 46,900 used for training) from the 17 September 2026 Inside Airbnb snapshot. XGBoost performed best: its predictions are about £49 off the real nightly price on average, and it explains about 71% of the variation in prices.

| Model | MAE (£) | RMSE (£) | R² |
|---|---|---|---|
| Ridge | 61.73 | 111.67 | 0.514 |
| Random Forest | 51.00 | 90.59 | 0.680 |
| **XGBoost** | **49.27** | **86.85** | **0.706** |

## Possible next steps

- Add text features from listing descriptions (TF-IDF or embeddings)
- Add SHAP explanations for each prediction
- Tune hyperparameters with cross-validation (Optuna)
- Deploy to Render / Railway / Azure App Service

## Data and acknowledgements

- Data: [Inside Airbnb](https://insideairbnb.com/), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- Inspired by Kalyan Murapaka's [Airbnb Price Prediction](https://github.com/KalyanMurapaka45/Airbnb-Price-Prediction) project. This repository is an independent re-implementation with a different dataset, code and app.

## Author

**Shibu Karosiya**
