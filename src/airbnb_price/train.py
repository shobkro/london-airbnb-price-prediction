"""Train and compare regression models, then save the best one."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from . import config

log = logging.getLogger(__name__)

FEATURES = config.NUMERIC_FEATURES + config.BOOL_FEATURES + config.CATEGORICAL_FEATURES


def make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    boolean = SimpleImputer(strategy="most_frequent")
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric, config.NUMERIC_FEATURES),
            ("bool", boolean, config.BOOL_FEATURES),
            ("cat", categorical, config.CATEGORICAL_FEATURES),
        ]
    )


def candidate_models() -> dict:
    rs = config.RANDOM_STATE
    return {
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=rs
        ),
        "xgboost": XGBRegressor(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=rs,
            n_jobs=-1,
        ),
    }


def build_pipeline(model) -> TransformedTargetRegressor:
    # Prices are right-skewed, so the model learns log(price) and predictions
    # are converted back to pounds automatically.
    return TransformedTargetRegressor(
        regressor=Pipeline([("prep", make_preprocessor()), ("model", model)]),
        func=np.log1p,
        inverse_func=np.expm1,
    )


def evaluate(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train(df: pd.DataFrame) -> dict:
    X, y = df[FEATURES], df[config.TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )

    results, fitted = {}, {}
    for name, model in candidate_models().items():
        log.info("Training %s ...", name)
        pipe = build_pipeline(model).fit(X_train, y_train)
        results[name] = evaluate(y_test, pipe.predict(X_test))
        fitted[name] = pipe
        log.info("%s: %s", name, results[name])

    best = min(results, key=lambda n: results[n]["mae"])
    log.info("Best model: %s", best)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted[best], config.MODEL_PATH)

    metrics = {
        "best_model": best,
        "results": results,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "snapshot_date": config.SNAPSHOT_DATE,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    # Values the web form needs (dropdown options and sensible defaults).
    neighbourhood_coords = (
        df.groupby("neighbourhood_cleansed")[["latitude", "longitude"]].median().round(5)
    )
    metadata = {
        "neighbourhoods": sorted(df["neighbourhood_cleansed"].dropna().unique().tolist()),
        "room_types": sorted(df["room_type"].dropna().unique().tolist()),
        "property_groups": sorted(df["property_group"].dropna().unique().tolist()),
        "neighbourhood_coords": neighbourhood_coords.to_dict(orient="index"),
        "defaults": {c: float(df[c].median()) for c in config.NUMERIC_FEATURES},
    }
    config.METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    return metrics
