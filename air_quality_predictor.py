"""
Delhi Next-Day AQI Predictor
Class 12 Artificial Intelligence Project

Input:
    dataset/air_quality.csv

Expected columns:
    date
    Index Value

The model predicts AQI for a date using only AQI observations from earlier
calendar dates, plus calendar features for the prediction date.
"""

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore", category=FutureWarning)

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "air_quality.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
GRAPH_DIR = OUTPUT_DIR / "graphs"

RANDOM_STATE = 42
TRAIN_FRACTION = 0.80


def load_and_validate_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Put your CSV file at dataset/air_quality.csv."
        )

    df = pd.read_csv(path)
    required = {"date", "Index Value"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"Missing required column(s): {', '.join(sorted(missing))}. "
            f"Found columns: {', '.join(map(str, df.columns))}"
        )

    df = df[["date", "Index Value"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["Index Value"] = pd.to_numeric(df["Index Value"], errors="coerce")
    df = df.dropna(subset=["date", "Index Value"])
    df["date"] = df["date"].dt.normalize()
    df = df.sort_values("date")

    # If a date occurs more than once, stop rather than silently choosing
    # one AQI value or inventing an aggregation rule.
    duplicates = df["date"].duplicated(keep=False)
    if duplicates.any():
        examples = df.loc[duplicates, "date"].dt.strftime("%Y-%m-%d").unique()[:5]
        raise ValueError(
            "Duplicate dates were found. Review the source data before training. "
            f"Examples: {', '.join(examples)}"
        )

    if (df["Index Value"] < 0).any():
        raise ValueError("Negative AQI values were found. Review the dataset.")

    df = df.set_index("date").sort_index()
    if len(df) < 100:
        raise ValueError("The dataset has too few valid observations to train reliably.")

    return df


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    # Reindex to the calendar so a missing date remains missing. This makes
    # lag_1 mean the previous calendar day's AQI, not merely the prior row.
    full_dates = pd.date_range(df.index.min(), df.index.max(), freq="D")
    daily = df.reindex(full_dates)
    daily.index.name = "date"
    daily["aqi"] = daily["Index Value"]

    features = pd.DataFrame(index=daily.index)

    # Exact previous-calendar-day lags.
    for lag in (1, 2, 3, 7):
        features[f"aqi_lag_{lag}"] = daily["aqi"].shift(lag)

    # Rolling averages over the previous calendar days only; the current
    # day's target is excluded. Require all days in each window to exist.
    previous_aqi = daily["aqi"].shift(1)
    features["aqi_rolling_mean_3d"] = previous_aqi.rolling(
        window=3, min_periods=3
    ).mean()
    features["aqi_rolling_mean_7d"] = previous_aqi.rolling(
        window=7, min_periods=7
    ).mean()

    # Calendar features for the date being predicted.
    features["day_of_week"] = features.index.dayofweek
    features["month"] = features.index.month
    features["day_of_year"] = features.index.dayofyear

    # Target is AQI on the feature date. All AQI predictors above are from
    # earlier dates, so the target date's AQI is never used as an input.
    features["actual_aqi"] = daily["aqi"]
    features["baseline_prediction"] = daily["aqi"].shift(1)

    features = features.dropna()
    return features


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_and_validate_data(DATASET_PATH)
    prepared = make_features(raw)

    if len(prepared) < 30:
        raise ValueError(
            "Too few rows remain after creating historical features. "
            "Check the date coverage and missing dates."
        )

    feature_columns = [
        "aqi_lag_1",
        "aqi_lag_2",
        "aqi_lag_3",
        "aqi_lag_7",
        "aqi_rolling_mean_3d",
        "aqi_rolling_mean_7d",
        "day_of_week",
        "month",
        "day_of_year",
    ]

    split_index = int(len(prepared) * TRAIN_FRACTION)
    train = prepared.iloc[:split_index]
    test = prepared.iloc[split_index:]

    X_train = train[feature_columns]
    y_train = train["actual_aqi"]
    X_test = test[feature_columns]
    y_test = test["actual_aqi"]

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=RANDOM_STATE,
        min_samples_leaf=2,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    baseline_predictions = test["baseline_prediction"].to_numpy()

    model_mae = mean_absolute_error(y_test, predictions)
    model_rmse = np.sqrt(mean_squared_error(y_test, predictions))
    model_r2 = r2_score(y_test, predictions)
    baseline_mae = mean_absolute_error(y_test, baseline_predictions)

    metrics = pd.DataFrame(
        [
            {"Metric": "MAE", "Value": model_mae},
            {"Metric": "RMSE", "Value": model_rmse},
            {"Metric": "R2", "Value": model_r2},
            {"Metric": "Persistence Baseline MAE", "Value": baseline_mae},
            {"Metric": "Training Observations", "Value": len(train)},
            {"Metric": "Testing Observations", "Value": len(test)},
            {"Metric": "First Test Date", "Value": test.index.min().strftime("%Y-%m-%d")},
            {"Metric": "Last Test Date", "Value": test.index.max().strftime("%Y-%m-%d")},
            {"Metric": "Valid Model Rows", "Value": len(prepared)},
        ]
    )
    metrics.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)

    results = pd.DataFrame(
        {
            "date": test.index.strftime("%Y-%m-%d"),
            "actual_aqi": y_test.to_numpy(),
            "predicted_aqi": predictions,
            "persistence_baseline_aqi": baseline_predictions,
        }
    )
    results.to_csv(OUTPUT_DIR / "prediction_results.csv", index=False)

    # Actual vs predicted graph.
    plt.figure(figsize=(12, 5))
    plt.plot(test.index, y_test, label="Actual AQI", linewidth=1.5)
    plt.plot(test.index, predictions, label="Predicted AQI", linewidth=1.2)
    plt.title("Delhi AQI: Actual vs Predicted (Test Period)")
    plt.xlabel("Date")
    plt.ylabel("AQI")
    plt.legend()
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "actual_vs_predicted_aqi.png", dpi=200)
    plt.close()

    # Feature importance graph.
    importance = pd.Series(
        model.feature_importances_, index=feature_columns
    ).sort_values()
    plt.figure(figsize=(9, 5))
    importance.plot(kind="barh")
    plt.title("Random Forest Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "feature_importance.png", dpi=200)
    plt.close()

    print("\nDELHI NEXT-DAY AQI PREDICTION — RESULTS")
    print("--------------------------------------")
    print(f"Training observations: {len(train)}")
    print(f"Testing observations:  {len(test)}")
    print(f"Test period: {test.index.min():%Y-%m-%d} to {test.index.max():%Y-%m-%d}")
    print(f"MAE:                     {model_mae:.3f}")
    print(f"RMSE:                    {model_rmse:.3f}")
    print(f"R²:                      {model_r2:.3f}")
    print(f"Persistence baseline MAE:{baseline_mae: .3f}")
    if model_mae < baseline_mae:
        print("Baseline comparison: Random Forest has a lower test MAE.")
    elif model_mae > baseline_mae:
        print("Baseline comparison: Persistence baseline has a lower test MAE.")
    else:
        print("Baseline comparison: Both methods have equal test MAE.")

    print(f"\nFiles saved in: {OUTPUT_DIR}")
    print("No missing AQI dates were filled with invented values.")


if __name__ == "__main__":
    main()
