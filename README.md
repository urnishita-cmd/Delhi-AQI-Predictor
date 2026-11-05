# Delhi Next-Day AQI Predictor

A machine learning project that predicts the next day's city-level Air Quality Index (AQI) for Delhi using Random Forest Regression.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

## Methodology
- Historical daily AQI observations
- Lag features (1, 2, 3 and 7 days)
- Rolling averages (3 and 7 days)
- Calendar-based features
- Chronological 80:20 train-test split
- Random Forest Regression

## Model Evaluation

| Metric | Result |
|---|---:|
| MAE | 30.532 |
| RMSE | 41.305 |
| R² | 0.848 |
| Persistence Baseline MAE | 31.606 |

## Dataset
Historical daily city-level AQI observations for Delhi, covering 1 May 2015 to 31 December 2023.

Place the appropriately licensed dataset at dataset/air_quality.csv.

Required columns: date and Index Value.

## Installation
Install dependencies using: python3 -m pip install -r requirements.txt

## Run the Project
Execute: python air_quality_predictor.py

## Outputs
- Model metrics CSV
- Prediction results CSV
- Actual vs predicted AQI graph
- Feature importance graph

## Disclaimer
This is an educational machine learning project. Predictions are estimates and should not replace official air quality readings or public health advisories.
