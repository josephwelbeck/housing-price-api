from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(title="California Housing Price Prediction API")

# --- Auto-train if model file doesn't exist ---
# This runs when the container starts on a fresh server (like Render)
# where the .joblib file doesn't exist yet.
MODEL_PATH = 'california_housing_model.joblib'

if not os.path.exists(MODEL_PATH):
    print("Model file not found. Training model now...")
    from sklearn.datasets import fetch_california_housing
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split

    housing = fetch_california_housing()
    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    y = pd.Series(housing.target, name='MedHouseVal')

    features_to_use = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population']
    X = X[features_to_use]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)
    print("Model trained and saved.")

model = joblib.load(MODEL_PATH)
print("Model loaded. API is ready.")

class HouseFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float

    class Config:
        json_schema_extra = {
            "example": {
                "MedInc": 8.3252,
                "HouseAge": 41.0,
                "AveRooms": 6.9841,
                "AveBedrms": 1.0238,
                "Population": 322.0
            }
        }

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Housing Price Prediction API!",
        "docs": "Visit /docs for interactive API documentation"
    }

@app.post("/predict")
def predict_price(features: HouseFeatures):
    input_data = pd.DataFrame([features.dict()])
    prediction = model.predict(input_data)
    predicted_value = float(prediction[0])
    return {
        "predicted_median_house_value_usd": round(predicted_value * 100_000, 2),
        "predicted_value_raw": predicted_value,
        "note": "Raw value is in units of $100,000 (standard for this dataset)"
    }