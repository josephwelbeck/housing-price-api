# 🏠 California Housing Price Prediction API

A machine learning model deployed as a production-ready REST API using **FastAPI** and **Docker**, with a user-friendly web interface anyone can use.

> **Live Demo:** [housing-price-api-9tt8.onrender.com](https://housing-price-api-9tt8.onrender.com)
> *(Free tier — may take ~30 seconds to wake on first visit)*

---

## Screenshots

### Interactive Landing Page
![Landing page](screenshots/Housing calc.png)

### Making a Prediction
![Predict request](screenshots/predict-request.png)

### Live Prediction Result
![Predict response](screenshots/predict-response.png)

---

## What This Project Does

This API predicts the **median house value** of a California neighborhood based on census features including median income, house age, and population. It uses a Random Forest model trained on the California Housing dataset (20,640 neighborhoods from the 1990 US Census).

Users interact with a clean web interface — adjust sliders, click predict, get an instant estimate. Developers can use the REST API directly via the /docs endpoint.

Send it a neighborhood's characteristics → get back a predicted house value.

---

## Why I Built This

Most ML tutorials end at model training. The real-world skill that employers want is **deployment** — making a model callable by any application, anywhere, over the internet. This project demonstrates the full pipeline:

**Train → Serialize → Wrap in API → Containerize → Deploy**

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| scikit-learn | Model training (RandomForestRegressor) |
| FastAPI | REST API framework |
| Pydantic | Input validation and schema definition |
| uvicorn | ASGI server |
| Docker | Containerization |
| Render | Cloud deployment |

---

## API Endpoints

### GET /
Serves the interactive web interface.

### GET /docs
Auto-generated developer documentation (Swagger UI).

### POST /predict
Accepts neighborhood features and returns a predicted median house value.

Request body:

    {
      "MedInc": 8.3252,
      "HouseAge": 41.0,
      "AveRooms": 6.9841,
      "AveBedrms": 1.0238,
      "Population": 322.0
    }

Response:

    {
      "predicted_median_house_value_usd": 220000.0,
      "predicted_value_raw": 2.2,
      "note": "Raw value is in units of $100,000 (standard for this dataset)"
    }

---

## Run Locally with Docker

    git clone https://github.com/josephwelbeck/housing-price-api.git
    cd housing-price-api
    docker build -t housing-price-api .
    docker run -p 8000:8000 housing-price-api

## Run Locally with Python

    git clone https://github.com/josephwelbeck/housing-price-api.git
    cd housing-price-api
    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt
    python train.py
    uvicorn main:app --reload

---

## Project Structure

    housing-price-api/
    ├── train.py          # Train and save the model
    ├── main.py           # FastAPI app + HTML frontend
    ├── requirements.txt  # Python dependencies
    ├── Dockerfile        # Container definition
    ├── .dockerignore     # Files excluded from Docker build
    └── screenshots/      # README screenshots

---

## What I Learned

- How REST APIs work and why they're the standard interface for ML models
- How FastAPI and Pydantic create typed, self-documenting API endpoints
- How to serialize and load ML models with joblib
- How Docker makes ML services portable and reproducible
- How to deploy a containerized service to a cloud platform
- How to serve a user-friendly HTML frontend directly from a FastAPI app

