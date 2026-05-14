# 🏠 California Housing Price Prediction API

A machine learning model deployed as a production-ready REST API using **FastAPI** and **Docker**.

> > **Live Demo:** [housing-price-api-9tt8.onrender.com/docs](https://housing-price-api-9tt8.onrender.com/docs) 
> *(Free tier — may take ~30 seconds to wake on first visit)*

---

## What This Project Does

This API predicts the **median house value** of a California neighborhood based on
five census features: median income, house age, average rooms, average bedrooms,
and population. It uses a Random Forest model trained on the California Housing
dataset (20,640 neighborhoods from the 1990 US Census).

Send it a neighborhood's characteristics → get back a predicted house value.

---

## Why I Built This

Most ML tutorials end at model training. The real-world skill that employers want
is **deployment** — making a model callable by any application, anywhere, over the
internet. This project demonstrates the full pipeline:

**Train → Serialize → Wrap in API → Containerize → Deploy**

---


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

### `GET /`
Health check — confirms the API is running.

**Response:**
```json
{
  "message": "Welcome to the Housing Price Prediction API!",
  "docs": "Visit /docs for interactive API documentation"
}
```

### `POST /predict`
Accepts neighborhood features and returns a predicted median house value.

**Request body:**
```json
{
  "MedInc": 8.3252,
  "HouseAge": 41.0,
  "AveRooms": 6.9841,
  "AveBedrms": 1.0238,
  "Population": 322.0
}
```

**Response:**
```json
{
  "predicted_median_house_value_usd": 220000.0,
  "predicted_value_raw": 2.2,
  "note": "Raw value is in units of $100,000 (standard for this dataset)"
}
```

---

## Run Locally with Docker

```bash
git clone https://github.com/josephwelbeck/housing-price-api.git
cd housing-price-api
docker build -t housing-price-api .
docker run -p 8000:8000 housing-price-api
# Visit http://127.0.0.1:8000/docs
```

## Run Locally with Python

```bash
git clone https://github.com/josephwelbeck/housing-price-api.git
cd housing-price-api
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python train.py
uvicorn main:app --reload
# Visit http://127.0.0.1:8000/docs
```

---

## Project Structure

---

## What I Learned

- How REST APIs work and why they're the standard interface for ML models
- How FastAPI and Pydantic create typed, self-documenting API endpoints
- How to serialize and load ML models with joblib
- How Docker makes ML services portable and reproducible
- How to deploy a containerized service to a cloud platform

---

