from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(title="California Housing Price Prediction API")

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

@app.get("/", response_class=HTMLResponse)
def read_root():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>California Housing Price Predictor</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --cream: #F5F0E8;
    --warm-white: #FDFAF5;
    --charcoal: #1C1C1E;
    --brown: #8B6914;
    --gold: #C9952A;
    --gold-light: #F0D080;
    --muted: #6B6560;
    --border: #E2D9CC;
    --card: #FFFFFF;
    --shadow: 0 2px 20px rgba(28,28,30,0.08);
    --shadow-lg: 0 8px 40px rgba(28,28,30,0.12);
  }

  body {
    font-family: 'DM Sans', sans-serif;
    background: var(--cream);
    color: var(--charcoal);
    min-height: 100vh;
  }

  .hero {
    background: var(--charcoal);
    color: var(--cream);
    padding: 64px 24px 80px;
    text-align: center;
    position: relative;
    overflow: hidden;
  }
  .hero::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse at 60% 0%, rgba(201,149,42,0.15) 0%, transparent 70%);
    pointer-events: none;
  }
  .hero-tag {
    display: inline-block;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--gold);
    border: 1px solid rgba(201,149,42,0.4);
    padding: 6px 14px;
    border-radius: 20px;
    margin-bottom: 24px;
  }
  .hero h1 {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(2.2rem, 5vw, 3.8rem);
    line-height: 1.1;
    margin-bottom: 16px;
    color: var(--warm-white);
  }
  .hero h1 em {
    font-style: italic;
    color: var(--gold-light);
  }
  .hero p {
    font-size: 1.05rem;
    color: rgba(245,240,232,0.7);
    max-width: 500px;
    margin: 0 auto;
    line-height: 1.6;
    font-weight: 300;
  }

  .how-strip {
    background: var(--gold);
    padding: 14px 24px;
    display: flex;
    justify-content: center;
    gap: 32px;
    flex-wrap: wrap;
  }
  .how-item {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--charcoal);
    letter-spacing: 0.04em;
    text-transform: uppercase;
  }

  .main {
    max-width: 900px;
    margin: 0 auto;
    padding: 48px 24px 80px;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
    align-items: start;
  }
  @media (max-width: 680px) { .main { grid-template-columns: 1fr; } }

  .card {
    background: var(--card);
    border-radius: 16px;
    border: 1px solid var(--border);
    box-shadow: var(--shadow);
    overflow: hidden;
  }
  .card-header { padding: 20px 24px 0; }
  .card-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.3rem;
    color: var(--charcoal);
    margin-bottom: 4px;
  }
  .card-subtitle {
    font-size: 0.82rem;
    color: var(--muted);
    margin-bottom: 20px;
  }
  .card-body { padding: 0 24px 24px; }

  .field { margin-bottom: 22px; }
  .field label {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--charcoal);
    letter-spacing: 0.03em;
    text-transform: uppercase;
    margin-bottom: 10px;
  }
  .field label .hint {
    font-size: 0.78rem;
    font-weight: 500;
    color: var(--brown);
    text-transform: none;
    letter-spacing: 0;
    background: rgba(201,149,42,0.1);
    padding: 2px 8px;
    border-radius: 10px;
  }
  .field input[type="range"] {
    width: 100%;
    height: 4px;
    -webkit-appearance: none;
    background: var(--border);
    border-radius: 2px;
    outline: none;
    cursor: pointer;
    margin-bottom: 6px;
  }
  .field input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 20px;
    height: 20px;
    border-radius: 50%;
    background: var(--charcoal);
    border: 3px solid var(--gold);
    cursor: pointer;
    transition: transform 0.15s;
  }
  .field input[type="range"]::-webkit-slider-thumb:hover { transform: scale(1.2); }
  .range-labels {
    display: flex;
    justify-content: space-between;
    font-size: 0.7rem;
    color: var(--muted);
    margin-top: 2px;
  }

  .divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 8px 0 20px;
  }

  .predict-btn {
    width: 100%;
    padding: 16px;
    background: var(--charcoal);
    color: var(--cream);
    border: none;
    border-radius: 10px;
    font-family: 'DM Sans', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    cursor: pointer;
    letter-spacing: 0.04em;
    transition: background 0.2s, transform 0.1s;
  }
  .predict-btn:hover { background: #2c2c2e; transform: translateY(-1px); }
  .predict-btn:active { transform: translateY(0); }
  .predict-btn.loading { opacity: 0.7; pointer-events: none; }

  .result-card {
    background: var(--card);
    border-radius: 16px;
    border: 1px solid var(--border);
    box-shadow: var(--shadow);
    padding: 32px 24px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 300px;
    text-align: center;
  }
  .result-idle .idle-icon { font-size: 2.5rem; margin-bottom: 16px; opacity: 0.4; }
  .result-idle p { font-size: 0.9rem; color: var(--muted); line-height: 1.5; max-width: 200px; }

  .result-loading .spinner {
    width: 40px; height: 40px;
    border: 3px solid var(--border);
    border-top-color: var(--gold);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    margin-bottom: 16px;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  .result-loading p { font-size: 0.9rem; color: var(--muted); }

  .result-success { animation: fadeIn 0.4s ease; }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }

  .result-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 8px;
  }
  .result-price {
    font-family: 'DM Serif Display', serif;
    font-size: 3rem;
    color: var(--charcoal);
    line-height: 1;
    margin-bottom: 6px;
  }
  .result-price-sub {
    font-size: 0.82rem;
    color: var(--muted);
    margin-bottom: 24px;
  }
  .result-bar {
    width: 100%;
    height: 6px;
    background: var(--border);
    border-radius: 3px;
    overflow: hidden;
    margin-bottom: 20px;
  }
  .result-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--gold), var(--brown));
    border-radius: 3px;
    transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
  }
  .result-context {
    font-size: 0.82rem;
    color: var(--muted);
    line-height: 1.5;
    padding: 12px 16px;
    background: var(--cream);
    border-radius: 8px;
    width: 100%;
    text-align: left;
  }

  .footer-note {
    text-align: center;
    padding: 0 24px 48px;
    font-size: 0.78rem;
    color: var(--muted);
    max-width: 600px;
    margin: 0 auto;
    line-height: 1.6;
  }
  .footer-note a { color: var(--brown); text-decoration: none; font-weight: 500; }
  .footer-note a:hover { text-decoration: underline; }
</style>
</head>
<body>

<div class="hero">
  <div class="hero-tag">ML Model — Live API</div>
  <h1>California Housing<br><em>Price Predictor</em></h1>
  <p>Enter neighborhood details below and get an instant machine learning prediction of median house value.</p>
</div>

<div class="how-strip">
  <div class="how-item"><span>①</span> Set neighborhood details</div>
  <div class="how-item"><span>②</span> Click predict</div>
  <div class="how-item"><span>③</span> Get instant estimate</div>
</div>

<div class="main">
  <div class="card">
    <div class="card-header">
      <div class="card-title">Neighborhood Details</div>
      <div class="card-subtitle">Adjust the sliders to match a neighborhood</div>
    </div>
    <div class="card-body">

      <div class="field">
        <label>Median Household Income <span class="hint" id="inc-hint">$83,252 / yr</span></label>
        <input type="range" id="MedInc" min="0.5" max="15" step="0.1" value="8.3252"
          oninput="updateHint('MedInc', 'inc-hint', v => '$' + Math.round(v * 10000).toLocaleString() + ' / yr')">
        <div class="range-labels"><span>$5,000</span><span>$150,000</span></div>
      </div>

      <div class="field">
        <label>Median House Age <span class="hint" id="age-hint">41 years old</span></label>
        <input type="range" id="HouseAge" min="1" max="52" step="1" value="41"
          oninput="updateHint('HouseAge', 'age-hint', v => Math.round(v) + ' years old')">
        <div class="range-labels"><span>1 year</span><span>52 years</span></div>
      </div>

      <div class="field">
        <label>Neighborhood Population <span class="hint" id="pop-hint">322 people</span></label>
        <input type="range" id="Population" min="50" max="3000" step="10" value="322"
          oninput="updateHint('Population', 'pop-hint', v => Math.round(v).toLocaleString() + ' people')">
        <div class="range-labels"><span>50 people</span><span>3,000 people</span></div>
      </div>

      <!-- Hidden inputs: required by the model, fixed at dataset averages -->
      <input type="hidden" id="AveRooms" value="6.9841">
      <input type="hidden" id="AveBedrms" value="1.0238">

      <hr class="divider">
      <button class="predict-btn" onclick="predict()" id="predict-btn">
        Get Price Estimate →
      </button>
    </div>
  </div>

  <div class="result-card result-idle" id="result-card">
    <div class="idle-icon">🏡</div>
    <p>Set the neighborhood details and click predict to see an estimated house value.</p>
  </div>
</div>

<div class="footer-note">
  Predictions are based on the <strong>1990 California Census</strong> dataset and are for
  demonstration purposes only. This is a portfolio project showing ML model deployment
  using FastAPI and Docker. <a href="/docs">View the API docs →</a>
</div>

<script>
  function updateHint(sliderId, hintId, formatter) {
    const val = document.getElementById(sliderId).value;
    document.getElementById(hintId).textContent = formatter(val);
  }

  async function predict() {
    const btn = document.getElementById('predict-btn');
    const card = document.getElementById('result-card');

    btn.classList.add('loading');
    btn.textContent = 'Predicting...';
    card.className = 'result-card result-loading';
    card.innerHTML = '<div class="spinner"></div><p>Running the model...</p>';

    const payload = {
      MedInc: parseFloat(document.getElementById('MedInc').value),
      HouseAge: parseFloat(document.getElementById('HouseAge').value),
      AveRooms: parseFloat(document.getElementById('AveRooms').value),
      AveBedrms: parseFloat(document.getElementById('AveBedrms').value),
      Population: parseFloat(document.getElementById('Population').value)
    };

    try {
      const res = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      const price = data.predicted_median_house_value_usd;
      const formatted = '$' + Math.round(price).toLocaleString();
      const pct = Math.min((price / 800000) * 100, 100).toFixed(1);

      let context = '';
      if (price < 100000) context = 'Well below average — likely reflects a low-income neighborhood with older housing.';
      else if (price < 200000) context = 'Below average for California. Modest income levels or an older neighborhood.';
      else if (price < 350000) context = 'Around the dataset median — a typical mid-range California neighborhood.';
      else if (price < 550000) context = 'Above average. Higher incomes or desirable neighborhood characteristics.';
      else context = 'High-value neighborhood. Strong income levels and favorable housing characteristics.';

      card.className = 'result-card result-success';
      card.innerHTML = `
        <div class="result-label">Estimated Median House Value</div>
        <div class="result-price">${formatted}</div>
        <div class="result-price-sub">Based on 1990 California Census data</div>
        <div class="result-bar"><div class="result-bar-fill" style="width: ${pct}%"></div></div>
        <div class="result-context">${context}</div>
      `;
    } catch (e) {
      card.className = 'result-card result-idle';
      card.innerHTML = '<p style="color:#c0392b;font-size:0.9rem">Something went wrong. Please try again.</p>';
    }

    btn.classList.remove('loading');
    btn.textContent = 'Get Price Estimate →';
  }
</script>
</body>
</html>
"""

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