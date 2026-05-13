import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing
import joblib

print("Starting the training script...")

housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = pd.Series(housing.target, name='MedHouseVal')

print("Dataset loaded successfully.")
print(f"Total rows in dataset: {len(X)}")

features_to_use = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population']
X = X[features_to_use]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training rows: {len(X_train)} | Test rows: {len(X_test)}")

model = RandomForestRegressor(n_estimators=100, random_state=42)
print("Training the model... (this takes about 10-30 seconds)")
model.fit(X_train, y_train)
print("Model training complete!")

score = model.score(X_test, y_test)
print(f"Model R2 score on test data: {score:.4f}")

joblib.dump(model, 'california_housing_model.joblib')
print("Model saved as 'california_housing_model.joblib'")
print("Done! You're ready to build the API.")