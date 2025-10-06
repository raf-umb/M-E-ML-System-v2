import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

PROCESSED_PATH = "data/processed/mne_data.csv"
MODEL_PATH = "models/mne_success_model.pkl"

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

df = pd.read_csv(PROCESSED_PATH)

# Features and target
# Use simple numeric features for demo. Add more engineering for real projects.
features = ["budget", "staff_count", "region_risk_index", "activity_completion_rate"]
for f in features:
    if f not in df.columns:
        df[f] = 0

X = df[features]
y = df["project_success"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, MODEL_PATH)
print("Model trained and saved to", MODEL_PATH)

# Also save a small "model info" file
with open("models/model_info.json", "w") as fh:
    import json
    json.dump({"features": features, "model_type": "RandomForestClassifier"}, fh)
