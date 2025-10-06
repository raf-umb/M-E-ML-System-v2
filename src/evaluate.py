import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import joblib

PROCESSED_PATH = "data/processed/mne_data.csv"
MODEL_PATH = "models/mne_success_model.pkl"

df = pd.read_csv(PROCESSED_PATH)
model = joblib.load(MODEL_PATH)

features = ["budget", "staff_count", "region_risk_index", "activity_completion_rate"]
X = df[features]
y = df["project_success"]

y_pred = model.predict(X)
print("Accuracy:", accuracy_score(y, y_pred))
print(classification_report(y, y_pred))
