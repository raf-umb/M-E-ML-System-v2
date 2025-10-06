import pandas as pd
import os

RAW_PATH = "data/raw/mne_data.csv"
PROCESSED_PATH = "data/processed/mne_data.csv"

os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)

print("Loading raw data:", RAW_PATH)
df = pd.read_csv(RAW_PATH)

# Simple cleaning example
print("Initial rows:", len(df))
df = df.dropna(subset=["budget", "staff_count", "activity_completion_rate"])
df["region_risk_index"] = df["region_risk_index"].fillna(df["region_risk_index"].mean())

# Ensure numeric types
df["budget"] = pd.to_numeric(df["budget"], errors="coerce").fillna(0)
df["staff_count"] = pd.to_numeric(df["staff_count"], errors="coerce").fillna(0)
df["activity_completion_rate"] = pd.to_numeric(df["activity_completion_rate"], errors="coerce").fillna(0)
df["region_risk_index"] = pd.to_numeric(df["region_risk_index"], errors="coerce").fillna(0)

# If latitude/longitude missing, you can add geocoding here (example commented)
# from geopy.geocoders import GoogleV3
# API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")
# if API_KEY:
#     geolocator = GoogleV3(api_key=API_KEY)
#     def geocode_address(addr):
#         try:
#             loc = geolocator.geocode(addr)
#             return loc.latitude, loc.longitude
#         except:
#             return None, None
#     df[['latitude','longitude']] = df['project_address'].apply(lambda a: pd.Series(geocode_address(a)))

df.to_csv(PROCESSED_PATH, index=False)
print("Processed data saved to", PROCESSED_PATH)
