# M&E Machine Learning System (GitHub Template)

This is a ready-to-upload GitHub repository template for a **Monitoring & Evaluation (M&E)** system that uses
machine learning to predict project success and visualizes project sites on a map using Streamlit.

## What’s included
- Sample dataset (community development projects) with latitude/longitude
- Preprocessing, training, and evaluation scripts (Python)
- Streamlit dashboard with map visualization (`st.map`)
- GitHub Actions workflow to retrain the model on push
- `requirements.txt`

## Quick start
1. Clone the repo or download the ZIP.
2. Create a virtual environment and install dependencies:
```bash
pip install -r requirements.txt
```
3. Preprocess the sample data:
```bash
python src/preprocess.py
```
4. Train the model:
```bash
python src/train_model.py
```
5. Evaluate:
```bash
python src/evaluate.py
```
6. Run the dashboard:
```bash
streamlit run dashboard/app.py
```

Notes:
- The default demo uses a **classification** model to predict project success (1 = success, 0 = fail).
- Map visualization uses Streamlit's built-in `st.map()` and the `latitude`/`longitude` columns.
- To use Google Maps geocoding (address → coordinates) you can supply your API key and adapt `src/preprocess.py`.
