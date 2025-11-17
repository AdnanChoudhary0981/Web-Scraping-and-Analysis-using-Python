
#!/usr/bin/env python3
from flask import Flask, request, jsonify
import joblib, json
import pandas as pd

app = Flask(__name__)
PREPRO_PATH = 'artifacts/preprocessor.joblib'
MODEL_PATH = 'artifacts/best_model.joblib'
FEATURES_PATH = 'artifacts/feature_names.json'

preprocessor = joblib.load(PREPRO_PATH)
model = joblib.load(MODEL_PATH)
feature_names = json.load(open(FEATURES_PATH, 'r'))

def preprocess_input(payload):
    df = pd.DataFrame([payload])
    df['date_listed'] = pd.to_datetime(df['date_listed'])
    df['days_since_listed'] = (pd.Timestamp.now().normalize() - df['date_listed']).dt.days
    def extract_weight(s):
        try:
            obj = json.loads(s)
            return obj.get('weight_g', None)
        except Exception:
            return None
    df['weight_g'] = df['specs'].apply(extract_weight)
    df['title_len'] = df['title'].str.len()
    df['desc_len'] = df['description'].str.len()
    X = df[['category','price','rating','num_reviews','availability','seller','days_since_listed','weight_g','title_len','desc_len']]
    X_p = preprocessor.transform(X)
    return X_p, X

@app.route('/predict', methods=['POST'])
def predict():
    payload = request.json
    X_p, X_raw = preprocess_input(payload)
    proba = float(model.predict_proba(X_p)[0,1]) if hasattr(model, 'predict_proba') else float(model.decision_function(X_p)[0])
    pred = int(model.predict(X_p)[0])
    return jsonify({'prediction': pred, 'probability': proba, 'input': X_raw.to_dict(orient='records')[0]})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
