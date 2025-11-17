
#!/usr/bin/env python3
"""Preprocessing script: feature engineering + ColumnTransformer pipeline save."""
import json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib

DATA_PATH = Path('data/mock_data.csv')
ARTIFACT_DIR = Path('artifacts')
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

def extract_weight(specs_json_str):
    try:
        specs = json.loads(specs_json_str)
        return specs.get('weight_g', np.nan)
    except Exception:
        return np.nan

def build_preprocessor():
    numeric_features = ['price', 'rating', 'num_reviews', 'days_since_listed', 'weight_g', 'title_len', 'desc_len']
    categorical_features = ['category', 'availability', 'seller']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
    ], remainder='drop')

    return preprocessor, numeric_features, categorical_features

def run_preprocessing():
    df = pd.read_csv(DATA_PATH)
    df['date_listed'] = pd.to_datetime(df['date_listed'])
    df['days_since_listed'] = (pd.Timestamp.now().normalize() - df['date_listed']).dt.days
    df['weight_g'] = df['specs'].apply(extract_weight)
    df['title_len'] = df['title'].str.len()
    df['desc_len'] = df['description'].str.len()

    feature_cols = ['category','price','rating','num_reviews','availability','seller','days_since_listed','weight_g','title_len','desc_len']
    X = df[feature_cols]
    y = df['is_popular']

    preprocessor, num_cols, cat_cols = build_preprocessor()
    preprocessor.fit(X)

    # Save artifacts
    joblib.dump(preprocessor, ARTIFACT_DIR / 'preprocessor.joblib')
    # Save feature names helper (we'll infer after fitting)
    ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
    ohe_cols = list(ohe.get_feature_names_out(cat_cols))
    feature_names = num_cols + ohe_cols
    (ARTIFACT_DIR / 'feature_names.json').write_text(json.dumps(feature_names))
    print(f'Preprocessor saved to {ARTIFACT_DIR / \'preprocessor.joblib\'}')
    print(f'Feature names saved to {ARTIFACT_DIR / \'feature_names.json\'}')

if __name__ == '__main__':
    run_preprocessing()
