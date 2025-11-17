
#!/usr/bin/env python3
"""Train models and save the best performing model (by ROC AUC on a hold-out)."""
import json
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from preprocessing import extract_weight, build_preprocessor

DATA_PATH = Path('data/mock_data.csv')
ARTIFACT_DIR = Path('artifacts')
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    df = pd.read_csv(DATA_PATH)
    df['date_listed'] = pd.to_datetime(df['date_listed'])
    df['days_since_listed'] = (pd.Timestamp.now().normalize() - df['date_listed']).dt.days
    df['weight_g'] = df['specs'].apply(extract_weight)
    df['title_len'] = df['title'].str.len()
    df['desc_len'] = df['description'].str.len()
    feature_cols = ['category','price','rating','num_reviews','availability','seller','days_since_listed','weight_g','title_len','desc_len']
    X = df[feature_cols]
    y = df['is_popular']
    return X, y

def main():
    X, y = load_data()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    preprocessor, num_cols, cat_cols = build_preprocessor()
    preprocessor.fit(X_train)
    X_train_p = preprocessor.transform(X_train)
    X_val_p = preprocessor.transform(X_val)

    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42)
    }

    best_model = None
    best_auc = -1
    for name, model in models.items():
        print(f'Training {name}...')
        model.fit(X_train_p, y_train)
        y_proba = model.predict_proba(X_val_p)[:,1] if hasattr(model, 'predict_proba') else model.decision_function(X_val_p)
        auc = roc_auc_score(y_val, y_proba)
        print(f'{name} ROC AUC = {auc:.4f}')
        if auc > best_auc:
            best_auc = auc
            best_model = (name, model)

    joblib.dump(preprocessor, ARTIFACT_DIR / 'preprocessor.joblib')
    joblib.dump(best_model[1], ARTIFACT_DIR / 'best_model.joblib')
    print(f'Saved best model: {best_model[0]} with AUC {best_auc:.4f}')

    ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
    cat_cols = ['category','availability','seller']
    feature_names = num_cols = ['price','rating','num_reviews','days_since_listed','weight_g','title_len','desc_len'] + list(ohe.get_feature_names_out(cat_cols))
    (ARTIFACT_DIR / 'feature_names.json').write_text(json.dumps(feature_names))

if __name__ == '__main__':
    main()
