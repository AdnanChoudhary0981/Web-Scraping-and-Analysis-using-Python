
#!/usr/bin/env python3
"""Evaluate the saved best model on the hold-out test set and plot metrics."""
import joblib, json
from pathlib import Path
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import extract_weight, build_preprocessor

DATA_PATH = Path('data/mock_data.csv')
ARTIFACT_DIR = Path('artifacts')

def load_test():
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
    X, y = load_test()
    preprocessor = joblib.load(ARTIFACT_DIR / 'preprocessor.joblib')
    model = joblib.load(ARTIFACT_DIR / 'best_model.joblib')

    X_p = preprocessor.transform(X)
    y_pred = model.predict(X_p)
    y_proba = model.predict_proba(X_p)[:,1] if hasattr(model, 'predict_proba') else model.decision_function(X_p)

    print('Classification Report:')
    from sklearn.metrics import classification_report
    print(classification_report(y, y_pred, digits=4))
    cm = confusion_matrix(y, y_pred)
    print('Confusion Matrix:\n', cm)
    try:
        auc = roc_auc_score(y, y_proba)
        print(f'ROC AUC = {auc:.4f}')
    except Exception as e:
        print('ROC AUC not available:', e)

    fpr, tpr, _ = roc_curve(y, y_proba)
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
    plt.plot([0,1],[0,1],'--',color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (All Data)')
    plt.legend()
    plt.grid(True)
    plt.savefig('artifacts/roc_curve.png')
    print('ROC curve saved to artifacts/roc_curve.png')

if __name__ == '__main__':
    main()
