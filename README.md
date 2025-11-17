# Web Scraping & Analysis Project

**Contents:** This repository contains a complete, runnable web scraping -> data preparation -> modeling -> explanation pipeline.
It uses a mock dataset generator to simulate scraped product listings and includes scripts to run EDA, preprocessing, modeling, and a Flask prediction endpoint.

**Structure:**
- data/mock_data.csv - Generated mock dataset (simulated scraped data).
- scripts/generate_mock_data.py - Script to generate the CSV dataset.
- scripts/scraper_stub.py - BeautifulSoup-based scraping template (adapt selectors to your target site).
- scripts/preprocessing.py - Preprocessing pipeline: feature engineering, transformers, and save artifacts.
- scripts/train_models.py - Train models (Logistic Regression, RandomForest, GradientBoosting) and save best model.
- scripts/evaluate.py - Evaluate saved models and plot metrics.
- flask_app.py - Minimal Flask app to serve predictions (loads preprocessor and model).
- notebooks/EDA_and_Modeling_instructions.md - Quick instructions to run analyses or open as notebook.
- requirements.txt - Python dependencies.

**How to run (locally):**
1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate      # Linux/Mac
   venv\Scripts\activate       # Windows
   pip install -r requirements.txt
   ```
2. Generate mock data and inspect it:
   ```bash
   python scripts/generate_mock_data.py
   head data/mock_data.csv
   ```
3. Preprocess and train models:
   ```bash
   python scripts/preprocessing.py
   python scripts/train_models.py
   ```
4. Evaluate:
   ```bash
   python scripts/evaluate.py
   ```
5. Serve predictions with Flask (example):
   ```bash
   python flask_app.py
   ```

**Notes:**
- Replace scraper_stub.py content with real selectors for your target site. Respect robots.txt and site TOS.
- Artifacts saved by scripts: artifacts/preprocessor.joblib, artifacts/best_model.joblib, artifacts/feature_names.json.

Generated on: 2025-11-14T15:05:28.655451 UTC
