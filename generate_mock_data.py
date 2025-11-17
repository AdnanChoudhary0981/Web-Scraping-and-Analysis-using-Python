
#!/usr/bin/env python3
"""Generate a realistic mock dataset simulating scraped product listings."""
import json, random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def generate_mock_data(n=2000, out_csv='data/mock_data.csv'):
    categories = ['laptops', 'headphones', 'smartphones', 'cameras', 'monitors']
    sellers = ['BestStore', 'ElectroHub', 'GadgetWorld', 'Deals4U', 'PrimeSeller']
    titles = [
        'ProBook {} - Ultra Performance',
        'SoundX {} - Noise Cancelling',
        'SmartEdge {} - 5G Ready',
        'PixelCam {} - 4K Capture',
        'ViewPlus {} - Wide Screen',
    ]

    rows = []
    base_date = datetime.now() - timedelta(days=365)
    for i in range(n):
        cat = random.choice(categories)
        model = "Model{}".format(random.randint(100,999))
        title = random.choice(titles).format(model)
        price = round(
            max(30, min(4000, np.random.normal(300 if cat != 'laptops' else 800, 150))),
            2,
        )
        rating = round(max(1.0, min(5.0, np.random.beta(5, 1.5) * 5)), 2)
        num_reviews = int(np.random.exponential(scale=40))
        availability = random.choices(['in_stock', 'out_of_stock'], weights=[0.85, 0.15])[0]
        seller = random.choice(sellers)
        days_ago = random.randint(0, 365)
        date_listed = (base_date + timedelta(days=days_ago)).date().isoformat()
        description = f"{title} - High quality {cat} with features and warranty."
        specs = json.dumps({"weight_g": random.randint(100, 2500), "color": random.choice(['black','white','silver','blue'])})
        product_id = f"PID{i:06d}"
        rows.append({
            'product_id': product_id,
            'title': title,
            'category': cat,
            'price': price,
            'rating': rating,
            'num_reviews': num_reviews,
            'availability': availability,
            'seller': seller,
            'date_listed': date_listed,
            'description': description,
            'specs': specs,
        })

    df = pd.DataFrame(rows)
    df['is_popular'] = ((df['rating'] >= 4.0) & (df['num_reviews'] >= 50)).astype(int)
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Mock data generated at {out_path} with shape {df.shape}")
    return df

if __name__ == '__main__':
    generate_mock_data(2000, out_csv='data/mock_data.csv')
