# V2 is independent of V1 (reads only what history_v2 needs: products and the co-occurrence matrix)

import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"


def load_all():
    with open(DATA_DIR / "dataset.json", encoding="utf-8") as f:
        raw = json.load(f)

    products = {}
    for category_key, items in raw.items():
        for item in items:
            name = item["name"]
            products[name] = {
                "name": name,
                "times_sold": item.get("times_sold"),
            }

    matrix = pd.read_excel(
        DATA_DIR / "Nintendo_Cooccurrence_Matrix.xlsx", index_col=0
    )
    return products, matrix
