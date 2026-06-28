# Reads the two source files (dataset.json and Nintendo_Cooccurrence_Matrix.xlsx) 
# and normalizes them into a single in-memory structure for the agent's tools

import json
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.parent / "data" 
JSON_PATH = DATA_DIR / "dataset.json"
MATRIX_PATH = DATA_DIR / "Nintendo_Cooccurrence_Matrix.xlsx"


def load_products(json_path: Path = JSON_PATH) -> dict:
    # Read dataset.json
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Swap category with type when needed
    from validations import fix_category_type_swap
    raw, _ = fix_category_type_swap(raw, strict=True)

    # Add each product's URL from product_urls.json
    urls_path = DATA_DIR / "product_urls.json"
    try:
        with open(urls_path, "r", encoding="utf-8") as f:
            product_urls = json.load(f)
    except FileNotFoundError:
        product_urls = {}

    products = {}
    for category_key, items in raw.items():
        for item in items:
            name = item["name"]
            if name in products:
                raise ValueError(f"Produto duplicado entre categorias: {name}")

            products[name] = {
                "name": name,
                "category": item.get("category"),
                "type": item.get("type"),
                "franchise": item.get("franchise"),       # None for Console/Accessories
                "min_age": item.get("min_age"),            # None for Console/Accessories
                "release_date": item.get("release_date"),  # None for Console/Accessories
                "times_sold": item.get("times_sold"),
                "url": product_urls.get(name),             # None if no URL
                "store_breakdown": {
                    "Store_A": item.get("Store A"),
                    "Store_B": item.get("Store B"),
                    "Store_C": item.get("Store C"),
                },
            }

    return products


def load_cooccurrence_matrix(matrix_path: Path = MATRIX_PATH) -> pd.DataFrame:
    df = pd.read_excel(matrix_path, index_col=0)
    return df


def validate_consistency(products: dict, matrix: pd.DataFrame) -> None:
    # Check that product names in the JSON and the matrix match exactly

    json_names = set(products.keys())
    matrix_names = set(matrix.index)

    missing_in_matrix = json_names - matrix_names
    missing_in_json = matrix_names - json_names

    if missing_in_matrix:
        raise ValueError(
            f"Produtos no JSON mas ausentes na matriz: {missing_in_matrix}"
        )
    if missing_in_json:
        raise ValueError(
            f"Produtos na matriz mas ausentes no JSON: {missing_in_json}"
        )
    if list(matrix.index) != list(matrix.columns):
        raise ValueError("A matriz de co-ocorrência não é simétrica em índice/colunas")


def load_all() -> tuple[dict, pd.DataFrame]:
    products = load_products()
    matrix = load_cooccurrence_matrix()
    validate_consistency(products, matrix)
    return products, matrix


if __name__ == "__main__":
    # Quick data_loader test
    products, matrix = load_all()
    print(f"{len(products)} produtos carregados.")
    print(f"Matriz: {matrix.shape[0]}x{matrix.shape[1]}")
    print("Exemplo de produto:", products["Super Mario Odyssey"])
    print(
        "Co-ocorrência Mario Odyssey x Mario Kart 8 Deluxe:",
        matrix.loc["Super Mario Odyssey", "Mario Kart 8 Deluxe"],
    )
