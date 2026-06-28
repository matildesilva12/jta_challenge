# Turns dataset.json and Excel matrix into several pandas DataFrames

import numpy as np
import pandas as pd

from data_loader import load_all


def build_dimensional_model():
    products, matrix = load_all()

    # dim_store (PK store_id): store_name
   
    store_names = ["Store_A", "Store_B", "Store_C"]
    dim_store = pd.DataFrame({
        "store_id": range(1, len(store_names) + 1),
        "store_name": store_names,
    })
    store_name_to_id = dict(zip(dim_store["store_name"], dim_store["store_id"]))

    # dim_product (PK product_id): name, category, type, franchise, min_age, release_date
   
    product_rows = []
    for name, p in products.items():
        product_rows.append({
            "name": name,
            "category": p["category"],
            "type": p["type"],
            "franchise": p["franchise"],
            "min_age": p["min_age"],
            "release_date": p["release_date"],
        })
    dim_product = pd.DataFrame(product_rows).sort_values("name").reset_index(drop=True)
    dim_product.insert(0, "product_id", range(1, len(dim_product) + 1))
    product_name_to_id = dict(zip(dim_product["name"], dim_product["product_id"]))

   # fact_sales (FK product_id, FK store_id): units_sold  

    sales_rows = []
    for name, p in products.items():
        pid = product_name_to_id[name]
        for store_name, units in p["store_breakdown"].items():
            if units is not None:
                sales_rows.append({
                    "product_id": pid,
                    "store_id": store_name_to_id[store_name],
                    "units_sold": int(units),
                })
    fact_sales = pd.DataFrame(sales_rows)

    # fact_cooccurrence (FK product_id_a, FK product_id_b): times_together

    cooc_rows = []
    names_in_matrix = list(matrix.index)
    for i, a in enumerate(names_in_matrix):
        for j, b in enumerate(names_in_matrix):
            if j <= i:
                continue  # upper triangle only (avoids duplicating symmetric pairs)
            value = matrix.loc[a, b]
            if value and value > 0:
                cooc_rows.append({
                    "product_id_a": product_name_to_id[a],
                    "product_id_b": product_name_to_id[b],
                    "times_together": int(value),
                })
    fact_cooccurrence = pd.DataFrame(cooc_rows)

    
    standalone = {
        product_name_to_id[n]: int(matrix.loc[n, n]) for n in names_in_matrix
    }
    dim_product["standalone_sales"] = dim_product["product_id"].map(standalone)

    # Type each column explicitly instead of letting pandas infer.
    dim_product = dim_product.astype({
        "product_id": "Int64",
        "name": "string",
        "category": "string",
        "type": "string",
        "franchise": "string",       # <NA> for Console/Accessories
        "min_age": "Int64",          # <NA> for non-games
        "standalone_sales": "Int64",
    })
    # release_date as a real date (NaT where N/A)
    dim_product["release_date"] = pd.to_datetime(
        dim_product["release_date"], errors="coerce"
    )

    dim_store = dim_store.astype({
        "store_id": "Int64",
        "store_name": "string",
    })

    fact_sales = fact_sales.astype({
        "product_id": "Int64",
        "store_id": "Int64",
        "units_sold": "Int64",
    })

    fact_cooccurrence = fact_cooccurrence.astype({
        "product_id_a": "Int64",
        "product_id_b": "Int64",
        "times_together": "Int64",
    })

    return {
        "dim_product": dim_product,
        "dim_store": dim_store,
        "fact_sales": fact_sales,
        "fact_cooccurrence": fact_cooccurrence,
    }


def check_referential_integrity(model: dict) -> list[str]:
    # Check referential integrity and report foreign keys with no match 

    warnings = []
    product_ids = set(model["dim_product"]["product_id"])
    store_ids = set(model["dim_store"]["store_id"])

    fs = model["fact_sales"]
    missing_p = set(fs["product_id"]) - product_ids
    missing_s = set(fs["store_id"]) - store_ids
    if missing_p:
        warnings.append(f"fact_sales: product_id(s) não tem correspondência na dim_product: {missing_p}")
    if missing_s:
        warnings.append(f"fact_sales: store_id(s) não tem correspondência na dim_store: {missing_s}")

    fc = model["fact_cooccurrence"]
    missing_a = set(fc["product_id_a"]) - product_ids
    missing_b = set(fc["product_id_b"]) - product_ids
    if missing_a:
        warnings.append(f"fact_cooccurrence: product_id_a não tem correspondência na dim_product: {missing_a}")
    if missing_b:
        warnings.append(f"fact_cooccurrence: product_id_b não tem correspondência na dim_product: {missing_b}")

    return warnings


if __name__ == "__main__":
    model = build_dimensional_model()
    warnings = check_referential_integrity(model)
    print("Integridade referencial:", "OK" if not warnings else warnings)
    for name, df in model.items():
        print(f"\n {name} ({len(df)} linhas) ")
        print(df.head().to_string(index=False))
  
