# Demonstrates a recommendation approach based on each customer's purchase history (toward sequential recommendation), instead of aggregated co-occurrence

import numpy as np
import pandas as pd

from data_loader_v2 import load_all


def build_history(scale: float = 0.001, n_customers: int = 500, seed: int = 42):
    # Generate synthetic dim_customers and orders from the co-occurrence matrix

    rng = np.random.default_rng(seed)
    products, matrix = load_all()
    names = list(matrix.index)
    M = matrix.values

    orders_pool = []  

    # solo sales (diagonal)
    for i, name in enumerate(names):
        n = int(M[i, i] * scale)
        orders_pool.extend([[name]] for _ in range(n))

    # co-occurring pairs (upper triangle)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            n = int(M[i, j] * scale)
            orders_pool.extend([[names[i], names[j]]] for _ in range(n))

    # flatten
    orders_pool = [o[0] if isinstance(o[0], list) else o for o in orders_pool]

    rng.shuffle(orders_pool)

    # dim_customers
    segmentos = ["casual", "entusiasta", "colecionador"]
    dim_customers = pd.DataFrame({
        "customer_id": range(1, n_customers + 1),
        "customer_name": [f"Cliente_{i:04d}" for i in range(1, n_customers + 1)],
        "segment": rng.choice(segmentos, size=n_customers,
                              p=[0.6, 0.3, 0.1]),
    })

    # Assign orders to customers (fabricated) and give a customer sequence
    order_rows = []
    seq_por_cliente = {cid: 0 for cid in dim_customers["customer_id"]}
    for k, basket in enumerate(orders_pool, start=1):
        cid = int(rng.integers(1, n_customers + 1))
        seq_por_cliente[cid] += 1
        for produto in basket:
            order_rows.append({
                "order_id": k,
                "customer_id": cid,
                "product": produto,
                "order_seq": seq_por_cliente[cid], 
            })
    orders = pd.DataFrame(order_rows)

    return dim_customers, orders


def recommend_from_history(customer_id: int, dim_customers, orders, top_n: int = 3):
    # A simple item-item collaborative filter, now anchored on individual history (the conceptual bridge to sequential recommendation)
    
    history = orders[orders["customer_id"] == customer_id]["product"].unique().tolist()
    if not history:
        return {"customer_id": customer_id, "history": [], "recommendations": [],
                "note": "Cliente sem histórico (cold start): usaria popularidade global."}

    # Co-occurrence at the order level: which products appear in the same orders as the products in the customer's history
    orders_com_historico = orders[orders["product"].isin(history)]["order_id"].unique()
    candidatos = orders[orders["order_id"].isin(orders_com_historico)]
    contagem = candidatos[~candidatos["product"].isin(history)]["product"].value_counts()

    recs = contagem.head(top_n).index.tolist()
    return {
        "customer_id": customer_id,
        "history": history,
        "recommendations": recs,
        "note": "Recomendação item-item ancorada no histórico do cliente (dados sintéticos)",
    }


if __name__ == "__main__":
    dim_customers, orders = build_history()
    print(" dim_customers (sintética) ")
    print(dim_customers.head().to_string(index=False))
    print(f"\n{len(dim_customers)} clientes, {orders['order_id'].nunique()} orders, "
          f"{len(orders)} linhas de order.")

    print("\n orders (sintética) — exemplo ")
    print(orders.head(8).to_string(index=False))


    cliente_exemplo = orders["customer_id"].value_counts().index[0]  
    print(f"\n Recomendação por histórico — cliente {cliente_exemplo} ")
    rec = recommend_from_history(cliente_exemplo, dim_customers, orders)
    print("Histórico:", rec["history"])
    print("Recomenda:", rec["recommendations"])
    print("Nota:", rec["note"])
