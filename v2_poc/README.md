# V2 History-based recommendation (exploratory)

**Independent** extension of the main system (v1). Imports nothing from v1, it has
its own copy of the data and a minimal loader.

Explores the brief's *what-if*: recommendation with **purchase history** and
**cold-start** handling, through a **hybrid** approach (sequential for customers with
history + content-based/popularity for new customers).

## Methodological note

`dim_customers` and `orders` are **synthetic**. The aggregates are aligned to the real
JTA data (they derive from the co-occurrence matrix; sale proportions follow
`times_sold`), but customer assignment and temporal order are fabricated. It serves to
**demonstrate the approach and structure**, not to validate quality.

## Files

- `data_loader_v2.py` — minimal, self-contained loader (products + matrix only).
- `history_v2.py` — generates the synthetic tables and a simple item-item recommender.
- `recomendacao_historico_v2.ipynb` — notebook with the hybrid proposal (cold start vs
  history, weighted transition) and the table demonstration.
- `data/` — copy of the required data.

## Running


```bash
pip install pandas numpy openpyxl matplotlib jupyter

#generates and tests the tables
python history_v2.py 

jupyter notebook recomendacao_historico_v2.ipynb
```