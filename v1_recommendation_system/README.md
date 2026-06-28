# JTA Challenge: Agentic recommendation system (Nintendo Switch)

## Project structure

```
jta_project/
├── README.md
├── requirements.txt
├── .gitignore
├── data/                       # input data (JTA's, intact + our URLs)
│   ├── dataset.json
│   ├── Nintendo_Cooccurrence_Matrix.xlsx
│   └── product_urls.json
├── src/                        # source code (logic, testable without the LLM)
│   ├── data_loader.py          # reading + category/type fix at ingestion
│   ├── validations.py          # data-quality validations
│   ├── data_model.py           # dimensional model (star schema)
│   ├── tools.py                # agent tools (co-occurrence, cosine, NPMI)
│   ├── agent.py                # agentic loop (function calling)
│   └── evaluation.py           # method evaluation metrics
├── app/                        # interfaces
│   ├── main.py                 # CLI
│   └── streamlit_app.py        # visual interface
└── notebooks/                  # analysis and demonstration
    ├── demonstracao_QA.ipynb
    ├── analise_datascience.ipynb
    └── validacao_escala.ipynb
```

## First things first: the OpenAI key

The key shared in the original email should be placed in a .env file as `OPENAI_API_KEY=<key>` to be loaded as an environment variable.

## Installation

```bash
pip install -r requirements.txt
```

## Modules

| File | Responsibility |
|------|----------------|
| `data_loader.py` | Reads and normalizes `dataset.json` + Excel matrix; validates consistency (fail-fast). |
| `validations.py` | Data-quality validations (schema, key consistency, category/type fix). |
| `data_model.py` | Dimensional model (star schema): shows production-oriented organization. |
| `tools.py` | Functions the agent can call. Deterministic logic, testable without the LLM. |
| `agent.py` | Agentic loop (OpenAI function calling). |
| `evaluation.py` | Recommendation method evaluation metrics. |
| `main.py` | Interactive CLI / single query. |
| `streamlit_app.py` | Visual chat interface. |

The separation is deliberate: the **business logic (`tools.py`) is deterministic and
testable on its own**; the LLM only orchestrates which tool to call and when.

## Data model (design decisions)

The raw data is turned into a **star-schema** dimensional model, materialized as
separate pandas DataFrames but designed as relational tables (each with its own key,
linked by foreign keys).

**Star schema:**
- `dim_product` (PK product_id): name, category, type, franchise, min_age, release_date
- `dim_store` (PK store_id): store_name
- `fact_sales` (FK product_id, FK store_id): units_sold; grain: product x store
- `fact_cooccurrence` (FK product_id_a, FK product_id_b): times_together

**Key decisions (defensible in the review):**

1. **Surrogate keys.** `product_id` and `store_id` are generated integers, not names.
   In production, names change (rebranding, typos) and are costly to index/join;
   integers are stable and efficient.

2. **`fact_sales` is aggregated, not transactional.** The data only gives totals
   (`times_sold` per store), not individual sale rows. We call it `fact_sales`
   (grain = product x store), not "orders" — there are no real orders. The Phase 2
   transactional approach is described in "Future improvements" below.

3. **`fact_cooccurrence` captures the product↔product relation** (the core of
   recommendation). Without it, a classic dimensional model would throw away the
   most important information. We keep only the upper triangle (a_id < b_id) to
   avoid duplicating pairs — the matrix is symmetric.

4. **No partitioning in Phase 1.** With few products it would be over-engineering;
   the model stays clean and keyed. The scaling strategy (including partitioning) is
   described in "Future improvements" below. Keys coexist with partitioning when it
   comes — they are not alternatives.

5. **Informative referential integrity.** `check_referential_integrity()` checks and
   reports foreign keys with no match in the dimension, without blocking — like
   Spark/Databricks/Delta, where enforcing constraints at scale is costly but keys
   still link the tables.

6. **Pandas materialization (Phase 1).** The schema is relational by design; migrating
   to SQLite or a partitioned data lake is mechanical (same fields and keys). We keep
   pandas now for fast iteration.

## Recommendation methods

Three complementary signals, all built from the co-occurrence matrix:

- **Raw co-occurrence** (`get_cooccurring_products`) — "bought together"
  (complementarity: console + controller + game).
- **Cosine similarity** (`get_similar_products`) — "similar to" (affinity: another
  product the same kind of customer would buy). Each product is a vector (its matrix
  row); cosine compares co-purchase *patterns*, dampening the popularity bias.
- **NPMI** (`get_recommendations_npmi`) — **main engine** (Market Basket Analysis).
  Scores how much two products co-occur above chance, correcting the popularity bias
  of raw co-occurrence. Cosine acts as a cross-check.

Honest note: with this dataset (16 products) the methods largely converge, dominated
by the console/best-sellers. This illustrates the scale problem the brief asks us to
discuss (see the analysis notebook).

## Running

```bash
# Interactive CLI
python app/main.py

# Single query
python app/main.py "Recommend something similar to Splatoon 3 at Store C"

# Or open the demo notebook:
jupyter notebook notebooks/demonstracao_QA.ipynb
```

## Visual interface (Streamlit)

Why Streamlit over static HTML: it runs as a Python process, so OPENAI_API_KEY
stays in an environment variable, never exposed to a browser. Reuses the existing
agent (`run_agent`); only handles presentation.

```bash
streamlit run app/streamlit_app.py
```

Opens a browser page with a natural-language chat. By default, recommendations show
only the product name and link; full detail is shown on explicit request.

## Data Science analysis

- `evaluation.py` — evaluation metrics for the recommendation methods: ranking
  agreement (Spearman), top-N overlap, and popularity-bias quantification (correlation
  between score and `times_sold`). Testable logic, without the LLM.
- `analise_datascience.ipynb` — analysis notebook: visualizes method agreement,
  **quantifies and plots the popularity bias** (showing NPMI corrects it), and discusses
  the **sequential recommendation / cold-start what-if** (Markov, matrix factorization,
  sequence models, hybrid strategies).

Central result: raw co-occurrence correlates ~0.85 with popularity (strong bias); NPMI
breaks that correlation: a quantified justification for choosing it as the main engine.

## Scale validation

- `validacao_escala.ipynb` validates the methods on a larger **real** dataset
  (Online Retail), molded to the same structure as the JTA data. Tests whether the
  weak scores on the challenge data come from the **method** or from the **data size**.
  The co-occurrence matrix is real (it validates NPMI); the per-store split is fabricated
  only to keep the structure, and does not enter the recommendation computation.

## Future improvements (production & scale rationale)

Deliberate next steps, not implemented in Phase 1 with 16 products they would be
over-engineering.

### Scaling the data model

The Phase 1 model is a clean, keyed star schema in pandas. When
data grows (thousands of products, millions of pairs), the strategy is:

- **Keys still structure the fact↔dim joins** and they coexist with partitioning; they
  are not alternatives.
- **`fact_sales`**: partition by `date_key` (queries are typically over a time window),
  with `store_id` as a sub-partition / cluster key.
- **`fact_cooccurrence`**: partition by `product_id_a`, recommendations look up the
  neighborhood of one product, so each query touches only that partition (partition
  pruning), avoiding a full scan as pairs grow (~N²).
- **`dim_product` / `dim_store`**: small, replicated/cached, no partitioning.
- **Engine**: migrate from pandas to SQLite (medium volumes) or a partitioned columnar
  data lake / Spark-Delta (large scale).

Physical partitioning would write the fact tables as partitioned Parquet (one subfolder
per key value), e.g. `fact_sales` by `date_key` and `fact_cooccurrence` by `product_id_a`.