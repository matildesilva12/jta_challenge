# Quantitative evaluation of the recommendation methods

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from tools import (
    _INDEX,
    _COSINE,
    _NPMI_NODIAG,
    cooccurrence_matrix,
    products,
)


def _raw_cooccurrence_matrix() -> np.ndarray:
   # Co-occurrence matrix with the diagonal zeroed

   M = cooccurrence_matrix.values.astype(float).copy()
   np.fill_diagonal(M, 0.0)
   return M


def _method_matrices() -> dict[str, np.ndarray]:
    # Return the three score matrices
    return {
        "cooccurrence_raw": _raw_cooccurrence_matrix(),
        "cosine": _COSINE.copy(),
        "npmi": _NPMI_NODIAG.copy(),
    }


def ranking_agreement() -> pd.DataFrame:
    # Mean Spearman correlation between the rankings each method produces

    matrices = _method_matrices()
    names = list(matrices.keys())
    n = len(_INDEX)
    result = pd.DataFrame(np.eye(len(names)), index=names, columns=names)

    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            ma, mb = matrices[names[a]], matrices[names[b]]
            corrs = []
            for i in range(n):
                row_a = np.delete(ma[i], i)
                row_b = np.delete(mb[i], i)
                if np.ptp(row_a) == 0 or np.ptp(row_b) == 0:
                    continue
                rho, _ = spearmanr(row_a, row_b)
                if not np.isnan(rho):
                    corrs.append(rho)
            mean_rho = float(np.mean(corrs)) if corrs else float("nan")
            result.iloc[a, b] = round(mean_rho, 3)
            result.iloc[b, a] = round(mean_rho, 3)
    return result


def topn_overlap(top_n: int = 3) -> pd.DataFrame:
    # Mean top-N overlap between each pair of methods 
    matrices = _method_matrices()
    names = list(matrices.keys())
    n = len(_INDEX)
    result = pd.DataFrame(np.ones((len(names), len(names))), index=names, columns=names)

    def top_set(matrix, i):
        row = matrix[i].copy()
        row[i] = -np.inf  # exclui o próprio
        return set(np.argsort(row)[::-1][:top_n])

    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            ma, mb = matrices[names[a]], matrices[names[b]]
            overlaps = []
            for i in range(n):
                sa, sb = top_set(ma, i), top_set(mb, i)
                if sa and sb:
                    overlaps.append(len(sa & sb) / top_n)
            mean_ov = float(np.mean(overlaps)) if overlaps else float("nan")
            result.iloc[a, b] = round(mean_ov, 3)
            result.iloc[b, a] = round(mean_ov, 3)
    return result


def popularity_bias() -> pd.DataFrame:
    # Quantify each method's popularity bias: Spearman correlation between a product's mean 
    # recommendation score and its popularity (times_sold) (returns correlation and p-value per method)
    matrices = _method_matrices()
    popularity = np.array([products[name]["times_sold"] for name in _INDEX], dtype=float)

    rows = []
    for method, M in matrices.items():
        scores = []
        for j in range(len(_INDEX)):
            col = np.delete(M[:, j], j)
            scores.append(np.mean(col))
        scores = np.array(scores)
        if np.ptp(scores) == 0:
            rho, p = float("nan"), float("nan")
        else:
            rho, p = spearmanr(scores, popularity)
        rows.append({
            "method": method,
            "corr_com_popularidade": round(float(rho), 3),
            "p_value": round(float(p), 3),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Concordância de rankings (Spearman médio)")
    print(ranking_agreement().to_string())
    print("\n Sobreposição top-3 ")
    print(topn_overlap(3).to_string())
    print("\n Viés de popularidade (corr. score vs times_sold)")
    print(popularity_bias().to_string(index=False))
