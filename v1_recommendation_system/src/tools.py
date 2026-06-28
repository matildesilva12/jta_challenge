# Tools the LLM can call via function calling

import difflib

import numpy as np

from data_loader import load_all

products, cooccurrence_matrix = load_all()

def _build_cosine_similarity(matrix):
    # Precompute the cosine similarity matrix once at startup

    M = matrix.values.astype(float)
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = M / norms
    sim = normalized @ normalized.T
    return sim


_COSINE = _build_cosine_similarity(cooccurrence_matrix)
_INDEX = list(cooccurrence_matrix.index)



def _build_npmi(matrix, include_diagonal: bool = True):
    # Precompute the NPMI matrix once. NPMI scores how much two products co-occur above chance, fixing the popularity bias of raw co-occurrence
    
    M = matrix.values.astype(float)
    diag = np.diag(M).copy()
    upper_pairs = np.triu(M, k=1).sum()

    if include_diagonal:
        N = diag.sum() + upper_pairs
        appearances = diag + (M.sum(axis=1) - diag)
    else:
        N = upper_pairs
        appearances = M.sum(axis=1) - diag

    if N <= 0:
        return np.zeros_like(M), 0.0

    p_x = appearances / N

    npmi = np.zeros_like(M)
    n = M.shape[0]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            joint = M[i, j]
            if joint <= 0 or p_x[i] <= 0 or p_x[j] <= 0:
                npmi[i, j] = 0.0
                continue
            p_joint = joint / N
            pmi = np.log(p_joint / (p_x[i] * p_x[j]))
            npmi[i, j] = pmi / (-np.log(p_joint))  # normaliza para [-1, 1]
    return npmi, N


_NPMI, _N_TRANSACTIONS_EST = _build_npmi(cooccurrence_matrix, include_diagonal=True)
_NPMI_NODIAG, _N_TRANSACTIONS_NODIAG = _build_npmi(cooccurrence_matrix, include_diagonal=False)


def _resolve_name(name: str) -> str | None:
    # Resolve an imprecise name to the canonical one (returns None if no match)

    if name in products:
        return name
    for canonical in products:
        if canonical.lower() == name.lower():
            return canonical
    matches = difflib.get_close_matches(name, list(products.keys()), n=1, cutoff=0.6)
    if matches:
        return matches[0]
    lowered = name.lower()
    for canonical in products:
        if lowered in canonical.lower():
            return canonical
    return None


def get_product_info(name: str) -> dict:
    # Return a product's full info

    resolved = _resolve_name(name)
    if resolved is None:
        return {
            "error": f"Produto '{name}' não encontrado.",
            "available_products": list(products.keys()),
        }
    return products[resolved]


def search_products(
    franchise: str | None = None,
    category: str | None = None,
    product_type: str | None = None,
    max_min_age: int | None = None,
) -> list[dict]:
    # Filter products by franchise, category, product_type, min_age

    results = []
    for product in products.values():
        if franchise is not None and product["franchise"] != franchise:
            continue
        if category is not None and product["category"] != category:
            continue
        if product_type is not None and product["type"] != product_type:
            continue
        if max_min_age is not None:
            min_age = product["min_age"]
            if min_age is not None and min_age > max_min_age:
                continue
        results.append(product)
    return results


def get_cooccurring_products(name: str, top_n: int = 5) -> dict:
    # Products most often bought together with the given one
    # Excludes the diagonal (product sold alone)

    resolved = _resolve_name(name)
    if resolved is None:
        return {
            "error": f"Produto '{name}' não encontrado.",
            "available_products": list(products.keys()),
        }

    row = cooccurrence_matrix.loc[resolved].drop(index=resolved)
    top = row.sort_values(ascending=False).head(top_n)

    return {
        "base_product": resolved,
        "standalone_sales": int(cooccurrence_matrix.loc[resolved, resolved]),
        "cooccurring": [
            {
                "product": product,
                "url": products.get(product, {}).get("url"),
                "times_bought_together": int(count),
            }
            for product, count in top.items()
            if count > 0
        ],
    }


def get_similar_products(name: str, top_n: int = 5) -> dict:
    # Products with a similar co-purchase profile (Good for finding alternatives)
    
    resolved = _resolve_name(name)
    if resolved is None:
        return {
            "error": f"Produto '{name}' não encontrado.",
            "available_products": list(products.keys()),
        }

    idx = _INDEX.index(resolved)
    scores = _COSINE[idx]
    ranked = sorted(
        ((_INDEX[j], scores[j]) for j in range(len(_INDEX)) if j != idx),
        key=lambda x: x[1],
        reverse=True,
    )[:top_n]

    return {
        "base_product": resolved,
        "method": "cosine_similarity",
        "similar": [
            {
                "product": product,
                "url": products.get(product, {}).get("url"),
                "similarity": round(float(score), 4),
            }
            for product, score in ranked
        ],
    }


def get_recommendations_npmi(name: str, top_n: int = 5, include_diagonal: bool = False) -> dict:
    resolved = _resolve_name(name)
    if resolved is None:
        return {
            "error": f"Produto '{name}' não encontrado.",
            "available_products": list(products.keys()),
        }

    idx = _INDEX.index(resolved)
    npmi_matrix = _NPMI if include_diagonal else _NPMI_NODIAG
    n_est = _N_TRANSACTIONS_EST if include_diagonal else _N_TRANSACTIONS_NODIAG
    npmi_scores = npmi_matrix[idx]
    cosine_scores = _COSINE[idx]

    ranked = sorted(
        ((j, npmi_scores[j]) for j in range(len(_INDEX)) if j != idx),
        key=lambda x: x[1],
        reverse=True,
    )[:top_n]

    return {
        "base_product": resolved,
        "method": "NPMI (principal) + cosine (validação cruzada)",
        "denominator": "com diagonal" if include_diagonal else "só pares (cabazes 2+)",
        "n_transactions_estimated": int(n_est),
        "assumption": "estimativa; sem nº real de transações e sem cabazes de 3+ produtos",
        "recommendations": [
            {
                "product": _INDEX[j],
                "url": products.get(_INDEX[j], {}).get("url"),
                "npmi": round(float(npmi_scores[j]), 4),
                "cosine_crosscheck": round(float(cosine_scores[j]), 4),
            }
            for j, _ in ranked
        ],
    }


def compare_npmi_denominators(name: str, top_n: int = 5) -> dict:
    # Compare NPMI recommendations with and without the diagonal for the same product
    return {
        "base_product": name,
        "com_diagonal": get_recommendations_npmi(name, top_n, include_diagonal=True),
        "sem_diagonal": get_recommendations_npmi(name, top_n, include_diagonal=False),
    }


def filter_by_store(product_names: list[str], store: str) -> list[dict]:
    if store not in ("Store_A", "Store_B", "Store_C"):
        return [{"error": f"Loja inválida: '{store}'. Use Store_A, Store_B ou Store_C."}]

    filtered = []
    for name in product_names:
        resolved = _resolve_name(name)
        if resolved is None:
            continue
        value = products[resolved]["store_breakdown"].get(store)
        if value:  # exclui None e 0
            filtered.append(products[resolved])
    return filtered


def exclude_franchise(product_names: list[str], franchise: str) -> list[str]:
    # Remove all products of the given franchise

    result = []
    for name in product_names:
        resolved = _resolve_name(name)
        if resolved is None:
            continue
        if products[resolved].get("franchise") != franchise:
            result.append(resolved)
    return result


# JSON schema describing these functions to the OpenAI model
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_product_info",
            "description": "Devolve informação completa sobre um produto específico pelo nome (aceita nomes aproximados).",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Nome do produto"}
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": (
                "Pesquisa produtos por franchise, categoria, tipo, e/ou idade mínima/máxima. "
                "Útil para encontrar produtos que cumprem critérios sem saber o nome exato."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "franchise": {"type": "string", "description": "Ex: 'Super Mario', 'The Legend of Zelda'"},
                    "category": {"type": "string", "description": "Ex: 'Game', 'Console', 'Accessory'"},
                    "product_type": {"type": "string", "description": "Ex: 'Platformer', 'Racing', 'Controller'"},
                    "max_min_age": {
                        "type": "integer",
                        "description": "Devolve produtos apropriados até esta idade (min_age <= valor)",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_cooccurring_products",
            "description": (
                "Produtos mais comprados junto com o produto dado (mesmo cabaz). "
                "Usa para complementaridade: 'quem comprou X também comprou Y'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Nome do produto base"},
                    "top_n": {"type": "integer", "description": "Quantos devolver (default 5)"},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_similar_products",
            "description": (
                "Produtos parecidos com o dado (perfil de cliente semelhante), via similaridade "
                "de cosseno. Usa quando o utilizador quer algo do mesmo estilo/alternativa, "
                "não necessariamente comprados juntos. Ideal para 'gostei de X, quero algo parecido'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Nome do produto base"},
                    "top_n": {"type": "integer", "description": "Quantos devolver (default 5)"},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recommendations_npmi",
            "description": (
                "Recomendações por força de associação via NPMI (Market Basket Analysis) "
                "corrige o viés de popularidade da co-ocorrência bruta. Inclui o score de "
                "cosseno como validação cruzada. Usa como motor principal de recomendação "
                "quando se quer saber que produtos têm ligação genuína com o que foi dado."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Nome do produto base"},
                    "top_n": {"type": "integer", "description": "Quantos devolver (default 5)"},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "filter_by_store",
            "description": "Filtra uma lista de produtos, mantendo apenas os disponíveis numa loja específica.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_names": {"type": "array", "items": {"type": "string"}},
                    "store": {"type": "string", "enum": ["Store_A", "Store_B", "Store_C"]},
                },
                "required": ["product_names", "store"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "exclude_franchise",
            "description": "Remove de uma lista de produtos todos os que pertencem a uma franchise específica.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_names": {"type": "array", "items": {"type": "string"}},
                    "franchise": {"type": "string"},
                },
                "required": ["product_names", "franchise"],
            },
        },
    },
]

AVAILABLE_FUNCTIONS = {
    "get_product_info": get_product_info,
    "search_products": search_products,
    "get_cooccurring_products": get_cooccurring_products,
    "get_similar_products": get_similar_products,
    "get_recommendations_npmi": get_recommendations_npmi,
    "filter_by_store": filter_by_store,
    "exclude_franchise": exclude_franchise,
}


if __name__ == "__main__":
    print(" Co-ocorrência direta (Mario Odyssey) ")
    print(get_cooccurring_products("Super Mario Odyssey", top_n=3))
    print("\n Similaridade de cosseno (Mario Odyssey) ")
    print(get_similar_products("Super Mario Odyssey", top_n=3))
    print("\n Resolução de nome aproximado ('Odyssey') ")
    print(_resolve_name("Odyssey"))
    print("\n Excluir franchise Super Mario ")
    similar = [p["product"] for p in get_similar_products("Super Mario Odyssey", top_n=10)["similar"]]
    print(exclude_franchise(similar, "Super Mario"))
