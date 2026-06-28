import json
from pathlib import Path

BASE_DIR = Path(__file__).parent
JSON_PATH = BASE_DIR.parent / "data" / "dataset.json"

# The set of keys that each product in that category must have
EXPECTED_SCHEMA = {
    "Console": {
        "name", "category", "times_sold", "Store A", "Store B", "Store C",
    },
    "Games": {
        "name", "category", "type", "franchise", "min_age",
        "release_date", "times_sold", "Store A", "Store B", "Store C",
    },
    "Accessories": {
        "name", "category", "type", "times_sold",
        "Store A", "Store B", "Store C",
    },
}


def validate_schema(data: dict) -> list[str]:
    # Check each product has exactly the expected keys for its category

    errors = []
    for category, items in data.items():
        expected = EXPECTED_SCHEMA.get(category)
        if expected is None:
            errors.append(f"Categoria inesperada no JSON: '{category}'")
            continue
        for i, item in enumerate(items):
            keys = set(item.keys())
            missing = expected - keys
            extra = keys - expected
            name = item.get("name", f"<sem nome, índice {i}>")
            if missing:
                errors.append(f"[{category}] '{name}': campos em falta: {sorted(missing)}")
            if extra:
                errors.append(f"[{category}] '{name}': campos inesperados: {sorted(extra)}")
    return errors


def validate_consistent_keys_within_category(data: dict) -> list[str]:
    errors = []
    for category, items in data.items():
        if not items:
            continue
        reference = set(items[0].keys())
        for i, item in enumerate(items[1:], start=1):
            keys = set(item.keys())
            if keys != reference:
                name = item.get("name", f"<índice {i}>")
                errors.append(
                    f"[{category}] '{name}' tem chaves diferentes dos restantes: "
                    f"falta {sorted(reference - keys)}, a mais {sorted(keys - reference)}"
                )
    return errors


def validate_sales_checksum(data: dict) -> list[str]:
    errors = []
    for category, items in data.items():
        for item in items:
            stores = sum(
                (item.get(s) or 0) for s in ("Store A", "Store B", "Store C")
            )
            ts = item.get("times_sold")
            if ts is not None and stores != ts:
                errors.append(
                    f"[{category}] '{item.get('name')}': soma das lojas ({stores}) "
                    f"!= times_sold ({ts})"
                )
    return errors


def validate_no_duplicate_names(data: dict) -> list[str]:
    seen = {}
    errors = []
    for category, items in data.items():
        for item in items:
            name = item.get("name")
            if name in seen:
                errors.append(f"Nome duplicado: '{name}' em '{category}' e '{seen[name]}'")
            else:
                seen[name] = category
    return errors


def validate_value_ranges(data: dict) -> list[str]:
    # sales (times_sold and per store) non-negative
    # min_age in a acceptable range (0-21)

    errors = []
    for category, items in data.items():
        for item in items:
            name = item.get("name")
            for field in ("times_sold", "Store A", "Store B", "Store C"):
                v = item.get(field)
                if v is not None and v < 0:
                    errors.append(f"[{category}] '{name}': {field} negativo ({v})")
            age = item.get("min_age")
            if age is not None and not (0 <= age <= 21):
                errors.append(f"[{category}] '{name}': min_age fora do intervalo (0-21): {age}")
    return errors


def run_all_validations(data: dict) -> dict[str, list[str]]:
    return {
        "schema": validate_schema(data),
        "consistencia_chaves": validate_consistent_keys_within_category(data),
        "checksum_vendas": validate_sales_checksum(data),
        "nomes_duplicados": validate_no_duplicate_names(data),
        "intervalos_valores": validate_value_ranges(data),
    }



def _root(s: str) -> str:
    # Reduce a string to its root by stripping common plural/singular suffixes
    s = s.lower()
    for suffix in ("ies", "es", "s", "y"):  # do mais longo para o mais curto
        if s.endswith(suffix):
            return s[: -len(suffix)]
    return s


def category_matches_key(category: str | None, key: str) -> bool:
    if category is None:
        return False
    rc, rk = _root(category), _root(key)
    return rc in rk or rk in rc


def diagnose_category_vs_key(data: dict) -> dict:
    report = {}
    for key, items in data.items():
        categories_seen = {}
        for it in items:
            c = it.get("category")
            categories_seen[c] = categories_seen.get(c, 0) + 1
        is_internally_consistent = len(categories_seen) == 1

        mismatches = []
        for it in items:
            if not category_matches_key(it.get("category"), key):
                mismatches.append({
                    "name": it.get("name"),
                    "category_atual": it.get("category"),
                    "type_atual": it.get("type"),
                })

        mismatches_sem_type = [
            m["name"] for m in mismatches
            if "type" not in {k for k in next(
                (it for it in items if it.get("name") == m["name"]), {}
            )}
        ]

        report[key] = {
            "categories_encontradas": categories_seen,
            "category_consistente_na_key": is_internally_consistent,
            "todas_batem_com_key": len(mismatches) == 0,
            "n_mismatches": len(mismatches),
            "mismatches": mismatches,
            "mismatches_sem_campo_type": mismatches_sem_type,
            "troca_segura": len(mismatches) > 0 and not mismatches_sem_type,
        }
    return report


def print_category_diagnosis(data: dict) -> None:
    
    report = diagnose_category_vs_key(data)
    for key, r in report.items():
        print(f" Key: {key} ")
        print(f" categories encontradas: {r['categories_encontradas']}")
        print(f" category consistente dentro da key? {r['category_consistente_na_key']}")
        print(f" todas batem com a key (containment)? {r['todas_batem_com_key']}")
        if r["n_mismatches"]:
            print(f"  {r['n_mismatches']} produto(s) onde category NÃO bate com a key:")
            for m in r["mismatches"]:
                print(f"      - {m['name']!r}: category={m['category_atual']!r}, type={m['type_atual']!r}")
            if r["mismatches_sem_campo_type"]:
                print(f" Sem campo 'type' (troca INSEGURA): {r['mismatches_sem_campo_type']}")
            else:
                print(f" Todos os mismatches têm campo 'type' (troca segura)")
        else:
            print(" Sem mismatches (nada a corrigir nesta key)")
        print()


def fix_category_type_swap(data: dict, strict: bool = True) -> tuple[dict, list[str]]:
    import copy
    data = copy.deepcopy(data)
    log = []

    for key, items in data.items():
        for item in items:
            current = item.get("category")
            if category_matches_key(current, key):
                continue 

            name = item.get("name")

            # Discover which fields match the key
            campos_que_batem = [
                campo for campo, valor in item.items()
                if campo != "category"
                and isinstance(valor, str)
                and category_matches_key(valor, key)
            ]

            if len(campos_que_batem) == 0:
                msg = (f"NÃO corrigido [{key}] '{name}': category={current!r} não "
                       f"bate com a key e nenhum outro campo bate (troca impossível)")
                if strict:
                    raise ValueError(msg)
                log.append("AVISO: " + msg)
                continue

            if len(campos_que_batem) > 1:
                msg = (f"NÃO corrigido [{key}] '{name}': category={current!r} não bate, "
                       f"mas {len(campos_que_batem)} campos batem com a key {campos_que_batem} "
                       f"(troca ambígua)")
                if strict:
                    raise ValueError(msg)
                log.append("AVISO: " + msg)
                continue

            # Exactly one field matches (generic swap category)
            campo = campos_que_batem[0]
            subtipo_original = current
            item["category"], item[campo] = item[campo], item["category"]
            log.append(
                f"Corrigido [{key}] '{name}': category {subtipo_original!r}->"
                f"{item['category']!r} (trocada com campo '{campo}'; "
                f"valor original preservado em '{campo}'={item[campo]!r})"
            )

    return data, log


if __name__ == "__main__":
    with open(JSON_PATH, encoding="utf-8") as f:
        data = json.load(f)

    # General quality validations
    print("=" * 60)
    print("1) VALIDAÇÕES GERAIS")
    print("=" * 60)
    report = run_all_validations(data)
    total = sum(len(v) for v in report.values())
    print(f"{total} problema(s) encontrado(s)\n")
    for check, problems in report.items():
        status = "OK" if not problems else f"{len(problems)} problema(s)"
        print(f"[{status}] {check}")
        for p in problems:
            print(f"    - {p}")

    # Category vs key diagnosis
    print("\n" + "=" * 60)
    print("2) DIAGNÓSTICO: category vs key")
    print("=" * 60)
    print_category_diagnosis(data)

    # Apply the safe fix and re-validate
    print("=" * 60)
    print("3) CORREÇÃO category<->type (apenas onde seguro)")
    print("=" * 60)
    fixed, log = fix_category_type_swap(data, strict=True)
    for line in log:
        print("  " + line)
    print("\n  Re-diagnóstico após correção:")
    after = diagnose_category_vs_key(fixed)
    for key, r in after.items():
        print(f"    {key}: todas batem com a key? {r['todas_batem_com_key']} "
              f"(mismatches: {r['n_mismatches']})")



def diagnose_category_mismatches(data: dict) -> list[dict]:
    # Returns a list of mismatches (empty = all match)

    mismatches = []
    for key, items in data.items():
        for item in items:
            category = item.get("category")
            if category_matches_key(category, key):
                continue  # bate -> mantém-se, nada a reportar

            campos_que_batem = []
            for campo, valor in item.items():
                if campo == "category":
                    continue
                if isinstance(valor, str) and category_matches_key(valor, key):
                    campos_que_batem.append((campo, valor))

            mismatches.append({
                "key": key,
                "name": item.get("name"),
                "category_atual": category,
                "campos_que_batem_com_a_key": campos_que_batem,
                "todas_as_colunas": dict(item),
            })
    return mismatches


def print_category_mismatches(data: dict) -> None:

    mismatches = diagnose_category_mismatches(data)
    if not mismatches:
        print("Todas as categories batem com as respetivas keys")
        return

    print(f"{len(mismatches)} produto(s) onde category não bate com a key:\n")
    for m in mismatches:
        print(f"  [{m['key']}] {m['name']!r}")
        print(f"      category atual: {m['category_atual']!r} (não bate com '{m['key']}')")
        if m["campos_que_batem_com_a_key"]:
            for campo, valor in m["campos_que_batem_com_a_key"]:
                print(f"      → campo '{campo}' = {valor!r} BATE com a key")
        else:
            print(f"      nenhum outro campo bate com a key '{m['key']}'")
        print(f"      todas as colunas (para decidir a transformação):")
        for col, val in m["todas_as_colunas"].items():
            print(f"          {col} = {val!r}")
        print()
