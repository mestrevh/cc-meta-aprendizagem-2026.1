"""
Módulo de seleção híbrida de meta-features.

Combina soluções dos métodos exatos (MILP, knapsack, correlação penalizada)
com o Algoritmo Genético via Seeded GA — as soluções exatas são convertidas
em indivíduos binários e inseridas como elites na população inicial do AG.
"""

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from pulp import (
    LpProblem, LpMaximize, LpVariable, lpSum,
    PULP_CBC_CMD,
)

from ga_common import GAConfig, GAResult
from genetic_selector import GeneticFeatureSelector


# ═══════════════════════════════════════════════════════════════════
# Conversores
# ═══════════════════════════════════════════════════════════════════

def features_to_individual(
    selected_features: list[str],
    feature_cols: list[str],
) -> np.ndarray:
    """
    Converte uma lista de nomes de features selecionadas em um
    indivíduo binário compatível com o AG.

    Parâmetros
    ----------
    selected_features : lista de nomes de features selecionadas.
    feature_cols : lista completa de features (define a ordem dos bits).

    Retorna
    -------
    np.ndarray de uint8 com 1 nas posições selecionadas.
    """
    individual = np.zeros(len(feature_cols), dtype=np.uint8)
    col_index = {col: i for i, col in enumerate(feature_cols)}

    for feat in selected_features:
        if feat in col_index:
            individual[col_index[feat]] = 1

    return individual


# ═══════════════════════════════════════════════════════════════════
# Métodos exatos (wrappers reutilizáveis)
# ═══════════════════════════════════════════════════════════════════

def best_subset_selection_filter_milp(
    meta_dataset: pd.DataFrame,
    meta_models: dict,
    meta_model_feature_scores: dict,
    max_feature: int = 200,
) -> dict[str, list[str]]:
    """
    Seleção de subconjunto ótimo via MILP — seleciona exatamente
    `max_feature` meta-features que maximizam a soma das importâncias.

    Retorna dict {nome_algoritmo → lista de features selecionadas}.
    """
    warnings.filterwarnings("ignore")

    meta_features = [
        col for col in meta_dataset.columns
        if col not in ("Best", "Dataset")
    ]
    selected_by_algorithm: dict[str, list[str]] = {}

    for key in meta_models:
        problem = LpProblem(f"Best_Subset_Filter_{key}", LpMaximize)

        x = {
            i: LpVariable(f"{meta_features[i]}", cat="Binary")
            for i in range(len(meta_features))
        }

        # Objetivo: maximizar soma ponderada das importâncias
        problem += lpSum(
            meta_model_feature_scores[key][i] * x[i]
            for i in range(len(meta_features))
        )

        # Restrição: selecionar exatamente max_feature
        n_select = min(max_feature, len(meta_features))
        problem += lpSum(x[i] for i in range(len(meta_features))) == n_select

        problem.solve(PULP_CBC_CMD(msg=False))

        selected = [
            meta_features[i]
            for i in range(len(meta_features))
            if x[i].varValue == 1.0
        ]
        selected_by_algorithm[key] = selected

    return selected_by_algorithm


def knapsack_problem_ilp(
    meta_dataset: pd.DataFrame,
    meta_models: dict,
    meta_model_feature_scores: dict,
    max_feature: int = 200,
    l: float = 0.002,
) -> dict[str, list[str]]:
    """
    Seleção via problema da mochila (ILP) — seleciona até `max_feature`
    meta-features maximizando (importância - λ) para cada feature.

    Retorna dict {nome_algoritmo → lista de features selecionadas}.
    """
    warnings.filterwarnings("ignore")

    meta_features = [
        col for col in meta_dataset.columns
        if col not in ("Best", "Dataset")
    ]
    selected_by_algorithm: dict[str, list[str]] = {}

    for key in meta_models:
        problem = LpProblem("Meta_Feature_Selection", LpMaximize)

        x = {
            i: LpVariable(f"{meta_features[i]}", cat="Binary")
            for i in range(len(meta_features))
        }

        # Objetivo: maximizar (importância - λ) * x
        problem += lpSum(
            (meta_model_feature_scores[key][i] - l) * x[i]
            for i in range(len(meta_features))
        )

        # Restrição: no máximo max_feature
        problem += lpSum(x[i] for i in range(len(meta_features))) <= max_feature

        problem.solve(PULP_CBC_CMD(msg=False))

        selected = [
            meta_features[i]
            for i in range(len(meta_features))
            if x[i].varValue == 1.0
        ]
        selected_by_algorithm[key] = selected

    return selected_by_algorithm


def correlation_penalized_by_milp(
    meta_dataset: pd.DataFrame,
    meta_models: dict,
    meta_model_feature_scores: dict,
    max_feature: int = 200,
    gama: float = 0.05,
    corr_threshold: float = 0.3,
) -> dict[str, list[str]]:
    """
    Seleção via MILP com penalização de correlação — maximiza importância
    enquanto penaliza pares de features altamente correlacionados.

    Retorna dict {nome_algoritmo → lista de features selecionadas}.
    """
    warnings.filterwarnings("ignore")

    meta_features = [
        col for col in meta_dataset.columns
        if col not in ("Best", "Dataset")
    ]
    selected_by_algorithm: dict[str, list[str]] = {}

    corr_matrix = meta_dataset[meta_features].corr().abs().fillna(0).values

    for key in meta_models:
        print(f"   -> Otimizando subconjunto para o modelo: {key}...")

        problem = LpProblem(f"Penalizacao_Corr_{key}", LpMaximize)

        x = {
            i: LpVariable(f"x_{i}", cat="Binary")
            for i in range(len(meta_features))
        }

        # Variáveis auxiliares para pares correlacionados
        z = {}
        for i in range(len(meta_features)):
            for j in range(i + 1, len(meta_features)):
                if corr_matrix[i][j] >= corr_threshold:
                    z[(i, j)] = LpVariable(f"z_{i}_{j}", cat="Binary")

        # Restrições de linearização
        for (i, j) in z:
            problem += z[i, j] <= x[i]
            problem += z[i, j] <= x[j]
            problem += z[i, j] >= x[i] + x[j] - 1

        # Restrição de cardinalidade
        problem += lpSum(x[i] for i in range(len(meta_features))) <= max_feature

        # Objetivo: importância - penalidade de correlação
        linear_scores = lpSum(
            meta_model_feature_scores[key][i] * x[i]
            for i in range(len(meta_features))
        )
        redundancy_penalty = lpSum(
            corr_matrix[i][j] * z[i, j] for (i, j) in z
        )
        problem += linear_scores - (gama * redundancy_penalty)

        solver = PULP_CBC_CMD(msg=False, timeLimit=45, gapRel=0.02)
        problem.solve(solver)

        selected = [
            meta_features[i]
            for i in range(len(meta_features))
            if x[i].varValue == 1.0
        ]
        selected_by_algorithm[key] = selected
        print(f"   Concluído! Selecionadas {len(selected)} features.")

    return selected_by_algorithm


# ═══════════════════════════════════════════════════════════════════
# Pipeline híbrido
# ═══════════════════════════════════════════════════════════════════

def run_exact_methods(
    meta_dataset: pd.DataFrame,
    meta_models: dict,
    meta_model_feature_scores: dict,
    max_feature: int = 200,
    knapsack_lambda: float = 0.002,
    corr_gamma: float = 0.05,
    corr_threshold: float = 0.3,
) -> dict[str, dict[str, list[str]]]:
    """
    Executa os três métodos exatos e retorna as features selecionadas.

    Retorna
    -------
    dict com estrutura {nome_método → {nome_algoritmo → [features]}}
    """
    print("=" * 70)
    print("Executando métodos exatos para gerar sementes do AG híbrido")
    print("=" * 70)

    results: dict[str, dict[str, list[str]]] = {}

    print("\n[1/3] Best Subset Selection by Filter MILP...")
    results["Best Subset Selection by Filter MILP"] = (
        best_subset_selection_filter_milp(
            meta_dataset, meta_models, meta_model_feature_scores,
            max_feature=max_feature,
        )
    )

    print("[2/3] Knapsack Problem ILP...")
    results["Knapsack Problem ILP"] = knapsack_problem_ilp(
        meta_dataset, meta_models, meta_model_feature_scores,
        max_feature=max_feature, l=knapsack_lambda,
    )

    print("[3/3] Correlation Penalized MILP...")
    results["Correlation Penalized milp"] = correlation_penalized_by_milp(
        meta_dataset, meta_models, meta_model_feature_scores,
        max_feature=max_feature, gama=corr_gamma,
        corr_threshold=corr_threshold,
    )

    # Resumo
    print("\n" + "-" * 70)
    print("Resumo das sementes geradas:")
    for method_name, alg_features in results.items():
        for alg_name, features in alg_features.items():
            print(f"  {method_name} | {alg_name}: {len(features)} features")
    print("-" * 70)

    return results


def build_seeds_from_exact(
    exact_results: dict[str, dict[str, list[str]]],
    feature_cols: list[str],
    n_seeds_per_method: int = 3,
) -> list[np.ndarray]:
    """
    Converte os resultados dos métodos exatos em indivíduos binários
    para serem usados como sementes no AG.

    Parâmetros
    ----------
    exact_results : saída de `run_exact_methods()`.
    feature_cols : lista de features usadas no AG (pós-filtragem).
    n_seeds_per_method : número máximo de sementes por método exato.

    Retorna
    -------
    Lista de arrays binários (indivíduos) prontos para inserção no AG.
    """
    seeds: list[np.ndarray] = []
    seen_keys: set[bytes] = set()

    for method_name, alg_features in exact_results.items():
        count = 0

        for alg_name, features in alg_features.items():
            if count >= n_seeds_per_method:
                break

            if len(features) == 0:
                continue

            ind = features_to_individual(features, feature_cols)

            # Ignorar sementes vazias (nenhuma feature correspondente)
            if ind.sum() == 0:
                continue

            # Evitar duplicatas
            key = np.packbits(ind).tobytes()
            if key in seen_keys:
                continue

            seeds.append(ind)
            seen_keys.add(key)
            count += 1

    print(f"Total de sementes únicas geradas: {len(seeds)}")
    return seeds


def run_hybrid_selection(
    X_df: pd.DataFrame,
    y: pd.Series,
    feature_cols: list[str],
    feature_ranking: np.ndarray,
    pipeline_factory,
    cv,
    ga_config: GAConfig,
    exact_results: dict[str, dict[str, list[str]]],
    n_seeds_per_method: int = 3,
    verbose: bool = True,
) -> GAResult:
    """
    Pipeline híbrido completo: converte soluções exatas em sementes,
    cria um AG semeado e executa a evolução.

    Parâmetros
    ----------
    X_df : DataFrame com meta-features (pós-filtragem).
    y : série com a coluna alvo (Best).
    feature_cols : lista de nomes de features.
    feature_ranking : ranking de features por importância.
    pipeline_factory : função que cria o pipeline de classificação.
    cv : objeto de validação cruzada.
    ga_config : configuração do AG.
    exact_results : saída de `run_exact_methods()`.
    n_seeds_per_method : sementes por método exato (padrão: 3).
    verbose : imprimir progresso.

    Retorna
    -------
    GAResult com o melhor resultado do AG híbrido.
    """
    if verbose:
        print("=" * 70)
        print(f"AG Híbrido: {ga_config.name}")
        print("=" * 70)

    # Converter resultados exatos em sementes binárias
    seeds = build_seeds_from_exact(
        exact_results, feature_cols,
        n_seeds_per_method=n_seeds_per_method,
    )

    if verbose:
        print(f"Sementes injetadas na população inicial: {len(seeds)}")
        for i, seed in enumerate(seeds):
            print(f"  Semente {i+1}: {int(seed.sum())} features ativas")
        print("-" * 70)

    # Criar e executar o seletor genético com sementes
    selector = GeneticFeatureSelector(
        X_df=X_df,
        y=y,
        feature_cols=feature_cols,
        feature_ranking=feature_ranking,
        pipeline_factory=pipeline_factory,
        cv=cv,
        config=ga_config,
        initial_seeds=seeds,
    )

    result = selector.fit(verbose=verbose)

    return result
