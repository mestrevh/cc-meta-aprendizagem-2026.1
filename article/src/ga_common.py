from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GAConfig:
    name: str = "AG"
    target_col: str = "Best"
    population_size: int = 40
    generations: int = 60
    elitism: int = 4
    tournament_size: int = 4
    crossover_rate: float = 0.85
    mutation_start: float = 0.08
    mutation_end: float = 0.015
    init_min_prob: float = 0.05
    init_max_prob: float = 0.30
    parsimony_weight: float = 0.08
    parsimony_power: float = 1.25
    stability_weight: float = 0.10
    cv_splits: int = 5
    scoring: str = "accuracy"
    patience: int = 15
    immigrant_rate: float = 0.12
    local_search_rounds: int = 4
    local_search_add_candidates: int = 25
    min_samples_per_class: int = 2
    hard_max_features: Optional[int] = None
    initialization_strategy: str = "guided"
    crossover_mode: str = "uniform"
    mutation_mode: str = "remove_bias"
    use_local_search: bool = True
    sparse_target_ratio: float = 0.12
    n_jobs: int = -1
    random_state: int = 42
    seed_source: str = ""  # Origem das sementes externas (ex: "exact_milp", "hybrid")


@dataclass(frozen=True)
class Score:
    fitness: float
    accuracy_mean: float
    accuracy_std: float
    n_features: int


@dataclass
class GAResult:
    name: str
    selected_features: list
    selected_mask: np.ndarray
    fitness: float
    accuracy_mean: float
    accuracy_std: float
    n_features: int
    history: pd.DataFrame
    model: object | None = None
    last_error: object | None = None
    config: GAConfig | None = None
