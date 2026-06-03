import warnings
import threading
from dataclasses import dataclass
from joblib import Parallel, delayed

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import cross_val_score

from ga_common import GAConfig, Score, GAResult


class GeneticFeatureSelector:
    def __init__(
        self,
        X_df,
        y,
        feature_cols,
        feature_ranking,
        pipeline_factory,
        cv,
        config,
    ):
        self.X_df = X_df
        self.y = y
        self.feature_cols = list(feature_cols)
        self.feature_ranking = np.asarray(feature_ranking)
        self.pipeline_factory = pipeline_factory
        self.cv = cv
        self.config = config
        self.rng = np.random.default_rng(config.random_state)
        self.n_total_features = len(feature_cols)
        self.fitness_cache = {}
        self.cache_lock = threading.Lock()
        self.last_error = None

        self.rank_position = np.empty(self.n_total_features, dtype=int)
        self.rank_position[self.feature_ranking] = np.arange(self.n_total_features)

    def individual_key(self, individual):
        return np.packbits(individual.astype(np.uint8)).tobytes()

    def repair(self, individual):
        individual = individual.astype(np.uint8, copy=True)

        if individual.sum() == 0:
            individual[self.feature_ranking[0]] = 1

        if self.config.hard_max_features is not None and individual.sum() > self.config.hard_max_features:
            selected = np.flatnonzero(individual)
            selected_sorted = selected[np.argsort(self.rank_position[selected])]
            keep = selected_sorted[: self.config.hard_max_features]
            individual[:] = 0
            individual[keep] = 1

        return individual

    def evaluate_individual(self, individual):
        individual = self.repair(individual)
        key = self.individual_key(individual)

        with self.cache_lock:
            cached = self.fitness_cache.get(key)

        if cached is not None:
            return cached

        selected_idx = np.flatnonzero(individual)
        n_features = len(selected_idx)

        if n_features == 0:
            score = Score(-1e9, 0.0, 0.0, 0)

            with self.cache_lock:
                self.fitness_cache[key] = score

            return score

        selected_cols = [self.feature_cols[i] for i in selected_idx]
        X_subset = self.X_df[selected_cols]

        try:
            model = self.pipeline_factory()

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                fold_scores = cross_val_score(
                    model,
                    X_subset,
                    self.y,
                    cv=self.cv,
                    scoring=self.config.scoring,
                    n_jobs=1,
                    error_score=0.0,
                )
            accuracy_mean = float(np.mean(fold_scores))
            accuracy_std = float(np.std(fold_scores))

            size_ratio = n_features / self.n_total_features
            size_penalty = self.config.parsimony_weight * (size_ratio ** self.config.parsimony_power)
            stability_penalty = self.config.stability_weight * accuracy_std

            fitness_value = accuracy_mean - size_penalty - stability_penalty

            score = Score(fitness=fitness_value, accuracy_mean=accuracy_mean, accuracy_std=accuracy_std, n_features=n_features)

        except Exception as exc:
            self.last_error = exc

            score = Score(fitness=-1e9, accuracy_mean=0.0, accuracy_std=0.0, n_features=n_features)

        with self.cache_lock:
            self.fitness_cache[key] = score

        return score

    def evaluate_population(self, population):
        return Parallel(n_jobs=self.config.n_jobs, prefer="threads")(delayed(self.evaluate_individual)(ind) for ind in population)

    def random_individual(self):
        prob = self.rng.uniform(self.config.init_min_prob, self.config.init_max_prob)
        individual = (self.rng.random(self.n_total_features) < prob).astype(np.uint8)
        return self.repair(individual)

    def top_k_individual(self, k):
        individual = np.zeros(self.n_total_features, dtype=np.uint8)
        k = max(1, min(k, self.n_total_features))
        individual[self.feature_ranking[:k]] = 1
        return self.repair(individual)

    def initialize_population(self):
        population = []
        seen = set()

        if self.config.initialization_strategy in ["guided", "mixed"]:
            seed_sizes = [
                1,
                2,
                3,
                5,
                8,
                13,
                max(1, self.n_total_features // 20),
                max(1, self.n_total_features // 10),
                max(1, self.n_total_features // 5),
            ]

            for k in seed_sizes:
                ind = self.top_k_individual(k)
                key = self.individual_key(ind)

                if key not in seen:
                    population.append(ind)
                    seen.add(key)

        attempts = 0
        max_attempts = self.config.population_size * 200

        while len(population) < self.config.population_size and attempts < max_attempts:
            ind = self.random_individual()
            key = self.individual_key(ind)

            if key not in seen:
                population.append(ind)
                seen.add(key)

            attempts += 1

        while len(population) < self.config.population_size:
            population.append(self.random_individual())

        return population[: self.config.population_size]

    def tournament_select(self, population, scores):

        candidates = self.rng.choice(len(population), size=min(self.config.tournament_size, len(population)), replace=False)

        best_idx = max(candidates, key=lambda idx: (scores[idx].fitness, scores[idx].accuracy_mean, -scores[idx].n_features))

        return population[best_idx].copy()

    def crossover(self, parent1, parent2):
        if self.rng.random() > self.config.crossover_rate:
            return parent1.copy(), parent2.copy()

        if self.config.crossover_mode == "one_point" and self.n_total_features > 1:
            point = self.rng.integers(1, self.n_total_features)

            child1 = np.concatenate([parent1[:point], parent2[point:]]).astype(np.uint8)
            child2 = np.concatenate([parent2[:point], parent1[point:]]).astype(np.uint8)

        else:
            mask = self.rng.random(self.n_total_features) < 0.5

            child1 = np.where(mask, parent1, parent2).astype(np.uint8)
            child2 = np.where(mask, parent2, parent1).astype(np.uint8)

        return self.repair(child1), self.repair(child2)

    def mutate(self, individual, generation):
        progress = generation / max(1, self.config.generations - 1)
        mutation_prob = self.config.mutation_start * (1.0 - progress) + self.config.mutation_end * progress

        individual = individual.copy()

        if self.config.mutation_mode == "flip":
            flip_mask = self.rng.random(self.n_total_features) < mutation_prob
            individual[flip_mask] = 1 - individual[flip_mask]

        elif self.config.mutation_mode == "sparse_adaptive":
            selected = individual == 1
            not_selected = individual == 0

            target_features = self.config.hard_max_features

            if target_features is None:
                target_features = max(1, int(self.n_total_features * self.config.sparse_target_ratio))

            current_features = int(individual.sum())

            if current_features > target_features:
                remove_factor = 2.00
                add_factor = 0.30
            else:
                remove_factor = 1.10
                add_factor = 0.70

            remove_mask = selected & (self.rng.random(self.n_total_features) < mutation_prob * remove_factor)
            add_mask = not_selected & (self.rng.random(self.n_total_features) < mutation_prob * add_factor)

            individual[remove_mask] = 0
            individual[add_mask] = 1

        else:
            selected = individual == 1
            not_selected = individual == 0

            remove_mask = selected & (self.rng.random(self.n_total_features) < mutation_prob * 1.25)
            add_mask = not_selected & (self.rng.random(self.n_total_features) < mutation_prob * 0.65)

            individual[remove_mask] = 0
            individual[add_mask] = 1

        return self.repair(individual)

    def is_better(self, new_score, old_score, eps=1e-8):
        if new_score.fitness > old_score.fitness + eps:
            return True

        if abs(new_score.fitness - old_score.fitness) <= eps:
            if new_score.accuracy_mean > old_score.accuracy_mean + eps:
                return True

            if abs(new_score.accuracy_mean - old_score.accuracy_mean) <= eps:
                return new_score.n_features < old_score.n_features

        return False

    def local_search(self, best_individual, best_score):
        current = best_individual.copy()
        current_score = best_score

        for _ in range(self.config.local_search_rounds):
            candidates = []
            selected = np.flatnonzero(current)

            if len(selected) > 1:
                for idx in selected:
                    candidate = current.copy()
                    candidate[idx] = 0
                    candidates.append(self.repair(candidate))

            add_pool = [idx for idx in self.feature_ranking if current[idx] == 0]
            add_pool = add_pool[: self.config.local_search_add_candidates]

            for idx in add_pool:
                candidate = current.copy()
                candidate[idx] = 1
                candidates.append(self.repair(candidate))

            if not candidates:
                break

            candidate_scores = self.evaluate_population(candidates)

            best_candidate_idx = max(
                range(len(candidates)), key=lambda i: (candidate_scores[i].fitness, candidate_scores[i].accuracy_mean, -candidate_scores[i].n_features)
            )

            candidate = candidates[best_candidate_idx]
            candidate_score = candidate_scores[best_candidate_idx]

            if self.is_better(candidate_score, current_score):
                current = candidate
                current_score = candidate_score
            else:
                break

        return current, current_score

    def fit(self, verbose=True):
        population = self.initialize_population()
        scores = self.evaluate_population(population)

        history = []

        best_global = None
        best_global_score = Score(-1e9, 0.0, 0.0, 0)
        no_improvement = 0

        if verbose:
            print("=" * 90)
            print(self.config.name)
            print("=" * 90)
            print(f"População: {self.config.population_size}")
            print(f"Gerações máximas: {self.config.generations}")
            print(f"Scoring: {self.config.scoring}")
            print(f"Inicialização: {self.config.initialization_strategy}")
            print(f"Crossover: {self.config.crossover_mode}")
            print(f"Mutação: {self.config.mutation_mode}")
            print(f"Busca local: {self.config.use_local_search}")
            print("-" * 90)

        for gen in range(self.config.generations):
            ranked_idx = sorted(
                range(len(population)),
                key=lambda i: (scores[i].fitness, scores[i].accuracy_mean, -scores[i].n_features),
                reverse=True,
            )

            current_best = population[ranked_idx[0]].copy()
            current_score = scores[ranked_idx[0]]

            if self.is_better(current_score, best_global_score):
                best_global = current_best.copy()
                best_global_score = current_score
                no_improvement = 0
            else:
                no_improvement += 1

            history.append({
                "generation": gen,
                "fitness": best_global_score.fitness,
                "accuracy_mean": best_global_score.accuracy_mean,
                "accuracy_std": best_global_score.accuracy_std,
                "n_features": best_global_score.n_features,
                "cache_size": len(self.fitness_cache),
            })

            if verbose:
                print(
                    f"Geração {gen:03d} | "
                        f"Acurácia: {best_global_score.accuracy_mean:.5f} ± {best_global_score.accuracy_std:.5f} | "
                    f"Features: {best_global_score.n_features}/{self.n_total_features} | "
                    f"Fitness: {best_global_score.fitness:.5f} | "
                    f"Sem melhora: {no_improvement}/{self.config.patience}"
                )

            if no_improvement >= self.config.patience:
                if verbose:
                    print("-" * 90)
                    print(f"Early stopping ativado na geração {gen}.")

                break

            new_population = [population[i].copy() for i in ranked_idx[: self.config.elitism]]

            immigrant_count = max(1, int(self.config.population_size * self.config.immigrant_rate))
            child_limit = self.config.population_size - immigrant_count

            while len(new_population) < child_limit:
                parent1 = self.tournament_select(population, scores)
                parent2 = self.tournament_select(population, scores)

                child1, child2 = self.crossover(parent1, parent2)

                child1 = self.mutate(child1, gen)
                child2 = self.mutate(child2, gen)

                new_population.append(child1)

                if len(new_population) < child_limit:
                    new_population.append(child2)

            while len(new_population) < self.config.population_size:
                new_population.append(self.random_individual())

            population = new_population[: self.config.population_size]
            scores = self.evaluate_population(population)

        if self.config.use_local_search:
            if verbose:
                print("-" * 90)
                print("Aplicando busca local final...")

            best_global, best_global_score = self.local_search(best_global, best_global_score)

        selected_idx = np.flatnonzero(best_global)
        final_features = [self.feature_cols[i] for i in selected_idx]

        if verbose:
            print("-" * 90)
            print("RESULTADO FINAL")
            print(f"Features selecionadas: {best_global_score.n_features}/{self.n_total_features}")
            print(f"Acurácia estimada: {best_global_score.accuracy_mean:.5f} ± {best_global_score.accuracy_std:.5f}")
            print(f"Fitness final: {best_global_score.fitness:.5f}")
            print("Atributos selecionados:")
            print(final_features)

            if self.last_error is not None and best_global_score.accuracy_mean == 0.0:
                print("Último erro capturado durante avaliação:")
                print(repr(self.last_error))

        return GAResult(
            name=self.config.name,
            selected_features=final_features,
            selected_mask=best_global,
            fitness=best_global_score.fitness,
            accuracy_mean=best_global_score.accuracy_mean,
            accuracy_std=best_global_score.accuracy_std,
            n_features=best_global_score.n_features,
            history=pd.DataFrame(history),
            last_error=self.last_error,
            config=self.config,
        )
