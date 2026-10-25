
"""
PHASES 17-19: STOCHASTIC AND ROBUST SHORTEST PATH RESEARCH

Phase 17: Mathematical correctness and reproducibility audit
Phase 18: Published-objective baselines on matched route pools
Phase 19: Five-city validation using existing Phase 13 GraphML networks

Existing network files:
    data/real_networks/phase13_pune.graphml
    data/real_networks/phase13_hyderabad.graphml
    data/real_networks/phase13_chennai.graphml
    data/real_networks/phase13_jaipur.graphml
    data/real_networks/phase13_kolkata.graphml

References:
    Rockafellar & Uryasev (2000), Optimization of Conditional Value-at-Risk
    DOI: 10.21314/jor.2000.038

    Bertsimas & Sim (2004), The Price of Robustness
    DOI: 10.1287/opre.1030.0065

IMPORTANT:
    These published objective formulations are optimized over finite
    candidate-route pools. This is NOT a reproduction of complete
    published routing algorithms or a global optimality claim.

    GraphML provides road topology, not observed traffic.
    Without observed traffic files, all uncertainty is simulated.

INSTALL:
    python -m pip install numpy pandas scipy networkx matplotlib

TEST:
    python run_phase_17.py --self-test

SMALL RUN:
    python run_phase_17.py --od-pairs 2 --budgets 3 \
        --dev-scenarios 200 --test-scenarios 500

FULL RUN:
    python run_phase_17.py --od-pairs 15 --budgets 3,6,10 \
        --dev-scenarios 1000 --test-scenarios 5000 \
        --gamma-budget 3

OPTIONAL:
    --exclude-od data/phase13_used_od_pairs.csv
    --observed-dir data/observed_traffic

Observed CSV schema, one file per city:
    timestamp,u,v,travel_seconds
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
import time

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from scipy.optimize import linprog
from scipy.special import ndtr
from scipy.stats import gamma as gamma_distribution


SEED = 20261014
ALPHA = 0.95

GRAPH_FILES = {
    "pune": "phase13_pune.graphml",
    "hyderabad": "phase13_hyderabad.graphml",
    "chennai": "phase13_chennai.graphml",
    "jaipur": "phase13_jaipur.graphml",
    "kolkata": "phase13_kolkata.graphml",
}

SHIFTS = (
    "iid_gamma",
    "correlated_gamma",
    "heavy_lognormal",
)

GENERATORS = (
    "mean_k",
    "random_diverse",
    "risk_informed",
)

METHODS = (
    "dijkstra",
    "sample_mean",
    "mean_std",
    "ru_cvar95",
    "bs_gamma_robust",
)

REFERENCES = {
    "ru_cvar95": {
        "authors": "Rockafellar and Uryasev",
        "year": 2000,
        "doi": "10.21314/jor.2000.038",
        "scope": "Empirical CVaR over finite candidate routes",
    },
    "bs_gamma_robust": {
        "authors": "Bertsimas and Sim",
        "year": 2004,
        "doi": "10.1287/opre.1030.0065",
        "scope": "Budgeted uncertainty over finite candidate routes",
    },
}


# ============================================================
# GENERAL UTILITIES
# ============================================================

def stable_seed(*parts):
    payload = "|".join(map(str, parts)).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def array_hash(array):
    data = np.ascontiguousarray(array)
    return hashlib.sha256(data.tobytes()).hexdigest()


def positive_number(value):
    try:
        number = float(value)
        if np.isfinite(number) and number > 0:
            return number
    except (ValueError, TypeError):
        pass
    return None


def coordinate_number(value):
    try:
        number = float(value)
        if np.isfinite(number):
            return number
    except (ValueError, TypeError):
        pass
    return None


def safe_markdown(frame):
    try:
        return frame.to_markdown(index=False)
    except ImportError:
        return frame.to_string(index=False)


# ============================================================
# PHASE 17: MATHEMATICAL CORRECTNESS
# ============================================================

def empirical_cvar(values, alpha=ALPHA):
    values = np.asarray(values, dtype=float).ravel()

    if not len(values) or not np.isfinite(values).all():
        raise ValueError("CVaR requires finite, nonempty losses")

    if not 0 < alpha < 1:
        raise ValueError("Invalid CVaR confidence level")

    eta = np.quantile(
        values,
        alpha,
        method="inverted_cdf",
    )

    return float(
        eta
        + np.maximum(values - eta, 0).mean() / (1 - alpha)
    )


def reference_cvar_lp(values, alpha=ALPHA):
    values = np.asarray(values, dtype=float).ravel()
    n = len(values)

    objective = np.concatenate([
        [1.0],
        np.full(n, 1.0 / ((1 - alpha) * n)),
    ])

    A = np.zeros((n, n + 1))
    A[:, 0] = -1
    A[np.arange(n), np.arange(n) + 1] = -1

    result = linprog(
        objective,
        A_ub=A,
        b_ub=-values,
        bounds=[(None, None)] + [(0, None)] * n,
        method="highs",
    )

    if not result.success:
        raise RuntimeError(result.message)

    return float(result.fun)


def bertsimas_sim_cost(means, deviations, gamma_budget):
    means = np.asarray(means, dtype=float)
    deviations = np.asarray(deviations, dtype=float)

    if len(means) != len(deviations):
        raise ValueError("Mismatched robust-cost arrays")

    if np.any(deviations < 0):
        raise ValueError("Negative uncertainty deviations")

    gamma_budget = float(
        np.clip(gamma_budget, 0, len(means))
    )

    deviations = np.sort(deviations)[::-1]

    whole = int(math.floor(gamma_budget))
    fractional = gamma_budget - whole

    premium = float(deviations[:whole].sum())

    if whole < len(deviations):
        premium += fractional * deviations[whole]

    return float(means.sum() + premium)


def scenario_matrix(mu, cv, zones, shift, n, rng):
    mu = np.asarray(mu, dtype=float)
    cv = np.asarray(cv, dtype=float)
    zones = np.asarray(zones)

    if np.any(mu <= 0) or np.any(cv <= 0):
        raise ValueError("Positive means and CVs required")

    m = len(mu)

    if shift == "iid_gamma":
        shape = 1.0 / cv**2

        return rng.gamma(
            shape=shape,
            scale=mu / shape,
            size=(n, m),
        )

    if shift == "correlated_gamma":
        unique_zones, zone_index = np.unique(
            zones,
            return_inverse=True,
        )

        loading = 0.75

        common = rng.standard_normal(
            (n, len(unique_zones))
        )

        independent = rng.standard_normal((n, m))

        latent = (
            loading * common[:, zone_index]
            + np.sqrt(1 - loading**2) * independent
        )

        probabilities = np.clip(
            ndtr(latent),
            1e-12,
            1 - 1e-12,
        )

        shape = 1.0 / cv**2

        return gamma_distribution.ppf(
            probabilities,
            a=shape,
            scale=mu / shape,
        )

    if shift == "heavy_lognormal":
        effective_cv = cv * 2.4

        sigma = np.sqrt(
            np.log1p(effective_cv**2)
        )

        z = rng.standard_normal((n, m))

        return mu * np.exp(
            sigma * z - 0.5 * sigma**2
        )

    raise ValueError(f"Unknown shift: {shift}")


def self_test():
    examples = [
        np.arange(1, 11, dtype=float),
        np.array([1, 1, 1, 5, 9], dtype=float),
        np.array([2, 3, 7, 12, 20, 40], dtype=float),
    ]

    for values in examples:
        for alpha in (0.5, 0.8, 0.95):
            direct = empirical_cvar(values, alpha)
            reference = reference_cvar_lp(values, alpha)

            assert np.isclose(
                direct, reference, atol=1e-7
            )

    assert np.isclose(
        bertsimas_sim_cost(
            [10, 20, 30],
            [2, 5, 8],
            1.5,
        ),
        70.5,
    )

    mu = np.array([10., 12., 15., 20.])
    cv = np.array([0.2, 0.3, 0.25, 0.4])
    zones = np.array([0, 0, 1, 1])

    first = scenario_matrix(
        mu, cv, zones,
        "correlated_gamma",
        30000,
        np.random.default_rng(123),
    )

    second = scenario_matrix(
        mu, cv, zones,
        "correlated_gamma",
        30000,
        np.random.default_rng(123),
    )

    assert np.array_equal(first, second)
    assert np.all(first > 0)

    assert np.allclose(
        first.mean(axis=0),
        mu,
        rtol=0.025,
    )

    assert np.allclose(
        first.std(axis=0) / first.mean(axis=0),
        cv,
        rtol=0.06,
    )

    correlation = np.corrcoef(first.T)

    assert correlation[0, 1] > correlation[0, 2]
    assert correlation[2, 3] > correlation[0, 2]

    dev = scenario_matrix(
        mu, cv, zones,
        "iid_gamma",
        100,
        np.random.default_rng(1000),
    )

    test = scenario_matrix(
        mu, cv, zones,
        "iid_gamma",
        100,
        np.random.default_rng(2000),
    )

    assert array_hash(dev) != array_hash(test)

    print(
        "PHASE 17 PASSED: CVaR LP, Bertsimas-Sim, "
        "Gamma marginals, correlation, seeds, leakage checks"
    )


# ============================================================
# PHASE 19: LOAD EXISTING PHASE 13 CITY NETWORKS
# ============================================================

def load_network(path, assumed_speed_kmh):
    """
    Read Phase 13 GraphML and construct a directed road graph.

    Parallel edges are collapsed to the lowest nominal travel time.
    Existing travel_time attributes take priority over speed assumptions.
    """

    original = nx.read_graphml(path)

    graph = nx.DiGraph()

    for node, attrs in original.nodes(data=True):
        graph.add_node(
            str(node),
            x=coordinate_number(attrs.get("x")),
            y=coordinate_number(attrs.get("y")),
        )

    skipped = 0
    used_travel_time = 0
    used_assumed_speed = 0

    for u, v, attrs in original.edges(data=True):
        travel_time = positive_number(
            attrs.get("travel_time")
        )

        length = positive_number(
            attrs.get("length")
        )

        if travel_time is not None:
            used_travel_time += 1

        if travel_time is None:
            if length is None:
                skipped += 1
                continue

            travel_time = (
                length / (assumed_speed_kmh * 1000 / 3600)
            )

            used_assumed_speed += 1

        u, v = str(u), str(v)

        if (
            not graph.has_edge(u, v)
            or travel_time < graph[u][v]["mean"]
        ):
            graph.add_edge(
                u,
                v,
                mean=float(travel_time),
                length=float(length or 0),
            )

    if graph.number_of_edges() == 0:
        raise ValueError(
            f"No usable road edges in {path}. "
            "Inspect GraphML edge attributes."
        )

    largest = max(
        nx.strongly_connected_components(graph),
        key=len,
    )

    graph = graph.subgraph(largest).copy()

    valid_coordinates = sum(
        attrs.get("x") is not None
        for _, attrs in graph.nodes(data=True)
    )

    print(f"\nLoaded: {path.name}")
    print(f"  Nodes: {graph.number_of_nodes():,}")
    print(f"  Edges: {graph.number_of_edges():,}")
    print(f"  Edges with travel_time: {used_travel_time:,}")
    print(f"  Edges using assumed speed: {used_assumed_speed:,}")
    print(f"  Skipped unusable edges: {skipped:,}")
    print(f"  Nodes with longitude: {valid_coordinates:,}")

    if valid_coordinates < 100:
        raise ValueError(
            f"{path.name} has insufficient longitude coordinates. "
            "Check whether Phase 13 uses x/y or lon/lat."
        )

    return graph


def load_city_graphs(graph_dir, assumed_speed_kmh):
    graph_dir = Path(graph_dir)

    graphs = {}

    for city, filename in GRAPH_FILES.items():
        path = graph_dir / filename

        if not path.exists():
            raise FileNotFoundError(
                f"\nMissing Phase 13 network: {path}\n"
                f"Expected: {path.resolve()}\n"
                "Use --graphs-dir data/real_networks"
            )

        graphs[city] = load_network(
            path,
            assumed_speed_kmh,
        )

    return graphs


def geographic_holdout_nodes(graph, fraction=0.25):
    valid = [
        (node, attrs["x"])
        for node, attrs in graph.nodes(data=True)
        if attrs.get("x") is not None
    ]

    if len(valid) < 100:
        raise ValueError(
            "Geographic holdout requires longitude coordinates"
        )

    threshold = np.quantile(
        [longitude for _, longitude in valid],
        1 - fraction,
    )

    return [
        node
        for node, longitude in valid
        if longitude >= threshold
    ]


def read_excluded_pairs(path, city):
    if not path:
        return set()

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"OD exclusion file not found: {path}"
        )

    frame = pd.read_csv(path)

    required = {"city", "source", "target"}

    if not required.issubset(frame.columns):
        raise ValueError(
            "Exclusion CSV requires city,source,target"
        )

    frame = frame[
        frame["city"].astype(str).str.lower() == city
    ]

    return set(
        zip(
            frame["source"].astype(str),
            frame["target"].astype(str),
        )
    )


def choose_od_pairs(graph, city, count, seed, excluded):
    rng = np.random.default_rng(seed)

    nodes = geographic_holdout_nodes(graph)

    selected = []
    seen = set(excluded)

    attempts = 0
    max_attempts = max(2000, count * 300)

    while len(selected) < count and attempts < max_attempts:
        attempts += 1

        indices = rng.choice(
            len(nodes),
            size=2,
            replace=False,
        )

        source = nodes[int(indices[0])]
        target = nodes[int(indices[1])]

        key = (str(source), str(target))

        if key in seen:
            continue

        seen.add(key)

        try:
            route = nx.shortest_path(
                graph,
                source,
                target,
                weight="mean",
            )
        except nx.NetworkXNoPath:
            continue

        if len(route) < 8:
            continue

        selected.append((source, target))

    if len(selected) < count:
        raise RuntimeError(
            f"{city}: only {len(selected)}/{count} OD pairs found"
        )

    return selected


# ============================================================
# PHASE 18: CANDIDATE ROUTE GENERATION
# ============================================================

def k_shortest(graph, source, target, weight, count):
    try:
        iterator = nx.shortest_simple_paths(
            graph,
            source,
            target,
            weight=weight,
        )

        return [
            tuple(path)
            for path in itertools.islice(iterator, count)
        ]

    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []


def generate_candidates(
    graph,
    source,
    target,
    generator,
    budget,
    rng,
):
    baseline = k_shortest(
        graph,
        source,
        target,
        "mean",
        budget,
    )

    if generator == "mean_k":
        return baseline

    paths = []
    seen = set()

    def add(new_paths):
        for path in new_paths:
            if path not in seen and len(paths) < budget:
                seen.add(path)
                paths.append(path)

    add(baseline[:1])

    corridor = set()

    for path in baseline:
        corridor.update(
            zip(path[:-1], path[1:])
        )

    if generator == "random_diverse":
        for _ in range(max(12, budget * 5)):
            factors = {
                edge: float(rng.lognormal(0, 0.6))
                for edge in corridor
            }

            def weight(u, v, attrs):
                return (
                    attrs["mean"]
                    * factors.get((u, v), 1.0)
                )

            add(
                k_shortest(
                    graph,
                    source,
                    target,
                    weight,
                    min(2, budget),
                )
            )

            if len(paths) >= budget:
                break

    elif generator == "risk_informed":
        for multiplier in (0.5, 1, 2, 4, 8):

            def weight(u, v, attrs, m=multiplier):
                return (
                    attrs["mean"]
                    * (1 + m * attrs["cv"])
                )

            add(
                k_shortest(
                    graph,
                    source,
                    target,
                    weight,
                    budget,
                )
            )

            if len(paths) >= budget:
                break

    else:
        raise ValueError(generator)

    add(baseline)

    return paths


def candidate_diversity(candidates):
    if len(candidates) < 2:
        return 0.0

    sets = [
        set(zip(path[:-1], path[1:]))
        for path in candidates
    ]

    return float(
        np.mean([
            1 - len(a & b) / len(a | b)
            for a, b in itertools.combinations(sets, 2)
        ])
    )


def edge_data_for_candidates(graph, candidates):
    edge_set = set()

    for path in candidates:
        edge_set.update(
            zip(path[:-1], path[1:])
        )

    edges = sorted(edge_set)
    lookup = {edge: i for i, edge in enumerate(edges)}

    mu = np.array([
        graph[u][v]["mean"]
        for u, v in edges
    ])

    cv = np.array([
        graph[u][v]["cv"]
        for u, v in edges
    ])

    zones = np.array([
        graph[u][v]["zone"]
        for u, v in edges
    ])

    matrix = np.zeros(
        (len(candidates), len(edges)),
        dtype=float,
    )

    for i, path in enumerate(candidates):
        for u, v in zip(path[:-1], path[1:]):
            matrix[i, lookup[(u, v)]] = 1

    return edges, mu, cv, zones, matrix


# ============================================================
# PHASE 18: ROUTE SELECTION METHODS
# ============================================================

def choose_method(
    method,
    development,
    route_means,
    matrix,
    mu,
    cv,
    gamma_budget,
):
    if method == "dijkstra":
        scores = route_means

    elif method == "sample_mean":
        scores = development.mean(axis=0)

    elif method == "mean_std":
        scores = (
            development.mean(axis=0)
            + 0.4 * development.std(axis=0)
        )

    elif method == "ru_cvar95":
        scores = np.array([
            empirical_cvar(development[:, j])
            for j in range(development.shape[1])
        ])

    elif method == "bs_gamma_robust":
        scores = []

        for incidence in matrix:
            indices = np.flatnonzero(incidence)

            scores.append(
                bertsimas_sim_cost(
                    mu[indices],
                    mu[indices] * cv[indices],
                    gamma_budget,
                )
            )

        scores = np.asarray(scores)

    else:
        raise ValueError(method)

    return (
        int(np.argmin(scores)),
        float(np.min(scores)),
    )


# ============================================================
# OPTIONAL OBSERVED TRAFFIC DATA
# ============================================================

def load_observed(city, directory):
    if not directory:
        return None

    path = Path(directory) / f"{city}.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Observed traffic file missing: {path}"
        )

    frame = pd.read_csv(path)

    required = {
        "timestamp",
        "u",
        "v",
        "travel_seconds",
    }

    if not required.issubset(frame.columns):
        raise ValueError(
            f"{path} requires columns {sorted(required)}"
        )

    frame["u"] = frame["u"].astype(str)
    frame["v"] = frame["v"].astype(str)

    frame["travel_seconds"] = pd.to_numeric(
        frame["travel_seconds"],
        errors="coerce",
    )

    frame = frame[
        np.isfinite(frame["travel_seconds"])
        & (frame["travel_seconds"] > 0)
    ].copy()

    frame["edge"] = list(
        zip(frame["u"], frame["v"])
    )

    if frame.duplicated(
        ["timestamp", "edge"]
    ).any():
        raise ValueError(
            f"Duplicate timestamp-edge measurements: {path}"
        )

    return frame


def observed_scenarios(frame, edges, minimum):
    subset = frame[
        frame["edge"].isin(set(edges))
    ]

    pivot = subset.pivot(
        index="timestamp",
        columns="edge",
        values="travel_seconds",
    )

    missing = [
        edge
        for edge in edges
        if edge not in pivot.columns
    ]

    if missing:
        raise ValueError(
            f"Missing observed measurements for "
            f"{len(missing)} route edges"
        )

    pivot = (
        pivot.reindex(columns=edges)
        .dropna()
        .sort_index()
    )

    if len(pivot) < minimum:
        raise ValueError(
            f"Only {len(pivot)} complete timestamps; "
            f"at least {minimum} required"
        )

    return pivot.to_numpy(dtype=float)


# ============================================================
# PHASE 19: CITY EXPERIMENT
# ============================================================

def prepare_uncertainty(graph, city, seed):
    """
    Assign synthetic uncertainty once per graph,
    independently of development and test scenarios.
    """

    for u, v, attrs in graph.edges(data=True):
        edge_rng = np.random.default_rng(
            stable_seed(seed, city, u, v)
        )

        attrs["cv"] = float(
            edge_rng.uniform(0.12, 0.42)
        )

        longitude = graph.nodes[u].get("x")

        attrs["zone"] = (
            int(longitude * 10)
            if longitude is not None
            else 0
        )


def run_city(city, graph, args, observed):
    print(f"\n{'=' * 60}")
    print(f"PHASE 19: {city.upper()}")
    print(f"{'=' * 60}")

    prepare_uncertainty(graph, city, args.seed)

    excluded = read_excluded_pairs(
        args.exclude_od,
        city,
    )

    pairs = choose_od_pairs(
        graph,
        city,
        args.od_pairs,
        stable_seed(args.seed, city, "od"),
        excluded,
    )

    records = []
    diagnostics = []
    od_records = []

    for od_index, (source, target) in enumerate(pairs):
        od_records.append({
            "city": city,
            "od": od_index,
            "source": str(source),
            "target": str(target),
        })

        # Build the union of all candidate-route edges first.
        # All generators and budgets for this OD then use the
        # same development/test edge scenarios on that union.

        pool_specs = []
        union_edges = set()

        for budget in args.budgets:
            for generator in GENERATORS:
                start = time.perf_counter()

                candidates = generate_candidates(
                    graph,
                    source,
                    target,
                    generator,
                    budget,
                    np.random.default_rng(
                        stable_seed(
                            args.seed,
                            city,
                            od_index,
                            budget,
                            generator,
                        )
                    ),
                )

                generation_seconds = (
                    time.perf_counter() - start
                )

                if not candidates:
                    continue

                pool_specs.append((
                    budget,
                    generator,
                    candidates,
                    generation_seconds,
                ))

                for path in candidates:
                    union_edges.update(
                        zip(path[:-1], path[1:])
                    )

        if not pool_specs:
            raise RuntimeError(
                f"{city} OD {od_index}: no candidate routes"
            )

        edges = sorted(union_edges)
        lookup = {
            edge: i
            for i, edge in enumerate(edges)
        }

        mu = np.array([
            graph[u][v]["mean"]
            for u, v in edges
        ])

        cv = np.array([
            graph[u][v]["cv"]
            for u, v in edges
        ])

        zones = np.array([
            graph[u][v]["zone"]
            for u, v in edges
        ])

        datasets = []

        if observed is not None:
            all_observations = observed_scenarios(
                observed,
                edges,
                args.observed_min,
            )

            cut = len(all_observations) // 2

            datasets.append((
                "observed_chronological",
                all_observations[:cut],
                all_observations[cut:],
            ))

        else:
            for shift in SHIFTS:
                dev_edges = scenario_matrix(
                    mu,
                    cv,
                    zones,
                    shift,
                    args.dev_scenarios,
                    np.random.default_rng(
                        stable_seed(
                            args.seed,
                            city,
                            od_index,
                            shift,
                            "dev",
                        )
                    ),
                )

                test_edges = scenario_matrix(
                    mu,
                    cv,
                    zones,
                    shift,
                    args.test_scenarios,
                    np.random.default_rng(
                        stable_seed(
                            args.seed,
                            city,
                            od_index,
                            shift,
                            "test",
                        )
                    ),
                )

                datasets.append((
                    shift,
                    dev_edges,
                    test_edges,
                ))

        for budget, generator, candidates, generation_seconds in pool_specs:
            matrix = np.zeros(
                (len(candidates), len(edges)),
                dtype=float,
            )

            for i, path in enumerate(candidates):
                for u, v in zip(path[:-1], path[1:]):
                    matrix[i, lookup[(u, v)]] = 1

            route_means = matrix @ mu

            diagnostics.append({
                "city": city,
                "od": od_index,
                "generator": generator,
                "budget": budget,
                "actual_candidates": len(candidates),
                "diversity": candidate_diversity(candidates),
                "generation_seconds": generation_seconds,
            })

            for shift, dev_edges, test_edges in datasets:
                dev_routes = dev_edges @ matrix.T
                test_routes = test_edges @ matrix.T

                dev_hash = array_hash(dev_edges)
                test_hash = array_hash(test_edges)

                if dev_hash == test_hash:
                    raise AssertionError(
                        "Development/test scenario leakage"
                    )

                for method in METHODS:
                    start = time.perf_counter()

                    selected, objective = choose_method(
                        method,
                        dev_routes,
                        route_means,
                        matrix,
                        mu,
                        cv,
                        args.gamma_budget,
                    )

                    selection_seconds = (
                        time.perf_counter() - start
                    )

                    outcomes = test_routes[:, selected]

                    records.append({
                        "city": city,
                        "od": od_index,
                        "source": str(source),
                        "target": str(target),
                        "shift": shift,
                        "generator": generator,
                        "budget": budget,
                        "actual_candidates": len(candidates),
                        "method": method,
                        "chosen_candidate": selected,
                        "chosen_route": "->".join(
                            candidates[selected]
                        ),
                        "development_objective": objective,
                        "test_mean": float(outcomes.mean()),
                        "test_cvar95": empirical_cvar(outcomes),
                        "test_p95": float(
                            np.quantile(outcomes, 0.95)
                        ),
                        "test_std": float(outcomes.std()),
                        "generation_seconds": generation_seconds,
                        "selection_seconds": selection_seconds,
                        "total_seconds": (
                            generation_seconds + selection_seconds
                        ),
                        "dev_sha256": dev_hash,
                        "test_sha256": test_hash,
                        "evaluation_type": (
                            "observed"
                            if observed is not None
                            else "synthetic_on_osm"
                        ),
                    })

        print(
            f"  {city}: OD {od_index + 1}/{len(pairs)} "
            f"| cumulative records: {len(records)}",
            flush=True,
        )

    return (
        pd.DataFrame(records),
        pd.DataFrame(diagnostics),
        pd.DataFrame(od_records),
    )


# ============================================================
# STATISTICAL ANALYSIS
# ============================================================

def paired_analysis(results):
    keys = [
        "city",
        "od",
        "shift",
        "generator",
        "budget",
    ]

    reference = results[
        results["method"] == "dijkstra"
    ][
        keys + [
            "test_cvar95",
            "test_mean",
            "chosen_route",
        ]
    ].rename(columns={
        "test_cvar95": "reference_cvar95",
        "test_mean": "reference_mean",
        "chosen_route": "reference_route",
    })

    paired = results.merge(
        reference,
        on=keys,
        validate="many_to_one",
    )

    paired["delta_cvar95"] = (
        paired["test_cvar95"]
        - paired["reference_cvar95"]
    )

    paired["delta_mean"] = (
        paired["test_mean"]
        - paired["reference_mean"]
    )

    paired["route_changed"] = (
        paired["chosen_route"]
        != paired["reference_route"]
    )

    return paired


def exact_signflip_pvalue(values):
    values = np.asarray(values, dtype=float)
    n = len(values)

    if n < 2:
        return np.nan

    if n > 18:
        raise ValueError(
            "Exact sign-flip restricted to <=18 clusters"
        )

    observed = abs(values.mean())

    null_values = []

    for signs in itertools.product(
        (-1, 1),
        repeat=n,
    ):
        null_values.append(
            abs(
                np.mean(
                    values * np.asarray(signs)
                )
            )
        )

    return float(
        np.mean(
            np.asarray(null_values)
            >= observed - 1e-12
        )
    )


def holm_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)

    adjusted = np.full(len(p), np.nan)

    valid = np.flatnonzero(np.isfinite(p))

    if not len(valid):
        return adjusted

    order = valid[np.argsort(p[valid])]
    m = len(order)
    running = 0.0

    for rank, index in enumerate(order):
        running = max(
            running,
            (m - rank) * p[index],
        )

        adjusted[index] = min(1.0, running)

    return adjusted


def statistical_analysis(paired):
    grouped = paired.groupby(
        ["shift", "method"],
        as_index=False,
    ).agg(
        mean_delta_cvar95=("delta_cvar95", "mean"),
        mean_delta_mean=("delta_mean", "mean"),
        route_change_rate=("route_changed", "mean"),
        mean_runtime_s=("total_seconds", "mean"),
        n_records=("delta_cvar95", "size"),
    )

    tests = []

    for (shift, method), subset in paired.groupby(
        ["shift", "method"]
    ):
        city_effects = (
            subset.groupby("city")["delta_cvar95"]
            .mean()
            .to_numpy()
        )

        tests.append({
            "shift": shift,
            "method": method,
            "n_cities": len(city_effects),
            "mean_city_effect": float(
                city_effects.mean()
            ),
            "exact_signflip_p": exact_signflip_pvalue(
                city_effects
            ),
        })

    tests = pd.DataFrame(tests)

    tests["holm_p"] = holm_adjust(
        tests["exact_signflip_p"]
    )

    return grouped, tests


# ============================================================
# FIGURES AND REPORT
# ============================================================

def plot_results(summary, output):
    for shift, subset in summary.groupby("shift"):
        data = subset[
            subset["method"] != "dijkstra"
        ]

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.bar(
            data["method"],
            data["mean_delta_cvar95"],
        )

        ax.axhline(
            0,
            color="black",
            linewidth=0.8,
        )

        ax.set(
            title=f"{shift}: CVaR95 relative to Dijkstra",
            ylabel="Paired CVaR95 difference (seconds)",
            xlabel="Routing method",
        )

        ax.tick_params(
            axis="x",
            rotation=20,
        )

        fig.tight_layout()

        fig.savefig(
            output / f"comparison_{shift}.png",
            dpi=200,
        )

        plt.close(fig)


def write_report(summary, tests, manifest, output):
    lines = [
        "# Phase 17-19 Research Validation Report",
        "",
        "## Experimental status",
        "",
        f"Cities: {', '.join(manifest['cities'])}",
        f"OD pairs per city: {manifest['od_pairs_per_city']}",
        f"Data mode: {manifest['data_mode']}",
        f"Network directory: {manifest['graphs_dir']}",
        "",
        "## Mathematical formulations",
        "",
        "Rockafellar-Uryasev empirical CVaR95:",
        "",
        "min_eta eta + mean(max(L - eta, 0)) / 0.05",
        "",
        "Bertsimas-Sim budgeted robust route cost:",
        "",
        "sum(mu) + largest Gamma-budget deviations.",
        "",
        "Both formulations are optimized over finite candidate pools.",
        "",
        "## Statistical interpretation",
        "",
        "Negative paired CVaR95 differences indicate improvement.",
        "Tests use city-level aggregates.",
        "Holm correction adjusts the reported comparison family.",
        "Five city clusters provide limited inferential power.",
        "",
        "## Method comparison",
        "",
        safe_markdown(summary),
        "",
        "## City-level tests",
        "",
        safe_markdown(tests),
        "",
        "## Important limitations",
        "",
        "- OSM topology is not observed traffic.",
        "- Synthetic edge CVs are modeling assumptions.",
        "- Existing Phase 13 graphs are reused.",
        "- New OD pairs are not guaranteed untouched unless an exclusion CSV is supplied.",
        "- Geographic holdout does not establish unseen road topology.",
        "- Published objectives are used, not full published routing algorithms.",
        "- Candidate generation can be expensive on large city networks.",
        "- Finite candidate pools do not guarantee global optimality.",
        "- Five-city statistical tests have limited power.",
    ]

    (output / "RESEARCH_REPORT.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


# ============================================================
# MAIN EXECUTION
# ============================================================

def execute(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    graph_dir = Path(args.graphs_dir)

    print("\nUsing existing Phase 13 road networks")
    print("Network directory:", graph_dir.resolve())

    graphs = load_city_graphs(
        graph_dir,
        args.assumed_speed_kmh,
    )

    all_results = []
    all_diagnostics = []
    all_od_pairs = []

    started = time.perf_counter()

    for city, graph in graphs.items():
        observed = load_observed(
            city,
            args.observed_dir,
        )

        result, diagnostic, od_pairs = run_city(
            city,
            graph,
            args,
            observed,
        )

        all_results.append(result)
        all_diagnostics.append(diagnostic)
        all_od_pairs.append(od_pairs)

        # Save intermediate city outputs for recovery.
        result.to_csv(
            output / f"{city}_results.csv",
            index=False,
        )

        print(f"Completed city: {city}")

    results = pd.concat(
        all_results,
        ignore_index=True,
    )

    diagnostics = pd.concat(
        all_diagnostics,
        ignore_index=True,
    )

    od_pairs = pd.concat(
        all_od_pairs,
        ignore_index=True,
    )

    results.to_csv(
        output / "all_method_results.csv",
        index=False,
    )

    diagnostics.to_csv(
        output / "candidate_pool_diagnostics.csv",
        index=False,
    )

    od_pairs.to_csv(
        output / "phase19_od_pairs.csv",
        index=False,
    )

    paired = paired_analysis(results)

    paired.to_csv(
        output / "paired_effects.csv",
        index=False,
    )

    summary, tests = statistical_analysis(paired)

    summary.to_csv(
        output / "summary.csv",
        index=False,
    )

    tests.to_csv(
        output / "city_statistical_tests.csv",
        index=False,
    )

    plot_results(summary, output)

    manifest = {
        "seed": args.seed,
        "cities": list(GRAPH_FILES),
        "graph_files": GRAPH_FILES,
        "graphs_dir": str(graph_dir.resolve()),
        "od_pairs_per_city": args.od_pairs,
        "budgets": args.budgets,
        "methods": METHODS,
        "generators": GENERATORS,
        "shifts": SHIFTS,
        "alpha": ALPHA,
        "gamma_budget": args.gamma_budget,
        "data_mode": (
            "observed"
            if args.observed_dir
            else "synthetic_on_osm"
        ),
        "excluded_od_file": args.exclude_od,
        "records": len(results),
        "duration_seconds": round(
            time.perf_counter() - started,
            3,
        ),
        "python": platform.python_version(),
        "networkx": nx.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "references": REFERENCES,
        "methodological_limits": [
            "Finite candidate optimization",
            "No global robust shortest-path optimality guarantee",
            "OSM topology is not measured traffic",
            "Synthetic CV assumptions",
            "Untouched Phase 13 pairs require exclusion CSV",
            "Only five independent city clusters",
        ],
    }

    (output / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    write_report(
        summary,
        tests,
        manifest,
        output,
    )

    print("\n" + "=" * 60)
    print("PHASES 17-19 COMPLETED")
    print("=" * 60)
    print("Output:", output.resolve())
    print("Total records:", len(results))
    print("Cities:", len(GRAPH_FILES))
    print("OD pairs:", len(od_pairs))
    print("Figures:", len(list(output.glob("*.png"))))


def main():
    parser = argparse.ArgumentParser(
        description="Phases 17-19: five-city routing research"
    )

    parser.add_argument(
        "--graphs-dir",
        default="data/real_networks",
    )

    parser.add_argument(
        "--output",
        default="data/results/phase17_19_full",
    )

    parser.add_argument(
        "--observed-dir",
        default=None,
    )

    parser.add_argument(
        "--exclude-od",
        default=None,
    )

    parser.add_argument(
        "--self-test",
        action="store_true",
    )

    parser.add_argument(
        "--od-pairs",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--budgets",
        default="3,6,10",
    )

    parser.add_argument(
        "--dev-scenarios",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--test-scenarios",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--gamma-budget",
        type=float,
        default=3.0,
    )

    parser.add_argument(
        "--assumed-speed-kmh",
        type=float,
        default=30.0,
    )

    parser.add_argument(
        "--observed-min",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )

    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    args.budgets = sorted(set(
        int(value.strip())
        for value in args.budgets.split(",")
    ))

    if (
        not args.budgets
        or min(args.budgets) < 1
        or args.od_pairs < 1
        or args.assumed_speed_kmh <= 0
        or args.gamma_budget < 0
        or args.dev_scenarios < 1
        or args.test_scenarios < 1
    ):
        parser.error("Invalid experiment parameters")

    self_test()
    execute(args)


if __name__ == "__main__":
    main()
