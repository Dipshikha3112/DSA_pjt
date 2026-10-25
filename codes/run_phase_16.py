
"""Phase 16.1–16.4: corrected exploratory stochastic-routing benchmark.

This script runs a controlled synthetic-network experiment. It is not a
reproduction of a specific published algorithm or a validation on measured
traffic. Results must not be described as demonstrating universal superiority.

Run:
    python run_phase_16.py --self-test
    python run_phase_16.py --output data/results/phase16_corrected \
        --graphs 30 --nodes 55 --budgets 3,6,10 \
        --dev-scenarios 1000 --test-scenarios 5000 --bootstrap 3000
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.stats import gamma as gamma_dist

SEED = 20261014
ALPHA = 0.95

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

SELECTORS = (
    "known_mean",
    "sample_mean",
    "empirical_cvar95",
    "mean_variance",
    "worst_model_cvar95",
)


def cvar(values, alpha=ALPHA):
    """Exact empirical CVaR for equally weighted observations."""
    x = np.asarray(values, dtype=float).reshape(-1)

    if x.size == 0 or not np.isfinite(x).all():
        raise ValueError("CVaR requires finite, nonempty observations")

    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between 0 and 1")

    eta = float(np.quantile(x, alpha, method="inverted_cdf"))

    return float(
        eta + np.maximum(x - eta, 0).mean() / (1 - alpha)
    )


def graph_instance(seed, n):
    """Create a directed synthetic geometric network."""
    rng = np.random.default_rng(seed)

    positions = rng.random((n, 2))
    order = np.argsort(positions[:, 0])

    graph = nx.DiGraph()

    for node in range(n):
        graph.add_node(node, xy=positions[node].tolist())

    edges = set()

    for a, b in zip(order[:-1], order[1:]):
        edges.add((int(a), int(b)))
        edges.add((int(b), int(a)))

    for i in range(n):
        distances = np.linalg.norm(
            positions - positions[i], axis=1
        )

        for j in np.argsort(distances)[1:6]:
            if rng.random() < 0.76:
                edges.add((int(i), int(j)))

    for u, v in sorted(edges):
        length = float(
            np.linalg.norm(positions[u] - positions[v])
        )

        mean = float(
            5 + 55 * length + rng.uniform(0, 3)
        )

        cv = float(rng.uniform(0.12, 0.42))

        zone = int(
            (positions[u, 0] + positions[v, 0]) * 2
        )

        graph.add_edge(
            u, v,
            mean=mean,
            cv=cv,
            zone=zone,
        )

    return graph, int(order[0]), int(order[-1])


def edges_for(graph):
    edges = sorted(graph.edges())
    lookup = {edge: i for i, edge in enumerate(edges)}

    mu = np.array(
        [graph[u][v]["mean"] for u, v in edges],
        dtype=float,
    )

    cv = np.array(
        [graph[u][v]["cv"] for u, v in edges],
        dtype=float,
    )

    zone = np.array(
        [graph[u][v]["zone"] for u, v in edges],
        dtype=int,
    )

    return edges, lookup, mu, cv, zone


def scenario_matrix(mu, cv, zone, shift, n, rng):
    """Generate positive edge costs with prescribed marginal means.

    Correlated Gamma uses a Gaussian copula with Gamma marginals.
    The copula parameter is a latent-normal correlation, not the
    resulting Pearson correlation of Gamma travel times.
    """
    m = len(mu)

    if shift == "iid_gamma":
        shape = 1.0 / np.square(cv)

        return rng.gamma(
            shape=shape,
            scale=mu / shape,
            size=(n, m),
        )

    if shift == "correlated_gamma":
        zones = np.unique(zone)
        zidx = np.searchsorted(zones, zone)

        rho = 0.75

        common = rng.standard_normal(
            (n, len(zones))
        )

        independent = rng.standard_normal((n, m))

        latent = (
            rho * common[:, zidx]
            + math.sqrt(1 - rho**2) * independent
        )

        probabilities = np.clip(
            ndtr(latent),
            1e-12,
            1 - 1e-12,
        )

        shape = 1.0 / np.square(cv)

        samples = gamma_dist.ppf(
            probabilities,
            a=shape,
            scale=mu / shape,
        )

        return np.asarray(samples, dtype=float)

    if shift == "heavy_lognormal":
        effective_cv = cv * 2.4

        sigma = np.sqrt(
            np.log1p(np.square(effective_cv))
        )

        z = rng.standard_normal((n, m))

        return mu * np.exp(
            sigma * z - 0.5 * sigma**2
        )

    raise ValueError(f"Unknown shift: {shift}")


def path_ids(path, lookup):
    return tuple(
        lookup[(u, v)]
        for u, v in zip(path[:-1], path[1:])
    )


def k_paths(graph, source, target, weight, k):
    try:
        iterator = nx.shortest_simple_paths(
            graph,
            source,
            target,
            weight=weight,
        )

        return [
            tuple(path)
            for path in itertools.islice(iterator, k)
        ]

    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []


def generate_candidates(
    graph,
    source,
    target,
    mu,
    cv,
    edges,
    kind,
    budget,
    rng,
):
    """Generate up to budget distinct paths.

    Candidate count is matched where feasible. Randomized search has
    a bounded search effort; actual counts are recorded separately.
    """
    baseline = k_paths(
        graph, source, target, "mean", budget
    )

    if kind == "mean_k":
        return baseline

    seen = set()
    result = []

    def add(paths):
        for path in paths:
            if path not in seen and len(result) < budget:
                seen.add(path)
                result.append(path)

    add(baseline[:1])

    if kind == "random_diverse":
        for _ in range(max(15, budget * 12)):
            factors = rng.lognormal(
                0, 0.65, size=len(edges)
            )

            for i, (u, v) in enumerate(edges):
                graph[u][v]["temporary_weight"] = float(
                    mu[i] * factors[i]
                )

            add(
                k_paths(
                    graph,
                    source,
                    target,
                    "temporary_weight",
                    min(2, budget),
                )
            )

            if len(result) >= budget:
                break

    elif kind == "risk_informed":
        for multiplier in (
            0.5, 1, 1.5, 2, 3, 4, 6, 8
        ):
            for i, (u, v) in enumerate(edges):
                graph[u][v]["temporary_weight"] = float(
                    mu[i] * (1 + multiplier * cv[i])
                )

            add(
                k_paths(
                    graph,
                    source,
                    target,
                    "temporary_weight",
                    budget,
                )
            )

            if len(result) >= budget:
                break

    else:
        raise ValueError(kind)

    add(baseline)

    return result


def route_matrix(paths, lookup, n_edges):
    matrix = np.zeros(
        (len(paths), n_edges),
        dtype=np.float64,
    )

    for i, path in enumerate(paths):
        matrix[i, list(path_ids(path, lookup))] = 1

    return matrix


def path_diversity(paths):
    if len(paths) < 2:
        return 0.0

    edge_sets = [
        set(zip(path[:-1], path[1:]))
        for path in paths
    ]

    distances = []

    for a, b in itertools.combinations(edge_sets, 2):
        distances.append(
            1 - len(a & b) / len(a | b)
        )

    return float(np.mean(distances))


def select_route(
    selector,
    development_models,
    known_route_means,
    active_shift,
):
    """Select using the same available development models.

    All selectors receive the same information object.
    Different objectives may use different subsets of it.
    """
    current = development_models[active_shift]

    if selector == "known_mean":
        scores = known_route_means

    elif selector == "sample_mean":
        scores = current.mean(axis=0)

    elif selector == "empirical_cvar95":
        scores = np.array([
            cvar(current[:, j])
            for j in range(current.shape[1])
        ])

    elif selector == "mean_variance":
        scores = (
            current.mean(axis=0)
            + 0.4 * current.std(axis=0)
        )

    elif selector == "worst_model_cvar95":
        model_scores = []

        for matrix in development_models.values():
            model_scores.append([
                cvar(matrix[:, j])
                for j in range(matrix.shape[1])
            ])

        scores = np.max(
            np.asarray(model_scores),
            axis=0,
        )

    else:
        raise ValueError(selector)

    return int(np.argmin(scores))


def hash_array(array):
    array = np.ascontiguousarray(array)

    return hashlib.sha256(
        array.tobytes()
    ).hexdigest()


def generate_scenarios(
    mu,
    cv,
    zone,
    seed,
    dev_n,
    test_n,
):
    dev = {}
    test = {}
    timings = {}
    hashes = {}

    for i, shift in enumerate(SHIFTS):
        start = time.perf_counter()

        dev[shift] = scenario_matrix(
            mu, cv, zone, shift, dev_n,
            np.random.default_rng(seed + 100 + i),
        )

        test[shift] = scenario_matrix(
            mu, cv, zone, shift, test_n,
            np.random.default_rng(seed + 10000 + i),
        )

        timings[shift] = time.perf_counter() - start
        hashes[shift] = hash_array(test[shift])

    return dev, test, timings, hashes


def bootstrap_graphs(values, reps, rng):
    values = np.asarray(values, dtype=float)

    if len(values) < 2:
        return np.nan, np.nan

    indices = rng.integers(
        0, len(values),
        size=(reps, len(values)),
    )

    samples = values[indices].mean(axis=1)

    return tuple(
        np.percentile(samples, [2.5, 97.5])
    )


def execute(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    if args.graphs < 2:
        raise ValueError("At least two graphs required")

    if args.nodes < 10:
        raise ValueError("At least ten nodes required")

    if args.dev_scenarios < 50:
        raise ValueError("Insufficient development scenarios")

    if args.test_scenarios < 100:
        raise ValueError("Insufficient test scenarios")

    if args.bootstrap < 100:
        raise ValueError("At least 100 bootstrap repetitions required")

    budgets = sorted(set(
        int(x.strip())
        for x in args.budgets.split(",")
    ))

    if not budgets or min(budgets) < 1:
        raise ValueError("Invalid candidate budgets")

    records = []
    diagnostics = []

    overall_start = time.perf_counter()

    for gi in range(args.graphs):
        seed = args.seed + gi * 10007

        graph, source, target = graph_instance(
            seed, args.nodes
        )

        edges, lookup, mu, cv, zone = edges_for(graph)

        dev_edges, test_edges, sample_times, hashes = (
            generate_scenarios(
                mu,
                cv,
                zone,
                seed,
                args.dev_scenarios,
                args.test_scenarios,
            )
        )

        for budget in budgets:
            for generator_index, generator in enumerate(
                GENERATORS
            ):
                start = time.perf_counter()

                candidates = generate_candidates(
                    graph,
                    source,
                    target,
                    mu,
                    cv,
                    edges,
                    generator,
                    budget,
                    np.random.default_rng(
                        seed + budget * 100 + generator_index
                    ),
                )

                generation_time = (
                    time.perf_counter() - start
                )

                if not candidates:
                    continue

                matrix = route_matrix(
                    candidates,
                    lookup,
                    len(edges),
                )

                known_means = matrix @ mu

                evaluation_start = time.perf_counter()

                development = {
                    shift: values @ matrix.T
                    for shift, values in dev_edges.items()
                }

                held_out = {
                    shift: values @ matrix.T
                    for shift, values in test_edges.items()
                }

                evaluation_time = (
                    time.perf_counter() - evaluation_start
                )

                diversity = path_diversity(candidates)

                diagnostics.append({
                    "graph": gi,
                    "generator": generator,
                    "budget": budget,
                    "actual_candidates": len(candidates),
                    "candidate_diversity": diversity,
                    "generation_seconds": generation_time,
                    "route_cost_matrix_seconds": evaluation_time,
                })

                for shift in SHIFTS:
                    for selector in SELECTORS:
                        selection_start = time.perf_counter()

                        chosen = select_route(
                            selector,
                            development,
                            known_means,
                            shift,
                        )

                        selection_time = (
                            time.perf_counter() - selection_start
                        )

                        observed = held_out[shift][:, chosen]

                        records.append({
                            "graph": gi,
                            "graph_seed": seed,
                            "shift": shift,
                            "generator": generator,
                            "budget": budget,
                            "selector": selector,
                            "chosen_candidate": chosen,
                            "chosen_route": "->".join(
                                map(str, candidates[chosen])
                            ),
                            "mean_seconds": float(
                                observed.mean()
                            ),
                            "cvar95_seconds": cvar(observed),
                            "p95_seconds": float(
                                np.quantile(observed, 0.95)
                            ),
                            "known_mean_seconds": float(
                                known_means[chosen]
                            ),
                            "actual_candidates": len(candidates),
                            "candidate_diversity": diversity,
                            "generation_seconds": generation_time,
                            "scenario_sampling_seconds": sample_times[shift],
                            "route_evaluation_seconds": evaluation_time,
                            "selection_seconds": selection_time,
                            "generation_plus_selection_seconds":
                                generation_time + selection_time,
                            "test_scenario_sha256": hashes[shift],
                            "dev_scenarios": args.dev_scenarios,
                            "test_scenarios": args.test_scenarios,
                        })

        print(
            f"Graph {gi + 1}/{args.graphs}: "
            f"{len(edges)} edges, "
            f"{len(records)} method records",
            flush=True,
        )

    df = pd.DataFrame(records)

    if df.empty:
        raise RuntimeError("No experiment results generated")

    df.to_csv(
        output / "all_method_results.csv",
        index=False,
    )

    pd.DataFrame(diagnostics).to_csv(
        output / "candidate_pool_diagnostics.csv",
        index=False,
    )

    keys = [
        "graph",
        "shift",
        "generator",
        "budget",
    ]

    reference = df[
        df.selector == "known_mean"
    ][
        keys + [
            "cvar95_seconds",
            "mean_seconds",
            "chosen_route",
        ]
    ].rename(columns={
        "cvar95_seconds": "reference_cvar",
        "mean_seconds": "reference_mean",
        "chosen_route": "reference_route",
    })

    paired = df.merge(
        reference,
        on=keys,
        validate="many_to_one",
    )

    paired["delta_cvar95_seconds"] = (
        paired.cvar95_seconds - paired.reference_cvar
    )

    paired["delta_mean_seconds"] = (
        paired.mean_seconds - paired.reference_mean
    )

    paired["route_changed"] = (
        paired.chosen_route != paired.reference_route
    )

    paired.to_csv(
        output / "paired_effects.csv",
        index=False,
    )

    summary = paired.groupby(
        ["shift", "generator", "budget", "selector"],
        as_index=False,
    ).agg(
        mean_delta_cvar95=(
            "delta_cvar95_seconds", "mean"
        ),
        mean_delta_mean=(
            "delta_mean_seconds", "mean"
        ),
        route_change_rate=(
            "route_changed", "mean"
        ),
        mean_runtime_s=(
            "generation_plus_selection_seconds", "mean"
        ),
        mean_candidate_count=(
            "actual_candidates", "mean"
        ),
        n_graphs=("graph", "nunique"),
    )

    summary.to_csv(
        output / "summary.csv",
        index=False,
    )

    bootstrap_rng = np.random.default_rng(
        args.seed + 3333
    )

    intervals = []

    for group_key, group in paired.groupby(
        ["shift", "generator", "budget", "selector"]
    ):
        values = group.groupby(
            "graph"
        ).delta_cvar95_seconds.mean().to_numpy()

        low, high = bootstrap_graphs(
            values,
            args.bootstrap,
            bootstrap_rng,
        )

        intervals.append({
            "shift": group_key[0],
            "generator": group_key[1],
            "budget": group_key[2],
            "selector": group_key[3],
            "mean_delta_cvar95": float(
                values.mean()
            ),
            "ci95_low": low,
            "ci95_high": high,
            "n_independent_graphs": len(values),
        })

    intervals = pd.DataFrame(intervals)

    intervals.to_csv(
        output / "graph_bootstrap_intervals.csv",
        index=False,
    )

    make_figures(
        summary,
        intervals,
        pd.DataFrame(diagnostics),
        output,
    )

    manifest = {
        "seed": args.seed,
        "graphs": args.graphs,
        "nodes": args.nodes,
        "budgets": budgets,
        "development_scenarios": args.dev_scenarios,
        "held_out_scenarios": args.test_scenarios,
        "alpha": ALPHA,
        "shifts": SHIFTS,
        "candidate_generators": GENERATORS,
        "selectors": SELECTORS,
        "records": len(df),
        "duration_seconds": round(
            time.perf_counter() - overall_start, 3
        ),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "networkx": nx.__version__,
        "matplotlib": matplotlib.__version__,
        "independence_unit": "synthetic graph seed",
        "known_limitations": [
            "Synthetic graphs and simulated travel times",
            "No measured traffic observations",
            "Not integrated with Phase 13 OSM road networks",
            "No independently reproduced named published algorithms",
            "Finite-model robustness is not formal DRO",
            "Exploratory confidence intervals are not multiplicity adjusted",
            "Scenario sampling and route evaluation are shared setup costs",
            "Graph-level bootstrap assumes independent graph instances",
        ],
    }

    (output / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    write_report(summary, manifest, output)

    print("\nFinished:", output.resolve())
    print(
        "Figures:",
        ", ".join(
            p.name for p in sorted(output.glob("*.png"))
        ),
    )


def make_figures(summary, intervals, pools, output):
    for shift in SHIFTS:
        fig, axes = plt.subplots(
            1, 3,
            figsize=(15, 4),
            sharey=True,
        )

        for ax, generator in zip(axes, GENERATORS):
            for selector in SELECTORS:
                if selector == "known_mean":
                    continue

                group = summary[
                    (summary["shift"] == shift)
                    & (summary.generator == generator)
                    & (summary.selector == selector)
                ].sort_values("budget")

                ax.plot(
                    group.budget,
                    group.mean_delta_cvar95,
                    marker="o",
                    label=selector,
                )

            ax.axhline(0, color="black", lw=0.8)
            ax.set_title(generator)
            ax.set_xlabel("Candidate budget")
            ax.grid(alpha=0.2)

        axes[0].set_ylabel(
            "CVaR95 minus known-mean route (s)"
        )

        fig.suptitle(
            f"Candidate-budget ablation: {shift}"
        )

        handles, labels = axes[-1].get_legend_handles_labels()

        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=4,
            fontsize=8,
        )

        fig.tight_layout(rect=(0, 0.10, 1, 0.93))

        fig.savefig(
            output / f"ablation_{shift}.png",
            dpi=180,
        )

        plt.close(fig)

    fig, axes = plt.subplots(
        1, 2,
        figsize=(12, 4),
    )

    for generator, group in pools.groupby("generator"):
        agg = group.groupby(
            "budget",
            as_index=False,
        ).agg(
            unique=("actual_candidates", "mean"),
            diversity=("candidate_diversity", "mean"),
        )

        axes[0].plot(
            agg.budget,
            agg.unique,
            marker="o",
            label=generator,
        )

        axes[1].plot(
            agg.budget,
            agg.diversity,
            marker="o",
            label=generator,
        )

    axes[0].set(
        xlabel="Candidate budget",
        ylabel="Actual unique candidates",
    )

    axes[1].set(
        xlabel="Candidate budget",
        ylabel="Mean edge-Jaccard distance",
    )

    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)

    fig.suptitle("Candidate diversity diagnostics")
    fig.tight_layout()

    fig.savefig(
        output / "candidate_diversity.png",
        dpi=180,
    )

    plt.close(fig)

    group = summary[
        (summary["shift"] == "heavy_lognormal")
        & (summary.selector != "known_mean")
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    for generator, sub in group.groupby("generator"):
        ax.scatter(
            sub.mean_runtime_s,
            sub.mean_delta_cvar95,
            label=generator,
        )

    ax.axhline(0, color="black", lw=0.8)

    ax.set(
        xlabel="Generation + selection runtime (seconds)",
        ylabel="CVaR95 difference (seconds)",
        title="Runtime versus tail-risk trade-off",
    )

    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()

    fig.savefig(
        output / "runtime_vs_risk.png",
        dpi=180,
    )

    plt.close(fig)

    max_budget = int(intervals.budget.max())

    group = intervals[
        (intervals.generator == "risk_informed")
        & (intervals.budget == max_budget)
        & (intervals.selector != "known_mean")
    ].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 7))

    y = np.arange(len(group))

    ax.errorbar(
        group.mean_delta_cvar95,
        y,
        xerr=[
            group.mean_delta_cvar95 - group.ci95_low,
            group.ci95_high - group.mean_delta_cvar95,
        ],
        fmt="o",
        capsize=3,
    )

    ax.axvline(0, color="black", lw=0.8)

    ax.set_yticks(
        y,
        [
            f"{r.shift} | {r.selector}"
            for r in group.itertuples()
        ],
        fontsize=8,
    )

    ax.set(
        xlabel="Paired CVaR95 difference (seconds)",
        title="Exploratory graph-bootstrap confidence intervals",
    )

    fig.tight_layout()

    fig.savefig(
        output / "independent_graph_intervals.png",
        dpi=180,
    )

    plt.close(fig)

    disagreement = (
        summary[summary.budget == max_budget]
        .groupby(["generator", "selector"])
        .route_change_rate.mean()
        .unstack()
    )

    fig, ax = plt.subplots(figsize=(10, 4))

    image = ax.imshow(
        disagreement.to_numpy(),
        vmin=0,
        vmax=1,
        aspect="auto",
        cmap="Blues",
    )

    ax.set_xticks(
        range(len(disagreement.columns)),
        disagreement.columns,
        rotation=25,
        ha="right",
        fontsize=8,
    )

    ax.set_yticks(
        range(len(disagreement.index)),
        disagreement.index,
    )

    for i in range(len(disagreement.index)):
        for j in range(len(disagreement.columns)):
            ax.text(
                j,
                i,
                f"{disagreement.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=9,
            )

    fig.colorbar(
        image,
        ax=ax,
        label="Route-change fraction",
    )

    ax.set_title("Route-selection disagreement")

    fig.tight_layout()

    fig.savefig(
        output / "route_disagreement.png",
        dpi=180,
    )

    plt.close(fig)


def write_report(summary, manifest, output):
    lines = [
        "# Phase 16 — Corrected Experimental Report",
        "",
        "## Status",
        "",
        "Exploratory synthetic-network study. Not a completed "
        "conference manuscript or a reproduction of a named paper.",
        "",
        "## Protocol",
        "",
        f"- Independent graph instances: {manifest['graphs']}",
        f"- Nodes per graph: {manifest['nodes']}",
        f"- Candidate budgets: {manifest['budgets']}",
        f"- Development scenarios: {manifest['development_scenarios']}",
        f"- Held-out scenarios: {manifest['held_out_scenarios']}",
        "- IID Gamma, Gaussian-copula correlated Gamma, and heavy Lognormal.",
        "- Known-mean Dijkstra candidate, sample-mean, empirical CVaR95, "
        "mean–variance, and finite-model worst-model CVaR95 selectors.",
        "- All methods share development-model information and the "
        "same held-out scenarios within each graph and shift.",
        "- Differences are paired against the known-mean route.",
        "",
        "## Selected results",
        "",
        "| Shift | Generator | Budget | Selector | Mean delta CVaR95 |",
        "|---|---|---:|---|---:|",
    ]

    selected = summary[
        (summary.budget == summary.budget.max())
        & (summary.generator == "risk_informed")
    ]

    for row in selected.itertuples():
        lines.append(
            f"| {row.shift} | {row.generator} | {row.budget} "
            f"| {row.selector} | {row.mean_delta_cvar95:.4f} |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "Negative delta CVaR95 favors the evaluated selector. "
        "Results are exploratory and not multiplicity adjusted.",
        "",
        "## Remaining limitations",
        "",
        "The study does not yet include measured traffic, independent "
        "published-method reproductions, or Phase 13 city-network integration.",
        "Known-mean selection is restricted to the generated candidate pool, "
        "which includes the true known-mean shortest path.",
        "Runtime comparisons exclude shared scenario-generation and "
        "route-cost evaluation costs.",
        "The worst-model selector uses three development distributions, "
        "so conclusions must specify this information assumption.",
    ])

    (output / "RESEARCH_REPORT.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def self_test():
    assert abs(
        cvar(np.arange(1, 11), 0.8) - 9.5
    ) < 1e-9

    assert abs(
        cvar([1, 2, 3, 4], 0.75) - 4
    ) < 1e-9

    graph, source, target = graph_instance(42, 20)

    edges, lookup, mu, cv, zone = edges_for(graph)

    candidates = generate_candidates(
        graph,
        source,
        target,
        mu,
        cv,
        edges,
        "risk_informed",
        5,
        np.random.default_rng(12),
    )

    assert candidates
    assert len(candidates) == len(set(candidates))

    for path in candidates:
        assert path[0] == source
        assert path[-1] == target
        assert all(
            graph.has_edge(u, v)
            for u, v in zip(path[:-1], path[1:])
        )

    matrix = route_matrix(
        candidates, lookup, len(edges)
    )

    dev, test, _, hashes = generate_scenarios(
        mu, cv, zone, 42, 300, 500
    )

    for shift in SHIFTS:
        assert dev[shift].shape == (
            300, len(edges)
        )

        assert test[shift].shape == (
            500, len(edges)
        )

        assert np.isfinite(test[shift]).all()
        assert (test[shift] > 0).all()
        assert len(hashes[shift]) == 64

    route_dev = {
        shift: values @ matrix.T
        for shift, values in dev.items()
    }

    known_means = matrix @ mu

    for shift in SHIFTS:
        for selector in SELECTORS:
            selected = select_route(
                selector,
                route_dev,
                known_means,
                shift,
            )

            assert 0 <= selected < len(candidates)

    print(
        "Self-test passed: CVaR, paths, Gamma copula, "
        "scenario separation, and route selectors"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output",
        default="data/results/phase16_corrected",
    )

    parser.add_argument(
        "--graphs", type=int, default=30
    )

    parser.add_argument(
        "--nodes", type=int, default=55
    )

    parser.add_argument(
        "--budgets", default="3,6,10"
    )

    parser.add_argument(
        "--dev-scenarios", type=int, default=1000
    )

    parser.add_argument(
        "--test-scenarios", type=int, default=5000
    )

    parser.add_argument(
        "--bootstrap", type=int, default=3000
    )

    parser.add_argument(
        "--seed", type=int, default=SEED
    )

    parser.add_argument(
        "--self-test", action="store_true"
    )

    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    self_test()
    execute(args)


if __name__ == "__main__":
    main()
