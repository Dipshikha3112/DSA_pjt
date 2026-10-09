"""Standalone, reproducible Phase 7 benchmark. Requires numpy, pandas, networkx, matplotlib."""
import json
import math
import time
import tracemalloc
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx

DISTS = ('normal', 'lognormal', 'gamma', 'uniform')
LAMBDAS = (0.0, 0.1, 1.0, 5.0)


def graph_for(topology, requested_n, seed):
    if topology == 'grid':
        side = math.ceil(math.sqrt(requested_n))
        base = nx.grid_2d_graph(side, side)
        base = nx.convert_node_labels_to_integers(base)
    elif topology == 'erdos_renyi':
        base = nx.erdos_renyi_graph(requested_n, min(0.18, 5 / max(requested_n - 1, 1)), seed=seed)
    elif topology == 'scale_free':
        base = nx.barabasi_albert_graph(requested_n, min(3, requested_n - 1), seed=seed)
    else:
        raise ValueError('unknown topology')
    # A directed bidirectional network, with deterministic connectivity repair.
    if not nx.is_connected(base):
        comps = [sorted(c) for c in nx.connected_components(base)]
        for a, b in zip(comps, comps[1:]):
            base.add_edge(a[0], b[0])
    rng = np.random.default_rng(seed + 314159)
    graph = nx.DiGraph()
    graph.add_nodes_from(base.nodes)
    for u, v in sorted(base.edges):
        for a, b in ((u, v), (v, u)):
            # Asymmetric directional costs; stable edge order.
            graph.add_edge(a, b, mean_cost=float(rng.uniform(4, 15)))
    return graph


def edge_table(graph, cv):
    edges = sorted(graph.edges())
    means = np.array([graph[u][v]['mean_cost'] for u, v in edges], dtype=float)
    return edges, means, means * cv


def sample_scenarios(means, stds, distribution, count, rng):
    if distribution not in DISTS:
        raise ValueError('unknown distribution')
    if count < 2 or np.any(means <= 0) or np.any(stds < 0):
        raise ValueError('invalid sample parameters')
    shape = (count, len(means))
    if distribution == 'normal':
        # Nonnegative normal by clipping; negligible moment distortion at CV<=0.6.
        return np.maximum(rng.normal(means, stds, size=shape), 0.0)
    if distribution == 'lognormal':
        s2 = np.log1p((stds / means) ** 2)
        return rng.lognormal(np.log(means) - s2 / 2, np.sqrt(s2), size=shape)
    if distribution == 'gamma':
        safe = np.maximum(stds, 1e-12)
        return rng.gamma((means / safe) ** 2, safe**2 / means, size=shape)
    return rng.uniform(means - np.sqrt(3) * stds, means + np.sqrt(3) * stds, size=shape).clip(min=0)


def path_indices(path, edge_index):
    return np.array([edge_index[(a, b)] for a, b in zip(path[:-1], path[1:])], dtype=int)


def shortest(graph, weights, edges, source, target):
    for (u, v), w in zip(edges, weights):
        graph[u][v]['weight_phase7'] = float(max(w, 1e-10))
    return tuple(nx.shortest_path(graph, source, target, weight='weight_phase7'))


def candidates(graph, edges, means, train, source, target, budget, seed):
    if budget < 1:
        raise ValueError('candidate budget must be positive')
    rng = np.random.default_rng(seed)
    pool = {}
    def add(weights):
        path = shortest(graph, weights, edges, source, target)
        pool[path] = None
    add(means)
    add(train.mean(axis=0))
    add(train.std(axis=0))
    add(train.max(axis=0))
    add(means + train.std(axis=0))
    add(means + 3 * train.std(axis=0))
    # Random positive edge reweighting supplies diverse routes.
    for _ in range(budget * 4):
        if len(pool) >= budget:
            break
        multiplier = rng.lognormal(0, rng.uniform(0.25, 1.7), size=len(edges))
        add(means * multiplier)
    return list(pool)[:budget]


def pareto_indices(mean, variance):
    return [i for i in range(len(mean)) if not any(
        (mean[j] <= mean[i] and variance[j] <= variance[i]) and
        (mean[j] < mean[i] or variance[j] < variance[i])
        for j in range(len(mean)) if j != i)]


def metrics(costs):
    threshold = np.quantile(costs, .95)
    return dict(mean_cost=float(costs.mean()), variance=float(costs.var()),
                var95=float(threshold), cvar95=float(costs[costs >= threshold].mean()),
                worst_cost=float(costs.max()))


def evaluate_case(topology, requested_n, seed, distribution, budget, scenarios=600, cv=.4):
    graph = graph_for(topology, requested_n, seed)
    edges, means, stds = edge_table(graph, cv)
    source, target = 0, max(graph.nodes)
    rng = np.random.default_rng(seed * 100003 + requested_n * 101 + DISTS.index(distribution))
    samples = sample_scenarios(means, stds, distribution, scenarios, rng)
    perm = rng.permutation(scenarios)
    cut = int(.7 * scenarios)
    train, test = samples[perm[:cut]], samples[perm[cut:]]
    edge_index = {edge: i for i, edge in enumerate(edges)}
    tracemalloc.start()
    started = time.perf_counter()
    pool = candidates(graph, edges, means, train, source, target, budget, seed + 1009)
    candidate_seconds = time.perf_counter() - started
    train_costs = np.stack([train[:, path_indices(p, edge_index)].sum(axis=1) for p in pool], axis=1)
    test_costs = np.stack([test[:, path_indices(p, edge_index)].sum(axis=1) for p in pool], axis=1)
    train_mean = train_costs.mean(axis=0)
    train_var = train_costs.var(axis=0)
    train_max = train_costs.max(axis=0)
    pareto = pareto_indices(train_mean, train_var)
    choices = [('expected_dijkstra', None, pool.index(shortest(graph, means, edges, source, target))),
               ('monte_carlo_saa', None, pool.index(shortest(graph, train.mean(axis=0), edges, source, target))),
               ('robust_minimax', None, int(np.argmin(train_max)))]
    for lam in LAMBDAS:
        choices.append(('mean_variance', lam, int(np.argmin(train_mean + lam * train_var))))
    rows = []
    for algorithm, lam, idx in choices:
        held = test_costs[:, idx]
        m = metrics(held)
        m.update(topology=topology, requested_n=requested_n, actual_n=graph.number_of_nodes(),
                 edges=graph.number_of_edges(), seed=seed, distribution=distribution,
                 uncertainty_cv=cv, candidate_budget=budget, candidate_count=len(pool),
                 train_size=len(train), test_size=len(test), algorithm=algorithm,
                 risk_lambda=lam, path='->'.join(map(str, pool[idx])),
                 train_mean=float(train_mean[idx]), train_variance=float(train_var[idx]),
                 train_worst=float(train_max[idx]), test_worst=float(held.max()),
                 generalization_gap_worst=float(held.max() - train_max[idx]),
                 candidate_generation_seconds=candidate_seconds,
                 pareto_count=len(pareto), selected_on_pareto=idx in pareto)
        rows.append(m)
    # Track peak traced Python allocations; excludes some native NumPy allocations.
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    elapsed = time.perf_counter() - started
    for r in rows:
        r['total_case_seconds'] = elapsed
        r['traced_peak_mb'] = peak / 1048576
    frontier = [dict(topology=topology, requested_n=requested_n, seed=seed,
                     distribution=distribution, candidate_budget=budget,
                     path='->'.join(map(str, pool[i])), train_mean=float(train_mean[i]),
                     train_variance=float(train_var[i])) for i in pareto]
    return rows, frontier


def plot_results(df, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out.mkdir(parents=True, exist_ok=True)
    base = df[df.algorithm.isin(['expected_dijkstra', 'robust_minimax'])]
    for metric in ['cvar95', 'total_case_seconds', 'traced_peak_mb']:
        grouped = base.groupby(['topology', 'actual_n', 'algorithm'])[metric].mean().reset_index()
        fig, ax = plt.subplots(figsize=(8, 5))
        for (topo, algo), g in grouped.groupby(['topology', 'algorithm']):
            g = g.sort_values('actual_n')
            ax.plot(g.actual_n, g[metric], marker='o', label=f'{topo}: {algo}')
        ax.set(xlabel='Number of nodes', ylabel=metric, title=f'Phase 7: {metric}')
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / f'{metric}.png', dpi=150)
        plt.close(fig)


def run(output_root='data/results/phase7', plot_root='outputs/plots/phase7',
        sizes=(25, 49, 100), seeds=(1, 2, 3), distributions=DISTS,
        budgets=(8, 24, 48), scenarios=600, cv=.4):
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    rows, fronts = [], []
    total = 3 * len(sizes) * len(seeds) * len(distributions) * len(budgets)
    completed = 0
    for topology in ('grid', 'erdos_renyi', 'scale_free'):
        for size in sizes:
            for seed in seeds:
                for dist in distributions:
                    for budget in budgets:
                        r, f = evaluate_case(topology, size, seed, dist, budget, scenarios, cv)
                        rows.extend(r)
                        fronts.extend(f)
                        completed += 1
                        if completed % 12 == 0 or completed == total:
                            print(f'Completed {completed}/{total} cases', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(output / 'benchmark_results.csv', index=False)
    pd.DataFrame(fronts).to_csv(output / 'pareto_frontiers.csv', index=False)
    summary = df.groupby(['topology', 'actual_n', 'algorithm', 'candidate_budget'], dropna=False)[
        ['mean_cost', 'cvar95', 'train_worst', 'test_worst', 'total_case_seconds', 'traced_peak_mb']
    ].agg(['mean', 'std'])
    summary.to_csv(output / 'benchmark_summary.csv')
    config = dict(sizes=list(sizes), seeds=list(seeds), distributions=list(distributions),
                  budgets=list(budgets), scenarios=scenarios, cv=cv,
                  train_fraction=.7, note='Independent seeded synthetic graphs and scenario draws; not Phase 3 replay')
    (output / 'experiment_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    plot_results(df, Path(plot_root))
    print(f'PHASE 7 COMPLETED SUCCESSFULLY: {len(df)} algorithm records')
    return df
