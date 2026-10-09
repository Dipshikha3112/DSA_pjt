
from __future__ import annotations
import json
from pathlib import Path
from time import perf_counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from src.monte_carlo import load_inputs, build_network, shortest_path, route_columns, summarize

FAMILIES = ('normal', 'lognormal', 'gamma', 'uniform')
ALGORITHMS = ('expected_dijkstra', 'monte_carlo_saa', 'mean_variance', 'robust_minimax')


def split_scenarios(samples, train_fraction=0.7, seed=2027):
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2 or len(samples) < 2 or not np.isfinite(samples).all() or np.any(samples < 0):
        raise ValueError('Expected at least two scenarios with finite nonnegative costs')
    if not 0 < train_fraction < 1:
        raise ValueError('train_fraction must be strictly between 0 and 1')
    order = np.random.default_rng(seed).permutation(len(samples))
    n_train = min(len(samples)-1, max(1, int(len(samples)*train_fraction)))
    return order[:n_train], order[n_train:]


def candidates_from_weights(edges, source, target, weight_sets, k=80):
    """Union of up to k simple shortest paths per supplied edge-weight vector."""
    if k < 1:
        raise ValueError('k must be positive')
    adj, lookup = build_network(edges)
    g = nx.DiGraph()
    g.add_nodes_from(adj)
    for u, v in lookup:
        g.add_edge(u, v)
    paths = set()
    for weights in weight_sets:
        weights = np.asarray(weights, dtype=float)
        if len(weights) != len(edges) or not np.isfinite(weights).all() or (weights < 0).any():
            raise ValueError('Invalid edge weight vector')
        for (u, v), idx in lookup.items():
            g[u][v]['cost'] = float(weights[idx])
        try:
            generator = nx.shortest_simple_paths(g, source, target, weight='cost')
            for _, path in zip(range(k), generator):
                paths.add(tuple(path))
        except (nx.NetworkXNoPath, nx.NodeNotFound) as exc:
            raise ValueError('No source-target path') from exc
    if not paths:
        raise ValueError('No candidate paths')
    return sorted(paths)


def evaluate_candidates(paths, lookup, train):
    incidence = np.zeros((len(lookup), len(paths)), dtype=np.float64)
    for j, path in enumerate(paths):
        incidence[route_columns(path, lookup), j] = 1.0
    return train @ incidence, incidence


def choose_routes(edges, train, source, target, risk_lambda=0.1, k=80, robust_scenarios=250, seed=2027):
    """Choose four paths using training data only; robust/mean-variance are candidate-restricted."""
    if risk_lambda < 0 or not np.isfinite(risk_lambda):
        raise ValueError('risk_lambda must be finite and nonnegative')
    if robust_scenarios < 1:
        raise ValueError('robust_scenarios must be positive')
    adj, lookup = build_network(edges)
    theory = edges['mean_cost'].to_numpy(dtype=float)
    empirical = train.mean(axis=0)
    t0 = perf_counter()
    _, expected_path = shortest_path(adj, theory, source, target)
    expected_seconds = perf_counter()-t0
    t0 = perf_counter()
    _, saa_path = shortest_path(adj, empirical, source, target)
    saa_seconds = perf_counter()-t0
    # Candidate construction uses only training scenarios; never evaluation samples.
    # Include shortest paths for mean, SAA, risk-penalized edges, and sampled extremes.
    variance = train.var(axis=0)
    scale = max(1.0, float(np.mean(empirical)))
    weights = [theory, empirical, empirical + risk_lambda * variance / scale]
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(train), size=min(robust_scenarios, len(train)), replace=False)
    subset = train[chosen]
    weights.extend([np.max(subset, axis=0), np.quantile(subset, .9, axis=0)])
    t0 = perf_counter()
    paths = candidates_from_weights(edges, source, target, weights, k=k)
    paths = sorted(set(paths) | {expected_path, saa_path})
    costs, _ = evaluate_candidates(paths, lookup, train)
    # Exact variance of each candidate's summed cost, including cross-edge covariance.
    means = costs.mean(axis=0)
    variances = costs.var(axis=0)
    # Risk lambda has units 1/cost, so mean + lambda * variance is cost-valued.
    risk_index = int(np.argmin(means + risk_lambda * variances))
    risk_seconds = perf_counter()-t0
    t0 = perf_counter()
    # Maximum across ALL training scenarios, not just the candidate-generation subset.
    robust_index = int(np.argmin(costs.max(axis=0)))
    robust_seconds = perf_counter()-t0 + risk_seconds
    return {
        'expected_dijkstra': (expected_path, expected_seconds),
        'monte_carlo_saa': (saa_path, saa_seconds),
        'mean_variance': (paths[risk_index], risk_seconds),
        'robust_minimax': (paths[robust_index], robust_seconds),
    }, len(paths)


def compare_family(edges, samples, source, target, train_fraction=.7, seed=2027,
                   risk_lambda=.1, k=80, robust_scenarios=250):
    train_ids, test_ids = split_scenarios(samples, train_fraction, seed)
    train, test = samples[train_ids], samples[test_ids]
    selected, n_candidates = choose_routes(edges, train, source, target, risk_lambda, k, robust_scenarios, seed)
    adj, lookup = build_network(edges)
    oracle = np.array([shortest_path(adj, w, source, target)[0] for w in test])
    # A single shared threshold defined without seeing held-out outcomes.
    expected_cost, _ = shortest_path(adj, edges['mean_cost'].to_numpy(float), source, target)
    threshold = 1.2 * expected_cost
    records, scenarios = [], []
    for name, (path, selection_seconds) in selected.items():
        columns = route_columns(path, lookup)
        realized = test[:, columns].sum(axis=1)
        regret = realized-oracle
        if np.any(regret < -1e-7):
            raise AssertionError('Negative regret: check edge alignment')
        regret = np.maximum(regret, 0.0)
        metrics = summarize(realized)
        records.append({'algorithm': name, 'path': '->'.join(map(str,path)), 'hops': len(path)-1,
                        'mean_cost': metrics['mean'], 'variance': metrics['variance'],
                        'std': metrics['std'], 'var95': metrics['var95'], 'cvar95': metrics['cvar95'],
                        'mean_regret': float(regret.mean()),
                        'optimal_fraction': float(np.mean(np.isclose(realized, oracle, atol=1e-8, rtol=1e-9))),
                        'reliability': float(np.mean(realized <= threshold)),
                        'reliability_threshold': float(threshold),
                        'selection_seconds': selection_seconds,
                        'train_size': len(train), 'test_size': len(test), 'candidate_count': n_candidates})
        scenarios.extend({'scenario_index': int(test_ids[i]), 'algorithm': name,
                          'cost': float(realized[i]), 'oracle_cost': float(oracle[i]),
                          'regret': float(regret[i])} for i in range(len(test)))
    return pd.DataFrame(records), pd.DataFrame(scenarios), {'train_indices': train_ids, 'test_indices': test_ids}


def plot_comparison(frame, directory, dataset):
    directory.mkdir(parents=True, exist_ok=True)
    for metric, ylabel in [('mean_cost','Mean cost'), ('variance','Cost variance'),
                           ('cvar95','CVaR 95%'), ('mean_regret','Mean regret')]:
        pivot = frame.pivot(index='distribution', columns='algorithm', values=metric)
        ax = pivot.plot.bar(figsize=(11,5), rot=0)
        ax.set(xlabel='Distribution', ylabel=ylabel, title=f'{dataset}: {ylabel} (held-out)')
        ax.legend(fontsize=8)
        ax.figure.tight_layout()
        ax.figure.savefig(directory / f'{dataset}_{metric}.png', dpi=150)
        plt.close(ax.figure)


def run_phase5(dataset='grid_25', families=FAMILIES, project_root=None,
               train_fraction=.7, seed=2027, risk_lambda=.1, k=80, robust_scenarios=250):
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[1]
    out = root / 'data' / 'results' / 'phase5'
    plots = root / 'outputs' / 'plots' / 'phase5'
    out.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for family in families:
        if family not in FAMILIES:
            raise ValueError(f'Unknown distribution: {family}')
        edges, samples, source, target = load_inputs(root, dataset, family)
        comparison, scenarios, split = compare_family(edges, samples, source, target,
                            train_fraction, seed, risk_lambda, k, robust_scenarios)
        comparison.insert(0, 'distribution', family)
        scenarios.insert(0, 'distribution', family)
        stem = f'{dataset}_{family}'
        comparison.to_csv(out / f'{stem}_algorithms.csv', index=False)
        scenarios.to_csv(out / f'{stem}_heldout_scenarios.csv', index=False)
        (out / f'{stem}_split.json').write_text(json.dumps({
            'seed': seed, 'train_fraction': train_fraction,
            'train_indices': split['train_indices'].tolist(),
            'test_indices': split['test_indices'].tolist(),
            'risk_lambda': risk_lambda, 'candidate_k': k,
            'robust_scenarios_for_candidate_generation': robust_scenarios,
            'candidate_restricted_optimization': True,
            'oracle_is_hindsight_only': True,
        }, indent=2), encoding='utf-8')
        all_rows.append(comparison)
        print(f'{family:<10} train={len(split["train_indices"]):5} test={len(split["test_indices"]):5} '
              f'candidates={int(comparison.candidate_count.iloc[0]):4} '
              f'best_heldout_mean={comparison.mean_cost.min():.3f}')
    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(out / f'{dataset}_algorithm_comparison.csv', index=False)
    plot_comparison(combined, plots, dataset)
    print('PHASE 5 COMPLETED SUCCESSFULLY')
    return combined
