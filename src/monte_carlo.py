"""Phase 4: scenario-wise shortest paths and fixed-route evaluation."""
from __future__ import annotations

from collections import Counter
import heapq
import json
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.probability_engine import load_edges

FAMILIES = ('normal', 'lognormal', 'gamma', 'uniform')


def load_inputs(root: str | Path, dataset: str, family: str):
    root = Path(root)
    edges = load_edges(root / 'data' / 'synthetic' / f'{dataset}_edges.csv')
    meta_path = root / 'data' / 'synthetic' / f'{dataset}_metadata.json'
    if not meta_path.exists():
        raise FileNotFoundError(f'Missing Phase 2 metadata: {meta_path}')
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    sample_path = root / 'data' / 'results' / 'phase3' / f'{dataset}_{family}_samples.npz'
    if not sample_path.exists():
        raise FileNotFoundError(f'Missing Phase 3 samples: {sample_path}; run python run_phase3.py')
    with np.load(sample_path, allow_pickle=False) as data:
        if 'edge_ids' not in data or not ({'costs', 'samples'} & set(data.files)):
            raise ValueError('NPZ must contain edge_ids and costs (or samples)')
        key = 'costs' if 'costs' in data else 'samples'
        samples = np.asarray(data[key], dtype=np.float64)
        saved_ids = data['edge_ids']
    if samples.ndim != 2 or samples.shape[1] != len(edges) or samples.shape[0] < 1:
        raise ValueError('Sample matrix shape does not match edge CSV')
    if not np.array_equal(saved_ids.astype(str), edges['edge_id'].to_numpy().astype(str)):
        raise ValueError('Edge IDs/order in NPZ do not match edge CSV; regenerate Phase 3')
    if not np.isfinite(samples).all() or (samples < 0).any():
        raise ValueError('Samples must be finite and nonnegative')
    source, target = int(meta['source']), int(meta['target'])
    return edges, samples, source, target


def build_network(edges: pd.DataFrame):
    """Map node IDs to adjacency entries (neighbor, edge column)."""
    adj = {}
    edge_lookup = {}
    for idx, row in enumerate(edges.itertuples(index=False)):
        u, v = int(row.source), int(row.target)
        adj.setdefault(u, []).append((v, idx))
        adj.setdefault(v, [])
        if (u, v) in edge_lookup:
            raise ValueError(f'Duplicate directed edge: {u}->{v}')
        edge_lookup[(u, v)] = idx
    for entries in adj.values():
        entries.sort(key=lambda pair: (pair[0], pair[1]))
    return adj, edge_lookup


def shortest_path(adj: dict, weights: np.ndarray, source: int, target: int):
    """Dijkstra; returns (cost, tuple of nodes)."""
    if source not in adj or target not in adj:
        raise ValueError('Source/target not present in graph')
    if np.any(~np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError('Dijkstra requires finite nonnegative weights')
    distance = {source: 0.0}
    predecessor = {}
    queue = [(0.0, source)]
    settled = set()
    while queue:
        cost, u = heapq.heappop(queue)
        if u in settled:
            continue
        settled.add(u)
        if u == target:
            route = [u]
            while route[-1] != source:
                route.append(predecessor[route[-1]])
            return cost, tuple(reversed(route))
        for v, idx in adj[u]:
            if v in settled:
                continue
            proposal = cost + float(weights[idx])
            if proposal < distance.get(v, float('inf')):
                distance[v] = proposal
                predecessor[v] = u
                heapq.heappush(queue, (proposal, v))
    raise ValueError(f'No route from {source} to {target}')


def route_columns(path: tuple[int, ...], edge_lookup: dict) -> np.ndarray:
    if len(path) < 2:
        return np.array([], dtype=int)
    return np.array([edge_lookup[(u, v)] for u, v in zip(path[:-1], path[1:])], dtype=int)


def summarize(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError('Expected nonempty one-dimensional costs')
    var95 = float(np.quantile(values, 0.95))
    # CVaR computed as the average cost in the worst 5% of observations.
    # This definition handles a finite sample by taking the largest ceil(5% * n) values.
    tail_count = max(1, int(np.ceil(0.05 * len(values))))
    cvar95 = float(np.mean(np.sort(values)[-tail_count:]))
    return {'mean': float(np.mean(values)), 'std': float(np.std(values, ddof=0)),
            'variance': float(np.var(values, ddof=0)), 'median': float(np.median(values)),
            'var95': var95, 'cvar95': cvar95, 'min': float(np.min(values)),
            'max': float(np.max(values))}


def simulate(edges: pd.DataFrame, samples: np.ndarray, source: int, target: int):
    adj, lookup = build_network(edges)
    baseline_weights = edges['mean_cost'].to_numpy(dtype=float)
    baseline_cost, baseline_path = shortest_path(adj, baseline_weights, source, target)
    cols = route_columns(baseline_path, lookup)
    fixed_costs = samples[:, cols].sum(axis=1)
    oracle_costs = np.empty(len(samples), dtype=float)
    oracle_paths = []
    start = perf_counter()
    for i, weights in enumerate(samples):
        oracle_costs[i], path = shortest_path(adj, weights, source, target)
        oracle_paths.append(path)
    runtime = perf_counter() - start
    regret = fixed_costs - oracle_costs
    if np.min(regret) < -1e-7:
        raise AssertionError('Negative regret: check graph/sample edge alignment')
    regret = np.maximum(regret, 0.0)
    equal_optimal = np.isclose(fixed_costs, oracle_costs, rtol=1e-9, atol=1e-8)
    scenarios = pd.DataFrame({
        'scenario': np.arange(len(samples)), 'fixed_cost': fixed_costs,
        'oracle_cost': oracle_costs, 'regret': regret,
        'fixed_is_optimal': equal_optimal,
        'oracle_path': ['->'.join(map(str, p)) for p in oracle_paths],
    })
    counts = Counter(oracle_paths)
    frequencies = pd.DataFrame([
        {'path': '->'.join(map(str, path)), 'count': count,
         'frequency': count / len(samples)} for path, count in counts.items()
    ]).sort_values(['count', 'path'], ascending=[False, True]).reset_index(drop=True)
    summary = {
        'source': source, 'target': target, 'scenarios': len(samples),
        'baseline_path': list(baseline_path), 'baseline_expected_cost': baseline_cost,
        'fixed_route': summarize(fixed_costs), 'hindsight_oracle': summarize(oracle_costs),
        'regret': summarize(regret), 'fixed_optimal_fraction': float(np.mean(equal_optimal)),
        'oracle_distinct_paths': len(counts), 'simulation_seconds': runtime,
        'reliability_definition': 'P(fixed route cost <= threshold)',
        'reliability_threshold': float(1.2 * baseline_cost),
        'fixed_reliability': float(np.mean(fixed_costs <= 1.2 * baseline_cost)),
        'oracle_is_hindsight_only': True,
    }
    return scenarios, frequencies, summary


def save_plots(scenarios: pd.DataFrame, out_dir: Path, stem: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(scenarios['fixed_cost'], bins=45, alpha=0.65, label='Fixed expected-cost path')
    ax.hist(scenarios['oracle_cost'], bins=45, alpha=0.55, label='Hindsight optimum')
    ax.set(xlabel='Realized source-to-target cost', ylabel='Scenario count',
           title=f'{stem}: route cost distributions')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f'{stem}_cost_histogram.png', dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(scenarios['regret'], bins=45)
    ax.set(xlabel='Fixed-route regret', ylabel='Scenario count', title=f'{stem}: regret distribution')
    fig.tight_layout()
    fig.savefig(out_dir / f'{stem}_regret_histogram.png', dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    n = np.arange(1, len(scenarios) + 1)
    for col, label in [('fixed_cost', 'Fixed route'), ('oracle_cost', 'Hindsight optimum')]:
        avg = np.cumsum(scenarios[col].to_numpy()) / n
        ax.plot(n, avg, label=label)
    ax.set(xlabel='Number of scenarios', ylabel='Running mean cost',
           title=f'{stem}: Monte Carlo convergence')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f'{stem}_convergence.png', dpi=160)
    plt.close(fig)


def run_phase4(dataset: str = 'grid_25', families=FAMILIES,
               project_root: str | Path | None = None):
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[1]
    results = root / 'data' / 'results' / 'phase4'
    plots = root / 'outputs' / 'plots' / 'phase4'
    results.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    aggregate = []
    for family in families:
        if family not in FAMILIES:
            raise ValueError(f'Unknown distribution family: {family}')
        edges, samples, source, target = load_inputs(root, dataset, family)
        scenarios, frequencies, summary = simulate(edges, samples, source, target)
        summary.update({'dataset': dataset, 'distribution': family})
        stem = f'{dataset}_{family}'
        scenarios.to_csv(results / f'{stem}_scenarios.csv', index=False)
        frequencies.to_csv(results / f'{stem}_path_frequencies.csv', index=False)
        (results / f'{stem}_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        save_plots(scenarios, plots, stem)
        aggregate.append({'distribution': family, 'scenarios': len(scenarios),
                          'fixed_mean': summary['fixed_route']['mean'],
                          'oracle_mean': summary['hindsight_oracle']['mean'],
                          'mean_regret': summary['regret']['mean'],
                          'fixed_optimal_fraction': summary['fixed_optimal_fraction'],
                          'fixed_var95': summary['fixed_route']['var95'],
                          'fixed_cvar95': summary['fixed_route']['cvar95'],
                          'fixed_reliability': summary['fixed_reliability'],
                          'simulation_seconds': summary['simulation_seconds']})
        print(f'{family:<10} scenarios={len(scenarios):5} '
              f'fixed_mean={summary["fixed_route"]["mean"]:.3f} '
              f'oracle_mean={summary["hindsight_oracle"]["mean"]:.3f} '
              f'mean_regret={summary["regret"]["mean"]:.3f} '
              f'optimal_fraction={summary["fixed_optimal_fraction"]:.1%}')
    pd.DataFrame(aggregate).to_csv(results / f'{dataset}_comparison.csv', index=False)
    print('PHASE 4 COMPLETED SUCCESSFULLY')
    return pd.DataFrame(aggregate)


if __name__ == '__main__':
    run_phase4()
