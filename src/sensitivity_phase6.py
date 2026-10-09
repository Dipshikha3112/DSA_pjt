"""Phase 6: leakage-free risk/uncertainty sensitivity and controlled benchmark.
Only imports existing Phase 4/5 modules; never modifies their outputs.
"""
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
from src.algorithms_phase5 import (FAMILIES, split_scenarios, candidates_from_weights,
                                   evaluate_candidates)
from src.monte_carlo import load_inputs, build_network, shortest_path, route_columns, summarize

LAMBDAS = (0., .1, .5, 1., 2., 5.)
UNCERTAINTIES = (.1, .2, .4, .6)


def scale_uncertainty(samples, means, original_cv=.2, target_cv=.2):
    """Rescale residuals around theoretical means; clip to nonnegative values.
    Not a fresh draw from the original parametric family at target CV.
    """
    x = np.asarray(samples, float)
    mu = np.asarray(means, float)
    if x.ndim != 2 or mu.shape != (x.shape[1],):
        raise ValueError('Sample/mean shape mismatch')
    if not np.isfinite(x).all() or not np.isfinite(mu).all() or (x < 0).any() or (mu <= 0).any():
        raise ValueError('Costs must be finite, nonnegative; means positive')
    if original_cv <= 0 or target_cv < 0 or not np.isfinite([original_cv, target_cv]).all():
        raise ValueError('Invalid uncertainty coefficient')
    return np.maximum(0., mu + (x-mu)*(target_cv/original_cv))


def candidate_pool(edges, train, source, target, k=40, robust_scenarios=150, seed=2027):
    """Candidates derived ONLY from training costs and theoretical means."""
    if k < 1 or robust_scenarios < 1:
        raise ValueError('k and robust_scenarios must be positive')
    mu = edges.mean_cost.to_numpy(float)
    empirical = train.mean(axis=0)
    variances = train.var(axis=0)
    rng = np.random.default_rng(seed)
    sample_ids = rng.choice(len(train), size=min(robust_scenarios, len(train)), replace=False)
    subset = train[sample_ids]
    weights = [mu, empirical, empirical + variances * 5.,
               np.max(subset, axis=0), np.quantile(subset, .9, axis=0)]
    paths = candidates_from_weights(edges, source, target, weights, k=k)
    adj, _ = build_network(edges)
    paths = sorted(set(paths) | {shortest_path(adj, mu, source, target)[1],
                                 shortest_path(adj, empirical, source, target)[1]})
    return paths


def heldout_metrics(path, lookup, test, oracle, threshold):
    realized = test[:, route_columns(path, lookup)].sum(axis=1)
    s = summarize(realized)
    regret = np.maximum(realized-oracle, 0.)
    return dict(mean_cost=float(s['mean']), variance=float(s['variance']),
                std=float(s['std']), var95=float(s['var95']), cvar95=float(s['cvar95']),
                worst_cost=float(realized.max()), mean_regret=float(regret.mean()),
                optimal_fraction=float(np.mean(np.isclose(realized, oracle, atol=1e-8, rtol=1e-9))),
                reliability=float(np.mean(realized <= threshold)))


def evaluate_setting(edges, samples, source, target, distribution='normal', uncertainty=.2,
                     original_cv=.2, lambdas=LAMBDAS, train_fraction=.7,
                     seed=2027, k=40, robust_scenarios=150):
    """Single split reused across all lambda settings. Held-out samples never select paths."""
    if not lambdas or any(not np.isfinite(l) or l < 0 for l in lambdas):
        raise ValueError('All lambdas must be finite and nonnegative')
    mu = edges.mean_cost.to_numpy(float)
    scaled = scale_uncertainty(samples, mu, original_cv, uncertainty)
    tr_ids, te_ids = split_scenarios(scaled, train_fraction, seed)
    train, test = scaled[tr_ids], scaled[te_ids]
    adj, lookup = build_network(edges)
    start = perf_counter()
    paths = candidate_pool(edges, train, source, target, k, robust_scenarios, seed)
    train_costs, _ = evaluate_candidates(paths, lookup, train)
    train_means = train_costs.mean(axis=0)
    train_vars = train_costs.var(axis=0)
    train_worst = train_costs.max(axis=0)
    pool_seconds = perf_counter()-start
    expected = shortest_path(adj, mu, source, target)[1]
    saa = shortest_path(adj, train.mean(axis=0), source, target)[1]
    robust = paths[int(np.argmin(train_worst))]
    oracle = np.array([shortest_path(adj, w, source, target)[0] for w in test])
    threshold = 1.2 * shortest_path(adj, mu, source, target)[0]
    choices = [('expected_dijkstra', None, expected), ('monte_carlo_saa', None, saa),
               ('robust_minimax', None, robust)]
    for lam in lambdas:
        idx = int(np.argmin(train_means + lam*train_vars))
        choices.append(('mean_variance', float(lam), paths[idx]))
    records = []
    for algo, lam, path in choices:
        record = dict(distribution=distribution, uncertainty_cv=float(uncertainty),
                      algorithm=algo, risk_lambda=lam,
                      path='->'.join(map(str,path)), hops=len(path)-1,
                      candidate_count=len(paths), train_size=len(train), test_size=len(test),
                      reliability_threshold=float(threshold),
                      candidate_generation_seconds=pool_seconds)
        record.update(heldout_metrics(path, lookup, test, oracle, threshold))
        records.append(record)
    return pd.DataFrame(records), dict(train_indices=tr_ids.tolist(), test_indices=te_ids.tolist(),
                                       candidate_count=len(paths), threshold=float(threshold))


def controlled_benchmark(lambdas=LAMBDAS, seed=91, n_train=6000, n_test=4000):
    """Two parallel routes: risky low-mean vs safe higher-mean.
    Route A ~ Normal(10, 5) clipped at zero; Route B ~ Normal(12, .5).
    All samples independent; train/test are separately generated.
    """
    rng = np.random.default_rng(seed)
    if n_train < 2 or n_test < 2:
        raise ValueError('At least two training and evaluation samples required')
    train = np.maximum(0, rng.normal([10., 12.], [5., .5], size=(n_train, 2)))
    test = np.maximum(0, rng.normal([10., 12.], [5., .5], size=(n_test, 2)))
    rows = []
    for lam in lambdas:
        if lam < 0 or not np.isfinite(lam):
            raise ValueError('Invalid lambda')
        scores = train.mean(axis=0) + lam * train.var(axis=0)
        choice = int(np.argmin(scores))
        cost = test[:, choice]
        s = summarize(cost)
        rows.append(dict(risk_lambda=float(lam), selected_route='A_risky' if choice == 0 else 'B_safe',
                         train_objective=float(scores[choice]), mean_cost=float(s['mean']),
                         variance=float(s['variance']), cvar95=float(s['cvar95']),
                         worst_cost=float(cost.max())))
    return pd.DataFrame(rows)


def plot_outputs(results, controlled, plot_dir):
    plot_dir.mkdir(parents=True, exist_ok=True)
    risk = results[(results.algorithm == 'mean_variance') & (np.isclose(results.uncertainty_cv, .2))]
    fig, ax = plt.subplots(figsize=(9, 5))
    for family, group in risk.groupby('distribution'):
        ax.plot(group.risk_lambda, group.cvar95, marker='o', label=family)
    ax.set(xlabel='Risk aversion λ', ylabel='Held-out CVaR95', title='Risk-aversion sensitivity (CV=20%)')
    ax.legend(); fig.tight_layout(); fig.savefig(plot_dir/'lambda_cvar95.png', dpi=160); plt.close(fig)

    base = results[(results.algorithm.isin(['expected_dijkstra','robust_minimax']))]
    fig, ax = plt.subplots(figsize=(9, 5))
    for (family, algorithm), group in base.groupby(['distribution','algorithm']):
        ax.plot(group.uncertainty_cv*100, group.cvar95, marker='o', label=f'{family} / {algorithm}')
    ax.set(xlabel='Target uncertainty CV (%)', ylabel='Held-out CVaR95', title='Uncertainty sensitivity')
    ax.legend(fontsize=7, ncol=2); fig.tight_layout()
    fig.savefig(plot_dir/'uncertainty_cvar95.png', dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(controlled.risk_lambda, controlled.mean_cost, marker='o', label='Held-out mean')
    ax.plot(controlled.risk_lambda, controlled.cvar95, marker='s', label='Held-out CVaR95')
    ax.set(xlabel='Risk aversion λ', ylabel='Cost', title='Controlled risk–cost trade-off')
    ax.legend(); fig.tight_layout(); fig.savefig(plot_dir/'controlled_tradeoff.png', dpi=160); plt.close(fig)

    div = results.groupby(['distribution','uncertainty_cv']).path.nunique().reset_index(name='distinct_routes')
    fig, ax = plt.subplots(figsize=(9, 4))
    for family, group in div.groupby('distribution'):
        ax.plot(group.uncertainty_cv*100, group.distinct_routes, marker='o', label=family)
    ax.set(xlabel='Target uncertainty CV (%)', ylabel='Number of distinct selected routes',
           title='Route diversity across algorithms and risk aversions')
    ax.legend(); fig.tight_layout(); fig.savefig(plot_dir/'route_diversity.png', dpi=160); plt.close(fig)
    return div


def run_phase6(dataset='grid_25', families=FAMILIES, project_root=None,
               lambdas=LAMBDAS, uncertainties=UNCERTAINTIES, original_cv=.2,
               train_fraction=.7, seed=2027, k=40, robust_scenarios=150):
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[1]
    output = root/'data'/'results'/'phase6'
    plots = root/'outputs'/'plots'/'phase6'
    output.mkdir(parents=True, exist_ok=True)
    all_rows, split_info = [], {}
    for family in families:
        if family not in FAMILIES:
            raise ValueError(f'Unknown distribution: {family}')
        edges, samples, source, target = load_inputs(root, dataset, family)
        for cv in uncertainties:
            frame, split = evaluate_setting(edges, samples, source, target, family, cv,
                                            original_cv, lambdas, train_fraction, seed, k,
                                            robust_scenarios)
            all_rows.append(frame)
            split_info[f'{family}_cv{cv:.2f}'] = split
            print(f'{family:<10} CV={cv:.0%} train={split["train_indices"].__len__()} '
                  f'test={split["test_indices"].__len__()} candidates={split["candidate_count"]} '
                  f'distinct_routes={frame.path.nunique()}')
    results = pd.concat(all_rows, ignore_index=True)
    results.to_csv(output/f'{dataset}_sensitivity.csv', index=False)
    controlled = controlled_benchmark(lambdas=lambdas)
    controlled.to_csv(output/'controlled_risk_cost_benchmark.csv', index=False)
    diversity = plot_outputs(results, controlled, plots)
    diversity.to_csv(output/f'{dataset}_route_diversity.csv', index=False)
    (output/f'{dataset}_experiment_config.json').write_text(json.dumps({
        'seed':seed, 'train_fraction':train_fraction, 'lambdas':list(lambdas),
        'uncertainties':list(uncertainties), 'original_cv':original_cv,
        'candidate_k':k, 'robust_scenarios':robust_scenarios,
        'note':'Residual rescaling; not parametric resampling at each target CV. '
               'Minimax and mean-variance optimize over training-derived candidate paths only. '
               'Held-out scenarios are not used for route selection.',
        'splits':split_info}, indent=2), encoding='utf-8')
    print('PHASE 6 COMPLETED SUCCESSFULLY')
    return results
