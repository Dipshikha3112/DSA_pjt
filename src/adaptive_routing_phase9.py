"""Phase 9: leakage-free, validation-guided adaptive risk routing (research prototype)."""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
from src.benchmark_phase7 import graph_for, edge_table, sample_scenarios, shortest, path_indices

ALPHAS = (0.0, 0.1, 0.25, 0.5, 1.0)
METHODS = ('dijkstra', 'saa', 'mean_variance', 'minimax', 'cvar_fixed', 'vgarr')


def split_scenarios(samples, seed, fractions=(.5, .25, .25)):
    a = np.asarray(samples, dtype=float)
    if a.ndim != 2 or len(a) < 12 or not np.isfinite(a).all() or np.any(a < 0):
        raise ValueError('scenarios must be a finite nonnegative 2-D array with >=12 rows')
    if len(fractions) != 3 or any(x <= 0 for x in fractions) or not np.isclose(sum(fractions), 1):
        raise ValueError('invalid split fractions')
    perm = np.random.default_rng(seed).permutation(len(a))
    n1, n2 = int(len(a)*fractions[0]), int(len(a)*fractions[1])
    if min(n1, n2, len(a)-n1-n2) < 2:
        raise ValueError('each split requires >=2 scenarios')
    return a[perm[:n1]], a[perm[n1:n1+n2]], a[perm[n1+n2:]]


def cvar(costs, q=.95):
    """Empirical upper-tail CVaR as average of largest ceil((1-q)*n) costs."""
    x = np.asarray(costs, dtype=float)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all() or not 0 < q < 1:
        raise ValueError('invalid costs or quantile')
    k = max(1, int(np.ceil((1-q)*len(x))))
    return float(np.partition(x, len(x)-k)[-k:].mean())


def stats(costs):
    x = np.asarray(costs, dtype=float)
    return dict(mean_cost=float(x.mean()), variance=float(x.var()),
                var95=float(np.quantile(x, .95)), cvar95=cvar(x), worst_cost=float(x.max()))


def diverse_candidates(graph, edges, means, train, source, target, budget, seed):
    """Deterministic seed paths + stochastic reweighting + edge-penalty diversification."""
    if budget < 1:
        raise ValueError('budget must be >=1')
    rng = np.random.default_rng(seed)
    paths = []
    seen = set()
    def add(weights):
        p = shortest(graph, np.maximum(weights, 1e-8), edges, source, target)
        if p not in seen:
            seen.add(p)
            paths.append(p)
    add(means)
    for weights in (train.mean(0), train.std(0), train.max(0),
                    train.mean(0)+train.std(0), train.mean(0)+3*train.std(0)):
        if len(paths) < budget:
            add(weights)
    edge_index = {e:i for i,e in enumerate(edges)}
    for attempt in range(max(24, budget*16)):
        if len(paths) >= budget:
            break
        scale = rng.uniform(.15, 1.8)
        weights = means*rng.lognormal(0, scale, len(edges))
        if paths and attempt % 2 == 0:
            previous = paths[attempt % len(paths)]
            weights[path_indices(previous, edge_index)] *= rng.uniform(1.5, 5.0)
        add(weights)
    return paths[:budget]


def choose_routes(train_cost, valid_cost, paths, risk_grid=ALPHAS, q=.95, tolerance=.05):
    """Choose all routes without accessing held-out test costs.

    VGARR: validation-normalized cost+CVaR, subject to a validation mean-cost
    budget relative to the validation-best route. Alpha selected using validation
    stability proxy: split validation in two halves and minimize worst-half score.
    """
    tr, va = np.asarray(train_cost), np.asarray(valid_cost)
    if tr.ndim != 2 or va.ndim != 2 or tr.shape[1] != va.shape[1] or tr.shape[1] != len(paths):
        raise ValueError('cost arrays must align with candidate paths')
    if min(tr.shape[0], va.shape[0]) < 4 or np.any(tr < 0) or np.any(va < 0):
        raise ValueError('invalid training/validation costs')
    if tolerance < 0 or not risk_grid or any(a < 0 for a in risk_grid):
        raise ValueError('invalid risk settings')
    # Deterministic tie-breaking: first path, then lower score.
    train_mean, train_var = tr.mean(0), tr.var(0)
    train_cvar = np.array([cvar(tr[:,i], q) for i in range(tr.shape[1])])
    val_mean = va.mean(0)
    val_cvar = np.array([cvar(va[:,i], q) for i in range(va.shape[1])])
    best_val_mean = val_mean.min()
    feasible = np.where(val_mean <= best_val_mean*(1+tolerance)+1e-12)[0]
    # Standard baselines use training data only (except VGARR's model selection).
    out = dict(dijkstra=0, saa=int(np.argmin(train_mean)),
               mean_variance=int(np.argmin(train_mean + train_var)),
               minimax=int(np.argmin(tr.max(0))),
               cvar_fixed=int(np.argmin(train_mean + .5*train_cvar)))
    # Risk-budgeted adaptive selection: minimize held-out-proxy tail risk
    # subject to a predeclared validation mean-cost tolerance.
    # This avoids comparing incompatible objectives with different alpha.
    selected = int(feasible[np.argmin(val_cvar[feasible])])
    out['vgarr'] = selected
    return out, dict(selected_alpha=float('nan'),
                     validation_criterion=float(val_cvar[selected]),
                     feasible_count=len(feasible), candidate_count=len(paths))


def run_case(topology='grid', n=25, seed=1, distribution='gamma', budget=24,
             scenarios=800, cv=.4, tolerance=.05):
    if scenarios < 40 or budget < 1 or not 0 < cv <= .55:
        raise ValueError('invalid experiment configuration')
    start = time.perf_counter()
    graph = graph_for(topology, n, seed)
    edges, means, stds = edge_table(graph, cv)
    source, target = min(graph.nodes), max(graph.nodes)
    rng = np.random.default_rng(seed*1000003 + n*1009 + {'normal':1,'lognormal':2,'gamma':3,'uniform':4}[distribution])
    all_samples = sample_scenarios(means, stds, distribution, scenarios, rng)
    train, valid, test = split_scenarios(all_samples, seed+199)
    paths = diverse_candidates(graph, edges, means, train, source, target, budget, seed+301)
    index = {e:i for i,e in enumerate(edges)}
    def costs(s):
        return np.column_stack([s[:,path_indices(p, index)].sum(1) for p in paths])
    tc, vc, hc = costs(train), costs(valid), costs(test)
    selected, meta = choose_routes(tc, vc, paths, tolerance=tolerance)
    records = []
    for method, i in selected.items():
        row = dict(topology=topology, requested_n=n, actual_n=len(graph), seed=seed,
                   distribution=distribution, budget=budget, cv=cv, scenarios=scenarios,
                   n_train=len(train), n_valid=len(valid), n_test=len(test),
                   method=method, path_id=i, path='->'.join(map(str, paths[i])),
                   hops=len(paths[i])-1, **stats(hc[:,i]),
                   train_mean=float(tc[:,i].mean()), valid_mean=float(vc[:,i].mean()),
                   selected_alpha=meta['selected_alpha'] if method=='vgarr' else np.nan,
                   candidate_count=meta['candidate_count'], feasible_count=meta['feasible_count'])
        records.append(row)
    elapsed = time.perf_counter()-start
    for r in records:
        r['case_seconds'] = elapsed
    return records


def run_benchmark(output='data/results/phase9', quick=False, scenarios=800):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sizes = (25,) if quick else (25,49,100)
    seeds = (1,) if quick else (1,2,3)
    dists = ('gamma','lognormal') if quick else ('normal','lognormal','gamma','uniform')
    budgets = (8,) if quick else (8,24)
    rows=[]
    total=3*len(sizes)*len(seeds)*len(dists)*len(budgets)
    count=0
    for topology in ('grid','erdos_renyi','scale_free'):
        for n in sizes:
            for seed in seeds:
                for dist in dists:
                    for budget in budgets:
                        rows.extend(run_case(topology,n,seed,dist,budget,scenarios))
                        count+=1
                        if count % 12 == 0 or count == total:
                            print(f'Completed {count}/{total} cases',flush=True)
    df=pd.DataFrame(rows)
    df.to_csv(output/'algorithm_results.csv',index=False)
    keys=['topology','requested_n','seed','distribution','budget']
    base=df[df.method=='dijkstra'][keys+['mean_cost','cvar95','variance']].rename(
        columns={k:'baseline_'+k for k in ('mean_cost','cvar95','variance')})
    paired=df.merge(base,on=keys,validate='many_to_one')
    for k in ('mean_cost','cvar95','variance'):
        paired['delta_'+k]=paired[k]-paired['baseline_'+k]
    paired.to_csv(output/'paired_results.csv',index=False)
    summary=paired.groupby(['topology','method'],as_index=False)[['mean_cost','cvar95','delta_mean_cost','delta_cvar95','case_seconds']].mean()
    summary.to_csv(output/'summary.csv',index=False)
    config=dict(quick=quick,scenarios=scenarios,cases=total,records=len(df),
                evaluation='train/validation/test disjoint, selection never sees test',
                limitation='prototype: validation selects among candidate routes, not global optimality')
    (output/'experiment_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    return df
