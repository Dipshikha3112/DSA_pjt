"""Phase 9.1: exploratory diversity-aware, validation-calibrated risk routing.

Uses independent scenario pools: train -> candidate creation; calibration ->
select budget; selection -> choose route; test -> report only. No test leakage.
This is a research prototype, not a claim of a novel global optimizer.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import networkx as nx
import numpy as np
import pandas as pd
from src.benchmark_phase7 import graph_for, edge_table, sample_scenarios, shortest, path_indices
from src.adaptive_routing_phase9 import cvar, stats, diverse_candidates

BUDGET_GRID = (0.02, 0.05, 0.10, 0.15)
METHODS = ('dijkstra', 'saa', 'cvar_fixed', 'vgarr_p9', 'fixed_05_diverse',
           'fixed_10_diverse', 'adaptive_diverse', 'adaptive_original')


def split_four(samples, seed):
    """50% training, 15% calibration, 15% selection, 20% final test."""
    a = np.asarray(samples, dtype=float)
    if a.ndim != 2 or len(a) < 40 or not np.isfinite(a).all() or np.any(a < 0):
        raise ValueError('expected >=40 finite nonnegative scenario rows')
    idx = np.random.default_rng(seed).permutation(len(a))
    n = len(a); cuts = [int(n*.5), int(n*.65), int(n*.8)]
    groups = np.split(idx, cuts)
    if min(map(len, groups)) < 4:
        raise ValueError('split too small')
    return tuple(a[x] for x in groups), tuple(tuple(map(int, x)) for x in groups)


def edge_overlap(a, b):
    aa, bb = set(zip(a[:-1], a[1:])), set(zip(b[:-1], b[1:]))
    return len(aa & bb) / len(aa | bb) if aa | bb else 1.0


def diversity(paths):
    if len(paths) < 2:
        return 0.0
    return float(np.mean([1-edge_overlap(paths[i],paths[j])
                          for i in range(len(paths)) for j in range(i+1,len(paths))]))


def candidates_91(graph, edges, means, train, source, target, budget, seed,
                  generator='diverse'):
    """Select diverse candidates from deterministic, Yen-k-shortest and perturbed paths.

    Original Phase 9 generator is available as a controlled ablation. All
    candidate generation uses training and known edge means, never validation/test.
    """
    if budget < 1 or generator not in ('diverse','original'):
        raise ValueError('invalid budget or generator')
    if generator == 'original':
        return diverse_candidates(graph, edges, means, train, source, target, budget, seed)
    rng = np.random.default_rng(seed)
    index = {e:i for i,e in enumerate(edges)}
    pool = []
    seen = set()
    def add(path):
        p = tuple(path)
        if p not in seen and len(p) >= 2:
            seen.add(p); pool.append(p)
    add(shortest(graph, means, edges, source, target))
    for w in (train.mean(0), train.mean(0)+train.std(0), train.mean(0)+2*train.std(0),
              train.max(0)):
        add(shortest(graph, w, edges, source, target))
    g = graph.copy()
    for (u,v), w in zip(edges, means):
        g[u][v]['phase91_weight'] = float(w)
    try:
        yen = nx.shortest_simple_paths(g, source, target, weight='phase91_weight')
        for _, p in zip(range(min(5*budget, 120)), yen):
            add(p)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        pass
    for k in range(max(60, budget*14)):
        if len(pool) >= max(8*budget, 80):
            break
        perturb = means * rng.lognormal(0, rng.uniform(.1,1.3), len(edges))
        if k % 2 == 0 and pool:
            reference = pool[k % len(pool)]
            perturb[path_indices(reference, index)] *= rng.uniform(1.5,5)
        add(shortest(graph, perturb, edges, source, target))
    # Greedy farthest-first with cost regularization; preserve mean shortest path.
    selected = [pool[0]]
    available = pool[1:]
    min_mean = float(means[path_indices(pool[0], index)].sum())
    while available and len(selected) < budget:
        scores=[]
        for p in available:
            overlap_div = min(1-edge_overlap(p, s) for s in selected)
            expected = float(means[path_indices(p, index)].sum())
            cost_ratio = expected / max(min_mean,1e-12) - 1
            scores.append(overlap_div - .65*max(cost_ratio,0))
        pick = int(np.argmax(scores))
        selected.append(available.pop(pick))
    return selected


def route_costs(scenarios, paths, index):
    return np.column_stack([scenarios[:,path_indices(p,index)].sum(axis=1) for p in paths])


def feasible_indices(costs, epsilon):
    mu = np.asarray(costs).mean(0)
    return np.flatnonzero(mu <= mu.min()*(1+epsilon)+1e-10)


def select_budget(calibration_costs, budgets=BUDGET_GRID, q=.95):
    """Select epsilon using calibration only.

    Calibration is split into two deterministic halves. For each epsilon,
    select a low-CVaR feasible route in the first half and score its CVaR on
    the second half, plus a fixed mean-cost penalty. No test/selection data.
    """
    a = np.asarray(calibration_costs, dtype=float)
    if a.ndim != 2 or a.shape[0] < 8 or a.shape[1] < 1 or not np.isfinite(a).all():
        raise ValueError('invalid calibration cost matrix')
    if not budgets or any(not 0 <= x <= 1 for x in budgets):
        raise ValueError('invalid budget grid')
    mid = len(a)//2
    first, second = a[:mid], a[mid:]
    baseline_mean = float(second.mean(0).min())
    baseline_cvar = min(cvar(second[:,j],q) for j in range(a.shape[1]))
    scores=[]
    for eps in budgets:
        feasible=feasible_indices(first,eps)
        selected=int(feasible[np.argmin([cvar(first[:,j],q) for j in feasible])])
        mean_penalty=max(0.,float(second[:,selected].mean())/max(baseline_mean,1e-12)-1)
        score=cvar(second[:,selected],q)/max(baseline_cvar,1e-12)+2.0*mean_penalty
        scores.append((float(score),float(eps),selected))
    return min(scores, key=lambda x:(x[0],x[1]))[1], scores


def select_route(selection_costs, epsilon, q=.95):
    feasible=feasible_indices(selection_costs,epsilon)
    selected=int(feasible[np.argmin([cvar(selection_costs[:,j],q) for j in feasible])])
    return selected, int(len(feasible))


def run_case(topology='grid', n=25, seed=101, distribution='gamma', budget=24,
             scenarios=800, cv=.4):
    if scenarios < 80 or budget < 2 or not 0 < cv <= .55:
        raise ValueError('invalid case configuration')
    start=time.perf_counter()
    graph=graph_for(topology,n,seed)
    edges,means,stds=edge_table(graph,cv)
    source,target=min(graph.nodes),max(graph.nodes)
    dist_id={'normal':1,'lognormal':2,'gamma':3,'uniform':4}[distribution]
    rng=np.random.default_rng(91000019 + seed*100003 + n*1009 + dist_id)
    samples=sample_scenarios(means,stds,distribution,scenarios,rng)
    (train,cal,sel,test), split_ids=split_four(samples,seed+8103)
    original=candidates_91(graph,edges,means,train,source,target,budget,seed+90,'original')
    improved=candidates_91(graph,edges,means,train,source,target,budget,seed+90,'diverse')
    # Union ensures the comparison methods see an identical route universe;
    # candidate-generator ablations are restricted to their own subsets.
    paths=list(dict.fromkeys(original+improved))
    ids={p:i for i,p in enumerate(paths)}
    oi=np.array([ids[p] for p in original],dtype=int)
    di=np.array([ids[p] for p in improved],dtype=int)
    index={e:i for i,e in enumerate(edges)}
    tc,cc,sc,hc=(route_costs(x,paths,index) for x in (train,cal,sel,test))
    eps_o,_=select_budget(cc[:,oi])
    eps_d,_=select_budget(cc[:,di])
    # All validation-based methods use the same selection partition.
    choices={
        'dijkstra':ids[original[0]],
        'saa':int(np.argmin(tc.mean(0))),
        'cvar_fixed':int(np.argmin(tc.mean(0)+.5*np.array([cvar(tc[:,j]) for j in range(len(paths))]))),
    }
    def choose(subset,costs,eps):
        local,count=select_route(costs[:,subset],eps)
        return int(subset[local]),count
    choices['vgarr_p9'],f_p9=choose(oi,sc,.05)
    choices['fixed_05_diverse'],f05=choose(di,sc,.05)
    choices['fixed_10_diverse'],f10=choose(di,sc,.10)
    choices['adaptive_diverse'],fad=choose(di,sc,eps_d)
    choices['adaptive_original'],fao=choose(oi,sc,eps_o)
    feas={'vgarr_p9':f_p9,'fixed_05_diverse':f05,'fixed_10_diverse':f10,
          'adaptive_diverse':fad,'adaptive_original':fao}
    rows=[]
    for method, j in choices.items():
        row=dict(topology=topology,requested_n=n,actual_n=len(graph),seed=seed,
                 distribution=distribution,budget=budget,cv=cv,scenarios=scenarios,
                 n_train=len(train),n_cal=len(cal),n_select=len(sel),n_test=len(test),
                 method=method,path_id=j,path='->'.join(map(str,paths[j])),
                 hops=len(paths[j])-1,**stats(hc[:,j]),
                 train_mean=float(tc[:,j].mean()),selection_mean=float(sc[:,j].mean()),
                 epsilon=eps_d if method=='adaptive_diverse' else eps_o if method=='adaptive_original' else
                 .05 if method in ('vgarr_p9','fixed_05_diverse') else .10 if method=='fixed_10_diverse' else np.nan,
                 candidate_count=len(di) if method in ('fixed_05_diverse','fixed_10_diverse','adaptive_diverse') else len(oi),
                 feasible_count=feas.get(method,np.nan),
                 candidate_diversity=diversity(improved if method in ('fixed_05_diverse','fixed_10_diverse','adaptive_diverse') else original))
        rows.append(row)
    elapsed=time.perf_counter()-start
    for r in rows: r['case_seconds']=elapsed
    return rows


def run_benchmark(output='data/results/phase91',quick=False,scenarios=800):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    sizes=(25,) if quick else (25,49,100)
    seeds=(101,) if quick else (101,102,103,104,105)
    dists=('gamma','lognormal') if quick else ('normal','lognormal','gamma','uniform')
    budgets=(8,) if quick else (8,24)
    rows=[]; total=3*len(sizes)*len(seeds)*len(dists)*len(budgets)
    k=0
    for topology in ('grid','erdos_renyi','scale_free'):
        for n in sizes:
            for seed in seeds:
                for dist in dists:
                    for budget in budgets:
                        rows.extend(run_case(topology,n,seed,dist,budget,scenarios))
                        k+=1
                        if k%20==0 or k==total: print(f'Completed {k}/{total} cases',flush=True)
    df=pd.DataFrame(rows)
    df.to_csv(out/'algorithm_results.csv',index=False)
    keys=['topology','requested_n','seed','distribution','budget']
    base=df[df.method=='dijkstra'][keys+['mean_cost','cvar95','variance','path_id']].rename(
        columns={k:'baseline_'+k for k in ('mean_cost','cvar95','variance','path_id')})
    paired=df.merge(base,on=keys,validate='many_to_one')
    for k in ('mean_cost','cvar95','variance'):
        paired['delta_'+k]=paired[k]-paired['baseline_'+k]
    paired['route_changed']=paired.path_id!=paired.baseline_path_id
    paired.to_csv(out/'paired_results.csv',index=False)
    summary=paired.groupby(['topology','method'],as_index=False).agg(
        mean_cost=('mean_cost','mean'),cvar95=('cvar95','mean'),
        delta_mean_cost=('delta_mean_cost','mean'),delta_cvar95=('delta_cvar95','mean'),
        changed_fraction=('route_changed','mean'),feasible_count=('feasible_count','mean'),
        diversity=('candidate_diversity','mean'))
    summary.to_csv(out/'summary.csv',index=False)
    config=dict(quick=quick,scenarios=scenarios,cases=total,records=len(df),
                seeds=list(seeds),risk_budgets=list(BUDGET_GRID),
                split='50% train / 15% calibration / 15% selection / 20% test',
                warning='Exploratory Phase 9.1; Phase 9 test findings influenced this design. '
                        'Fresh seeds mitigate but do not eliminate adaptive researcher overfitting. '
                        'No global optimality or novelty claim. '
                        'All candidates share one test scenario matrix per case; '
                        'repeated budgets and distributions for each graph seed are dependent.')
    (out/'experiment_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    return df
