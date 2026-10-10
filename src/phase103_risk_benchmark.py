"""Phase 10.3: prospective synthetic risk-discriminative benchmark.

Requires existing src.benchmark_phase7.
Never uses held-out scenarios in candidate construction or selection.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
from src.benchmark_phase7 import graph_for, edge_table, shortest

Q = .95
TOPOLOGIES = ('grid', 'erdos_renyi', 'scale_free')
SHIFTS = ('iid_gamma', 'heavy_lognormal', 'correlated_gamma')


def tail_mean(x, q=Q):
    """Empirical upper-tail CVaR via upper order statistics (fractional boundary)."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError('invalid tail sample')
    n = len(x)
    k = (1-q)*n
    if k <= 0: raise ValueError('invalid q')
    z = np.sort(x)[::-1]
    full = int(np.floor(k))
    return float((z[:full].sum() + (k-full)*z[full]) / k)


def heterogeneous_scenarios(means, risky, count, seed, shift='iid_gamma'):
    """Independent edge-wise baseline with prespecified rare congestion on risky edges.

    Rare shocks are positive multipliers and increase risky-edge means. The
    true mean is NOT assumed equal to nominal means; development sample means
    determine expected-cost feasibility and Dijkstra weights.
    """
    if shift not in SHIFTS or count < 100: raise ValueError('invalid scenarios')
    means = np.asarray(means, float)
    risky = np.asarray(risky, bool)
    if len(means) != len(risky) or np.any(means <= 0): raise ValueError('invalid edge inputs')
    rng = np.random.default_rng(seed)
    cv = .4 if shift != 'heavy_lognormal' else .7
    if shift == 'iid_gamma':
        a = 1 / cv**2
        base = rng.gamma(a, means / a, size=(count, len(means)))
    else:
        sigma = np.sqrt(np.log1p(cv**2))
        base = rng.lognormal(np.log(means)-sigma**2/2, sigma, size=(count,len(means)))
    # Congestion probability and severity are fixed ex ante, not inferred from tests.
    events = rng.random((count, len(means))) < .035
    base *= 1 + events * risky[None,:] * 5.0
    if shift == 'correlated_gamma':
        # This is a combined common-shock/marginal-variance shift, not pure correlation.
        shock = rng.lognormal(-.35**2/2, .35, size=(count,1))
        base *= shock
    return base


def mark_risky_edges(graph, edges, seed, risky_fraction=.3):
    """Deterministic edge-risk assignments independent of test scenarios."""
    if not 0 < risky_fraction < 1: raise ValueError('risky_fraction')
    rng = np.random.default_rng(seed)
    return rng.random(len(edges)) < risky_fraction


def path_cost_matrix(scenarios, paths, edge_index):
    if not paths: raise ValueError('empty paths')
    result = np.zeros((len(scenarios), len(paths)), float)
    for j, path in enumerate(paths):
        idx = [edge_index[(u,v)] for u,v in zip(path[:-1],path[1:])]
        result[:,j] = scenarios[:,idx].sum(axis=1)
    return result


def pool(graph, edges, weights, development, budget, seed, mode):
    """Equal-size, equal-attempt path generators with a common Dijkstra anchor.

    Both use exactly budget*8 shortest-path calls, even if duplicates occur.
    Candidate budget is a cap on unique routes, not equal runtime or oracle power.
    """
    if mode not in ('conventional','risk_informed') or budget < 2: raise ValueError('invalid pool')
    rng = np.random.default_rng(seed)
    source, target = min(graph.nodes), max(graph.nodes)
    weights = np.asarray(weights,float)
    edge_std = development.std(axis=0)
    edge_tail = np.array([tail_mean(development[:,j]) for j in range(development.shape[1])])
    score = edge_tail - development.mean(axis=0)
    paths = {}
    anchor = shortest(graph, weights, edges, source, target)
    paths[anchor] = None
    for attempt in range(budget*8):
        if mode == 'conventional':
            if attempt % 4 == 0: w = weights + edge_std
            elif attempt % 4 == 1: w = weights + 3*edge_std
            else: w = weights*rng.lognormal(0, .7, len(edges))
        else:
            if attempt % 4 == 0: w = weights + edge_tail
            elif attempt % 4 == 1: w = weights + 3*score
            elif attempt % 4 == 2: w = edge_tail
            else: w = (weights + rng.uniform(.3,3.0)*score)*rng.lognormal(0,.35,len(edges))
        p = shortest(graph, np.maximum(w,1e-9), edges, source, target)
        paths[p] = None
        if len(paths) >= budget: break
    return list(paths)[:budget]


def choose(dev_costs, eps=.1):
    if not 0 <= eps <= 1: raise ValueError('epsilon')
    mean = dev_costs.mean(axis=0)
    risk = np.array([tail_mean(dev_costs[:,j]) for j in range(dev_costs.shape[1])])
    feasible = mean <= mean.min()*(1+eps)+1e-9
    j = int(np.argmin(np.where(feasible,risk,np.inf)))
    return j, int(feasible.sum()), mean, risk


def evaluate_case(topology, n, seed, budget, shift, dev_count, test_count, eps):
    graph = graph_for(topology,n,seed)
    edges, nominal, _ = edge_table(graph,.4)
    risky = mark_risky_edges(graph,edges,seed+31000)
    # Exogenous risk pattern: cheaper nominal edges are somewhat more likely risky.
    median = np.median(nominal)
    risky |= (nominal < median*.8)
    base_seed = seed*100003 + n*211 + TOPOLOGIES.index(topology)*101
    dev = heterogeneous_scenarios(nominal,risky,dev_count,base_seed, 'iid_gamma')
    # Training and held-out test are independently sampled; dev data alone used for decisions.
    held = heterogeneous_scenarios(nominal,risky,test_count,base_seed+500000+SHIFTS.index(shift)*1009,shift)
    weights = dev.mean(axis=0)
    idx = {e:i for i,e in enumerate(edges)}
    conv = pool(graph,edges,weights,dev,budget,base_seed+73,'conventional')
    informed = pool(graph,edges,weights,dev,budget,base_seed+73,'risk_informed')
    anchor = tuple(shortest(graph,weights,edges,min(graph.nodes),max(graph.nodes)))
    allpaths = list(dict.fromkeys(conv+informed))
    dc = path_cost_matrix(dev,allpaths,idx)
    tc = path_cost_matrix(held,allpaths,idx)
    look = {p:i for i,p in enumerate(allpaths)}
    rows=[]; diagnostics=[]
    for mode, pp in [('conventional',conv),('risk_informed',informed)]:
        cols = [look[p] for p in pp]
        j, nfeas, means, risks = choose(dc[:,cols],eps)
        chosen = cols[j]
        for method, global_j in [('dijkstra',look[anchor]),('cvar_selected',chosen)]:
            x = tc[:,global_j]
            rows.append(dict(topology=topology,n=n,seed=seed,budget=budget,shift=shift,
                pool=mode,method=method,route='->'.join(map(str,allpaths[global_j])),
                route_changed_vs_dijkstra=bool(global_j!=look[anchor]),
                test_mean=float(x.mean()),test_cvar95=tail_mean(x),
                dev_mean=float(dc[:,global_j].mean()),dev_cvar95=tail_mean(dc[:,global_j]),
                candidate_count=len(pp),feasible_count=nfeas))
        diagnostics.append(dict(topology=topology,n=n,seed=seed,budget=budget,shift=shift,pool=mode,
            candidate_count=len(pp),feasible_count=nfeas,unique_vs_other=len(set(pp)-set(informed if mode=='conventional' else conv)),
            anchor_present=anchor in pp,selected_route= '->'.join(map(str,allpaths[chosen]))))
    return rows,diagnostics


def run(output='data/results/phase103',quick=False,dev_count=1200,test_count=3000,eps=.1,budget=12):
    out = Path(output)
    if out.exists() and any(out.iterdir()): raise FileExistsError(f'Output directory not empty: {out}')
    if dev_count < 100 or test_count < 100: raise ValueError('scenario counts too small')
    out.mkdir(parents=True,exist_ok=True)
    sizes=(25,) if quick else (25,100)
    seeds=(701,) if quick else (701,702,703,704,705)
    shifts=('iid_gamma',) if quick else SHIFTS
    rows=[]; diag=[]
    total=len(TOPOLOGIES)*len(sizes)*len(seeds)*len(shifts)
    k=0
    for topo in TOPOLOGIES:
        for n in sizes:
            for seed in seeds:
                for shift in shifts:
                    a,b=evaluate_case(topo,n,seed,budget,shift,dev_count,test_count,eps)
                    rows+=a;diag+=b;k+=1
                    if k%10==0 or k==total: print(f'Phase 10.3: {k}/{total} cases',flush=True)
    df=pd.DataFrame(rows)
    df.to_csv(out/'algorithm_results.csv',index=False)
    pd.DataFrame(diag).to_csv(out/'candidate_diagnostics.csv',index=False)
    paired=df[df.method=='cvar_selected'].pivot(index=['topology','n','seed','budget','shift'],columns='pool',values='test_cvar95').reset_index()
    paired['delta_risk_informed_minus_conventional']=paired['risk_informed']-paired['conventional']
    paired.to_csv(out/'paired_pool_comparison.csv',index=False)
    # Cluster first by physical graph to avoid treating multiple shifts as independent.
    cluster=paired.groupby(['topology','n','seed'],as_index=False).agg(
        delta_cvar95=('delta_risk_informed_minus_conventional','mean'))
    cluster.to_csv(out/'graph_cluster_deltas.csv',index=False)
    summary=df.groupby(['shift','pool','method'],as_index=False).agg(
        mean_cvar95=('test_cvar95','mean'),mean_cost=('test_mean','mean'),
        route_change_rate=('route_changed_vs_dijkstra','mean'),
        mean_feasible=('feasible_count','mean'))
    summary.to_csv(out/'shift_summary.csv',index=False)
    rng=np.random.default_rng(88101)
    values=cluster.delta_cvar95.to_numpy()
    if len(values)>1:
        boots=np.mean(rng.choice(values,(2000,len(values)),replace=True),axis=1)
        interval=[float(v) for v in np.quantile(boots,[.025,.975])]
    else: interval=[None,None]
    report=dict(phase='10.3',design='prospective synthetic benchmark',quick=quick,
        graph_clusters=len(cluster),cases=total,records=len(df),
        mean_cluster_delta=float(values.mean()),bootstrap_95_ci=interval,
        interpretation='Negative delta favors risk-informed candidates; CI is exploratory and unadjusted.',
        caveats=['Synthetic heterogeneous risk and rare shocks; not real-world evidence',
                 'Dijkstra anchor based on development sample means',
                 'Same candidate count cap and path-search attempt cap, not matched runtime',
                 'No globally optimal path benchmark; candidate-restricted comparison',
                 'No held-out test labels used in candidate construction or selection',
                 'Phase 10.3 hypotheses informed by earlier inspected results; not independent preregistration',
                 'Common shock changes marginal variance as well as correlation',
                 'Five graph seeds per topology/size in full run; larger validation still needed'])
    (out/'research_report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(f'Phase 10.3 complete: {total} cases, {len(df)} method records, {len(cluster)} graph clusters; outputs: {out}')
    return report


def main():
    p=argparse.ArgumentParser(description='Phase 10.3 prospective risk-discriminative candidate benchmark')
    p.add_argument('--output',default='data/results/phase103')
    p.add_argument('--quick',action='store_true')
    p.add_argument('--dev-scenarios',type=int,default=1200)
    p.add_argument('--test-scenarios',type=int,default=3000)
    p.add_argument('--epsilon',type=float,default=.1)
    p.add_argument('--budget',type=int,default=12)
    a=p.parse_args()
    run(a.output,a.quick,a.dev_scenarios,a.test_scenarios,a.epsilon,a.budget)

if __name__=='__main__': main()
