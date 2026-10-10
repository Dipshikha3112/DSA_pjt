"""Phase 10.4: fresh-seed, paired candidate-budget and runtime audit.

Uses Phase 10.3's graph and scenario model; test scenarios are NEVER used to
construct paths or choose parameters. Independent graph seeds 901+.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from src.benchmark_phase7 import graph_for, edge_table, shortest
from src.phase103_risk_benchmark import (TOPOLOGIES, SHIFTS, heterogeneous_scenarios,
    mark_risky_edges, path_cost_matrix, tail_mean, choose)

BUDGETS = (4, 8, 12, 16, 24)
MODES = ('conventional','risk_informed')


def make_candidates(graph, edges, dev, weights, budget, seed, mode, attempt_cap=None):
    """Use the same number of shortest-path *attempts* for each method.

    Return unique routes (up to budget) and accounting. No test data accepted.
    Runtime is wall-clock for path search, including scoring/preprocessing.
    """
    if mode not in MODES or budget < 2: raise ValueError('bad mode or budget')
    if attempt_cap is None: attempt_cap = budget * 8
    if attempt_cap < 1: raise ValueError('attempt_cap must be positive')
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    source, target = min(graph.nodes), max(graph.nodes)
    weights = np.asarray(weights, float)
    std = dev.std(axis=0)
    tails = np.array([tail_mean(dev[:, j]) for j in range(dev.shape[1])])
    excess = tails - dev.mean(axis=0)
    anchor = tuple(shortest(graph, weights, edges, source, target))
    paths = {anchor: None}
    attempts = 0
    for attempt in range(attempt_cap):
        if len(paths) >= budget: break
        if mode == 'conventional':
            if attempt % 4 == 0: w = weights + std
            elif attempt % 4 == 1: w = weights + 3 * std
            else: w = weights * rng.lognormal(0, .7, len(edges))
        else:
            if attempt % 4 == 0: w = weights + tails
            elif attempt % 4 == 1: w = weights + 3 * excess
            elif attempt % 4 == 2: w = tails
            else: w = (weights + rng.uniform(.3,3.) * excess) * rng.lognormal(0,.35,len(edges))
        p = tuple(shortest(graph, np.maximum(w,1e-9), edges, source, target))
        paths[p] = None
        attempts += 1
    seconds = time.perf_counter() - start
    return list(paths), {'attempts':attempts,'seconds':seconds,
                          'unique_count':len(paths),'reached_budget':len(paths)>=budget}


def paired_ci(cluster_values, seed=1004, reps=3000):
    """Unadjusted percentile cluster bootstrap; one value per physical graph."""
    x=np.asarray(cluster_values,float)
    if x.ndim!=1 or len(x)<2 or not np.isfinite(x).all():
        raise ValueError('at least two finite graph clusters required')
    rng=np.random.default_rng(seed)
    samples=rng.choice(x,(reps,len(x)),replace=True).mean(axis=1)
    return float(x.mean()), [float(z) for z in np.quantile(samples,[.025,.975])]


def evaluate_graph(topology,n,seed,budgets,shifts,dev_count,test_count,eps,attempt_multiplier):
    graph=graph_for(topology,n,seed)
    edges, nominal, _=edge_table(graph,.4)
    risky=mark_risky_edges(graph,edges,seed+31000)
    risky |= (nominal < np.median(nominal)*.8)
    base_seed=seed*100003+n*211+TOPOLOGIES.index(topology)*101
    dev=heterogeneous_scenarios(nominal,risky,dev_count,base_seed,'iid_gamma')
    tests={shift:heterogeneous_scenarios(nominal,risky,test_count,
          base_seed+500000+SHIFTS.index(shift)*1009,shift) for shift in shifts}
    weights=dev.mean(axis=0)
    idx={e:i for i,e in enumerate(edges)}
    anchor=tuple(shortest(graph,weights,edges,min(graph.nodes),max(graph.nodes)))
    rows=[]; audit=[]
    for budget in budgets:
        selected={}
        for mode in MODES:
            # Identical attempt caps; achieved unique candidate counts may differ.
            paths, info=make_candidates(graph,edges,dev,weights,budget,base_seed+73,
                              mode,attempt_cap=budget*attempt_multiplier)
            dcost=path_cost_matrix(dev,paths,idx)
            t0=time.perf_counter()
            j,feasible,devmeans,devrisks=choose(dcost,eps)
            selection_s=time.perf_counter()-t0
            selected[mode]=paths[j]
            audit.append(dict(topology=topology,n=n,seed=seed,budget=budget,pool=mode,
                candidate_count=len(paths),feasible_count=feasible,
                unique_attempts=info['attempts'],reached_budget=info['reached_budget'],
                generation_seconds=info['seconds'],selection_seconds=selection_s,
                total_seconds=info['seconds']+selection_s,
                route_changed_vs_dijkstra=paths[j]!=anchor,
                selected_dev_mean=float(devmeans[j]),selected_dev_cvar95=float(devrisks[j])))
        allpaths=list(dict.fromkeys([anchor]+list(selected.values())))
        for shift in shifts:
            tc=path_cost_matrix(tests[shift],allpaths,idx)
            for mode in MODES:
                for method, route in [('dijkstra',anchor),('cvar_selected',selected[mode])]:
                    j=allpaths.index(route)
                    x=tc[:,j]
                    rows.append(dict(topology=topology,n=n,seed=seed,budget=budget,
                        shift=shift,pool=mode,method=method,route='->'.join(map(str,route)),
                        route_changed_vs_dijkstra=route!=anchor,
                        test_mean=float(x.mean()),test_cvar95=tail_mean(x),
                        test_p95=float(np.quantile(x,.95)),
                        test_max=float(np.max(x))))
    return rows,audit


def run(output='data/results/phase104_full',quick=False,dev_scenarios=1200,
        test_scenarios=3000,epsilon=.1,budgets=BUDGETS,attempt_multiplier=8):
    out=Path(output)
    if out.exists() and any(out.iterdir()): raise FileExistsError(f'Output directory not empty: {out}')
    if dev_scenarios<100 or test_scenarios<100 or not 0<=epsilon<=1:
        raise ValueError('invalid scenario count or epsilon')
    budgets=tuple(int(b) for b in budgets)
    if not budgets or any(b<2 for b in budgets) or len(set(budgets))!=len(budgets):
        raise ValueError('budgets must be distinct integers >=2')
    if attempt_multiplier<1: raise ValueError('attempt multiplier')
    sizes=(25,) if quick else (25,100)
    seeds=(901,) if quick else (901,902,903,904,905)
    shifts=('iid_gamma',) if quick else SHIFTS
    budgets=(budgets[0],) if quick else budgets
    rows=[]; audit=[]
    total=len(TOPOLOGIES)*len(sizes)*len(seeds)
    k=0
    for topo in TOPOLOGIES:
        for n in sizes:
            for seed in seeds:
                r,a=evaluate_graph(topo,n,seed,budgets,shifts,dev_scenarios,
                                   test_scenarios,epsilon,attempt_multiplier)
                rows.extend(r); audit.extend(a); k+=1
                if k%5==0 or k==total: print(f'Phase 10.4: {k}/{total} physical graphs',flush=True)
    df=pd.DataFrame(rows)
    adf=pd.DataFrame(audit)
    # Compare methods on matched topology, size, seed, budget and shift.
    piv=df.pivot(index=['topology','n','seed','budget','shift','pool'],
                 columns='method',values=['test_cvar95','test_mean']).reset_index()
    piv.columns=['_'.join(c).rstrip('_') for c in piv.columns]
    piv['delta_cvar95_cvar_minus_dijkstra']=piv['test_cvar95_cvar_selected']-piv['test_cvar95_dijkstra']
    piv['delta_mean_cvar_minus_dijkstra']=piv['test_mean_cvar_selected']-piv['test_mean_dijkstra']
    pair=df[df.method=='cvar_selected'].pivot(index=['topology','n','seed','budget','shift'],
                   columns='pool',values='test_cvar95').reset_index()
    pair['delta_risk_informed_minus_conventional']=pair['risk_informed']-pair['conventional']
    # Average over shifts for each physical graph, within budget and pool.
    clusters=piv.groupby(['topology','n','seed','budget','pool'],as_index=False).agg(
        delta_cvar95=('delta_cvar95_cvar_minus_dijkstra','mean'),
        delta_mean=('delta_mean_cvar_minus_dijkstra','mean'))
    pool_clusters=pair.groupby(['topology','n','seed','budget'],as_index=False).agg(
        delta_cvar95=('delta_risk_informed_minus_conventional','mean'))
    cis=[]
    if not quick:
        for (budget,pool),part in clusters.groupby(['budget','pool']):
            mean,ci=paired_ci(part.delta_cvar95.to_numpy(),seed=1004+int(budget))
            m2,ci2=paired_ci(part.delta_mean.to_numpy(),seed=2004+int(budget))
            cis.append(dict(comparison='cvar_minus_dijkstra',budget=budget,pool=pool,
                            n_graph_clusters=len(part),mean_delta_cvar95=mean,
                            ci95_low=ci[0],ci95_high=ci[1],mean_delta_test_mean=m2,
                            mean_cost_ci95_low=ci2[0],mean_cost_ci95_high=ci2[1]))
        for budget,part in pool_clusters.groupby('budget'):
            mean,ci=paired_ci(part.delta_cvar95.to_numpy(),seed=3004+int(budget))
            cis.append(dict(comparison='risk_informed_minus_conventional',budget=budget,
                            pool='paired',n_graph_clusters=len(part),mean_delta_cvar95=mean,
                            ci95_low=ci[0],ci95_high=ci[1],mean_delta_test_mean=np.nan,
                            mean_cost_ci95_low=np.nan,mean_cost_ci95_high=np.nan))
    out.mkdir(parents=True,exist_ok=True)
    df.to_csv(out/'algorithm_results.csv',index=False)
    adf.to_csv(out/'runtime_candidate_audit.csv',index=False)
    piv.to_csv(out/'paired_dijkstra_cvar.csv',index=False)
    pair.to_csv(out/'paired_pool_comparison.csv',index=False)
    clusters.to_csv(out/'graph_cluster_deltas.csv',index=False)
    pd.DataFrame(cis,columns=['comparison','budget','pool','n_graph_clusters',
        'mean_delta_cvar95','ci95_low','ci95_high','mean_delta_test_mean',
        'mean_cost_ci95_low','mean_cost_ci95_high']).to_csv(out/'cluster_bootstrap_ci.csv',index=False)
    summary=adf.groupby(['budget','pool'],as_index=False).agg(
        mean_candidates=('candidate_count','mean'),mean_feasible=('feasible_count','mean'),
        fraction_reaching_budget=('reached_budget','mean'),
        mean_generation_seconds=('generation_seconds','mean'),
        mean_total_seconds=('total_seconds','mean'),
        route_change_rate=('route_changed_vs_dijkstra','mean'))
    summary.to_csv(out/'budget_summary.csv',index=False)
    report=dict(phase='10.4',design='fresh-graph-seed exploratory validation',
        quick=quick,graph_clusters=total,scenario_shift_conditions=len(shifts),
        budgets=list(budgets),method_records=len(df),seeds=list(seeds),
        epsilon=epsilon,dev_scenarios=dev_scenarios,test_scenarios=test_scenarios,
        attempt_multiplier=attempt_multiplier,
        primary_comparison='CVaR-selected minus Dijkstra; negative CVaR delta favors CVaR',
        notes=['New graph seeds relative to Phases 10.1-10.3, but hypothesis is informed by those phases',
               'One graph is the statistical cluster; repeated shifts and budgets are not independent',
               'Unadjusted percentile bootstrap CIs are exploratory; multiple budgets and pools tested',
               'Equal attempt caps do not imply equal unique candidate counts or equal runtime',
               'Wall-clock runtimes are noisy; benchmark on same hardware and repeat for stable estimates',
               'CVaR selection uses development scenarios only; tests are held out',
               'Synthetic risk models and candidate-restricted optimization; no global optimality guarantee',
               'Common-shock shift changes marginals as well as correlation',
               'No automatic tuning based on test performance'])
    (out/'research_report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(f'Phase 10.4 complete: {total} physical graphs, {len(df)} method records; outputs: {out}')
    return report


def main():
    p=argparse.ArgumentParser(description='Phase 10.4 candidate-budget fairness and paired statistical validation')
    p.add_argument('--output',default='data/results/phase104_full')
    p.add_argument('--quick',action='store_true')
    p.add_argument('--dev-scenarios',type=int,default=1200)
    p.add_argument('--test-scenarios',type=int,default=3000)
    p.add_argument('--epsilon',type=float,default=.1)
    p.add_argument('--budgets',default='4,8,12,16,24')
    p.add_argument('--attempt-multiplier',type=int,default=8)
    a=p.parse_args()
    run(a.output,a.quick,a.dev_scenarios,a.test_scenarios,a.epsilon,
        tuple(int(x.strip()) for x in a.budgets.split(',')),a.attempt_multiplier)

if __name__=='__main__': main()
