"""Phase 10: preregistered synthetic distribution-shift experiment.

Training/calibration/selection samples are drawn ONLY from Gamma CV=0.4.
Fresh independent test samples are drawn from the named shift. All methods
are evaluated on the same test edge-cost matrix within a case.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from src.benchmark_phase7 import graph_for, edge_table, sample_scenarios, shortest
from src.adaptive_routing_phase9 import cvar, stats
from src.vgarr_phase91 import (candidates_91, route_costs, select_budget,
                               select_route, diversity)

METHODS = ('dijkstra','saa','cvar_fixed','vgarr_p9','fixed_05_diverse',
           'fixed_10_diverse','adaptive_diverse','adaptive_original')
SHIFTS = ('iid_gamma','lognormal','heavy_lognormal','correlated_gamma')
TRAIN_CV = 0.4
TEST_CV = {'iid_gamma':.4,'lognormal':.4,'heavy_lognormal':.7,'correlated_gamma':.4}


def generate_test_scenarios(means, shift, count, seed, shock_strength=.35):
    """Independent test scenarios, mean-preserving common multiplicative shock.

    correlated_gamma has Gamma CV .4 idiosyncratic edges, multiplied by a
    mean-one lognormal shared shock per scenario; marginal CV thus increases.
    This intentionally combines correlation and variance change and is NOT
    a pure-correlation intervention.
    """
    if shift not in SHIFTS or count < 100 or not 0 <= shock_strength <= 2:
        raise ValueError('invalid test shift configuration')
    rng = np.random.default_rng(seed)
    cv = TEST_CV[shift]
    distribution = 'lognormal' if shift in ('lognormal','heavy_lognormal') else 'gamma'
    samples = sample_scenarios(means, means*cv, distribution, count, rng)
    if shift == 'correlated_gamma':
        shocks = rng.lognormal(-shock_strength**2/2, shock_strength,
                               size=(count,1))
        samples *= shocks
    return samples


def training_pools(means, scenarios, seed):
    """Disjoint training, calibration, selection from independent Gamma draws."""
    if scenarios < 100:
        raise ValueError('need at least 100 development scenarios')
    rng = np.random.default_rng(seed)
    samples = sample_scenarios(means, means*TRAIN_CV, 'gamma', scenarios, rng)
    perm = rng.permutation(scenarios)
    a,b = int(.625*scenarios),int(.8125*scenarios)
    return samples[perm[:a]], samples[perm[a:b]], samples[perm[b:]]


def choose_methods(graph, edges, means, train, calibration, selection,
                   budget, seed):
    """No test data accepted; return route map and diagnostics."""
    source,target = min(graph.nodes),max(graph.nodes)
    original = candidates_91(graph,edges,means,train,source,target,budget,seed,'original')
    improved = candidates_91(graph,edges,means,train,source,target,budget,seed,'diverse')
    paths = list(dict.fromkeys(original+improved))
    lookup={p:i for i,p in enumerate(paths)}
    oi=np.array([lookup[p] for p in original],dtype=int)
    di=np.array([lookup[p] for p in improved],dtype=int)
    edge_index={e:i for i,e in enumerate(edges)}
    tr=route_costs(train,paths,edge_index)
    ca=route_costs(calibration,paths,edge_index)
    se=route_costs(selection,paths,edge_index)
    eps_o,_=select_budget(ca[:,oi]);eps_d,_=select_budget(ca[:,di])
    selections={
        'dijkstra':lookup[tuple(shortest(graph,means,edges,source,target))],
        'saa':int(np.argmin(tr.mean(0))),
        'cvar_fixed':int(np.argmin(tr.mean(0)+.5*np.array([cvar(tr[:,j]) for j in range(len(paths))])))
    }
    feasible={}; epsilons={}; candidate_count={}; candidate_diversity={}
    configs=(('vgarr_p9',oi,.05,original),('fixed_05_diverse',di,.05,improved),
             ('fixed_10_diverse',di,.10,improved),
             ('adaptive_diverse',di,eps_d,improved),('adaptive_original',oi,eps_o,original))
    for method,subset,eps,pool in configs:
        local,nfeasible=select_route(se[:,subset],eps)
        selections[method]=int(subset[local])
        feasible[method]=nfeasible
        epsilons[method]=float(eps)
        candidate_count[method]=len(pool)
        candidate_diversity[method]=diversity(pool)
    return paths, selections, feasible, epsilons, candidate_count, candidate_diversity


def run_case(topology='grid', n=25, seed=201, shift='iid_gamma',
             budget=8, dev_scenarios=800, test_count=3000):
    if shift not in SHIFTS or budget < 2 or dev_scenarios < 100 or test_count < 100:
        raise ValueError('invalid case')
    started=time.perf_counter()
    graph=graph_for(topology,n,seed)
    edges,means,_=edge_table(graph,TRAIN_CV)
    # Stable independent RNG streams for development and test; test seed
    # changes across shifts, never feeds back into candidate selection.
    base_seed=1000003 + seed*10007+n*101 + {'grid':1,'erdos_renyi':2,'scale_free':3}[topology]
    train,cal,selection=training_pools(means,dev_scenarios,base_seed)
    paths,chosen,feas,eps,cnt,div=choose_methods(
        graph,edges,means,train,cal,selection,budget,base_seed+19)
    held=generate_test_scenarios(means,shift,test_count,base_seed+1000000+SHIFTS.index(shift)*997)
    costs=route_costs(held,paths,{e:i for i,e in enumerate(edges)})
    rows=[]
    for method in METHODS:
        j=chosen[method]
        rows.append(dict(topology=topology,requested_n=n,actual_n=len(graph),seed=seed,
                         shift=shift,train_distribution='gamma',train_cv=TRAIN_CV,
                         test_cv=TEST_CV[shift],budget=budget,
                         dev_scenarios=dev_scenarios,n_train=len(train),n_cal=len(cal),
                         n_select=len(selection),n_test=test_count,method=method,
                         path_id=j,path='->'.join(map(str,paths[j])),
                         hops=len(paths[j])-1,epsilon=eps.get(method,np.nan),
                         feasible_count=feas.get(method,np.nan),
                         candidate_count=cnt.get(method,np.nan),
                         candidate_diversity=div.get(method,np.nan),
                         **stats(costs[:,j])))
    duration=time.perf_counter()-started
    for row in rows: row['case_seconds']=duration
    return rows


def paired_results(df, baseline='cvar_fixed'):
    keys=['topology','requested_n','seed','shift','budget']
    base=df[df.method==baseline][keys+['mean_cost','cvar95','variance','path_id']]
    if base.duplicated(keys).any() or len(base)*len(METHODS)!=len(df):
        raise ValueError('incomplete or duplicated benchmark cases')
    base=base.rename(columns={c:'baseline_'+c for c in ('mean_cost','cvar95','variance','path_id')})
    paired=df.merge(base,on=keys,validate='many_to_one')
    if len(paired)!=len(df): raise ValueError('unpaired cases')
    for metric in ('mean_cost','cvar95','variance'):
        paired['delta_'+metric]=paired[metric]-paired['baseline_'+metric]
    paired['route_changed']=paired.path_id!=paired.baseline_path_id
    return paired


def cluster_bootstrap(paired, metric='cvar95', draws=2000, seed=20261010):
    """Cluster by physical graph (topology, size, seed), aggregating shifts/budgets.

    Percentile CIs are descriptive; no unadjusted significance claims.
    """
    if draws < 100: raise ValueError('draws must be >=100')
    col='delta_'+metric
    cluster_cols=['topology','requested_n','seed']
    grouped=paired.groupby(cluster_cols+['method'],as_index=False)[col].mean()
    rng=np.random.default_rng(seed)
    rows=[]
    for method,g in grouped.groupby('method',sort=True):
        x=g[col].to_numpy(dtype=float)
        indices=rng.integers(0,len(x),size=(draws,len(x)))
        boot=x[indices].mean(axis=1)
        rows.append(dict(method=method,metric=metric,n_clusters=len(x),
                         mean_cluster_difference=float(x.mean()),
                         ci95_low=float(np.quantile(boot,.025)),
                         ci95_high=float(np.quantile(boot,.975))))
    return pd.DataFrame(rows)


def run_benchmark(output='data/results/phase10',quick=False,dev_scenarios=800,
                  test_count=3000,draws=2000):
    out=Path(output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Output directory not empty: {out}. Choose a new --output to preserve prior results.')
    out.mkdir(parents=True,exist_ok=True)
    topologies=('grid','erdos_renyi','scale_free')
    sizes=(25,) if quick else (25,100)
    seeds=(201,) if quick else (201,202,203)
    shifts=('iid_gamma','heavy_lognormal') if quick else SHIFTS
    budgets=(8,) if quick else (8,24)
    cases=len(topologies)*len(sizes)*len(seeds)*len(shifts)*len(budgets)
    rows=[]; completed=0
    for topology in topologies:
        for n in sizes:
            for seed in seeds:
                for shift in shifts:
                    for budget in budgets:
                        rows.extend(run_case(topology,n,seed,shift,budget,dev_scenarios,test_count))
                        completed+=1
                        if completed%12==0 or completed==cases:
                            print(f'Completed {completed}/{cases} cases',flush=True)
    df=pd.DataFrame(rows)
    paired=paired_results(df)
    df.to_csv(out/'algorithm_results.csv',index=False)
    paired.to_csv(out/'paired_vs_fixed_cvar.csv',index=False)
    summary=paired.groupby(['shift','method'],as_index=False).agg(
        mean_cost=('mean_cost','mean'),cvar95=('cvar95','mean'),
        delta_mean_cost=('delta_mean_cost','mean'),delta_cvar95=('delta_cvar95','mean'),
        route_changed=('route_changed','mean'))
    summary.to_csv(out/'shift_summary.csv',index=False)
    inf=pd.concat([cluster_bootstrap(paired,metric,draws) for metric in ('mean_cost','cvar95','variance')],ignore_index=True)
    inf.to_csv(out/'cluster_bootstrap_ci.csv',index=False)
    config=dict(quick=quick,cases=cases,records=len(df),methods=list(METHODS),
                train_distribution='gamma',train_cv=TRAIN_CV,
                shifts=list(shifts),test_cv={s:TEST_CV[s] for s in shifts},
                graph_seeds=list(seeds),sizes=list(sizes),candidate_budgets=list(budgets),
                dev_scenarios=dev_scenarios,generate_test_scenarios=test_count,
                development_split='62.5% train / 18.75% calibration / 18.75% selection',
                design_note='Phase 10 design chosen after observing Phase 9.x results; treat as exploratory. '
                            'No test shift data used in route selection. Common-shock shift also increases marginal CV. '
                            'Repeated budgets and shifts clustered by physical graph. '
                            'Bootstrap percentile intervals unadjusted; do not treat as multiplicity-adjusted significance.')
    (out/'experiment_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    print(f'PHASE 10 COMPLETED: {len(df)} algorithm records; {cases} cases')
    print('Outputs:',out)
    return df,paired,summary,inf


def main():
    parser=argparse.ArgumentParser(description='Phase 10 independent distribution-shift benchmark')
    parser.add_argument('--quick',action='store_true')
    parser.add_argument('--dev-scenarios',type=int,default=800)
    parser.add_argument('--test-scenarios',type=int,default=3000)
    parser.add_argument('--draws',type=int,default=2000)
    parser.add_argument('--output',default='data/results/phase10')
    args=parser.parse_args()
    run_benchmark(args.output,args.quick,args.dev_scenarios,args.test_scenarios,args.draws)
if __name__=='__main__': main()
