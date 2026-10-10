"""Phase 10.1: route-level audit and exploratory shift-stress selection.

Requires the existing Phase 7, 9, 9.1, and 10 source modules. All route
selection occurs before held-out evaluation. Not a novelty/superiority claim.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.benchmark_phase7 import graph_for, edge_table, shortest
from src.adaptive_routing_phase9 import cvar, stats
from src.vgarr_phase91 import candidates_91, route_costs, edge_overlap
from src.distribution_shift_phase10 import (METHODS, SHIFTS, TRAIN_CV, TEST_CV,
    training_pools, generate_test_scenarios, choose_methods, cluster_bootstrap)

EXTRA_METHOD = 'shift_stress_cvar'
ALL_METHODS = METHODS + (EXTRA_METHOD,)

def stress_environments(means, count, seed, shock_strength=.35):
    """Development-only, prespecified plausible stress scenarios.

    Environments use Gamma(CV=.4), Lognormal(CV=.7), and shared-shock
    Gamma(CV=.4). These are hypotheses about future uncertainty, not
    learned shift probabilities or pure-correlation interventions.
    """
    if count < 100 or np.any(np.asarray(means) <= 0):
        raise ValueError('invalid stress parameters')
    return {name: generate_test_scenarios(means, name, count, seed + 109*i,
                                           shock_strength=shock_strength)
            for i, name in enumerate(('iid_gamma','heavy_lognormal','correlated_gamma'))}

def select_shift_stress(selection_costs, stress_costs, epsilon=.10, lam=.25, q=.95):
    """Minimize CVaR(selection) + lam * max(0, worst_stress_CVaR - CVaR(selection)).

    Mean feasibility is measured ONLY on the Gamma selection partition.
    Returns (index, diagnostic frame). All tuning values must be fixed before
    inspecting held-out results. This is a candidate-restricted heuristic.
    """
    s = np.asarray(selection_costs, dtype=float)
    if s.ndim != 2 or s.shape[0] < 20 or s.shape[1] < 1 or not np.isfinite(s).all():
        raise ValueError('invalid selection costs')
    if not (0 <= epsilon <= 1 and lam >= 0 and .5 <= q < 1):
        raise ValueError('invalid selection parameters')
    stress = {k: np.asarray(v,dtype=float) for k,v in stress_costs.items()}
    if not stress or any(v.ndim != 2 or v.shape[1] != s.shape[1] or v.shape[0] < 20 or not np.isfinite(v).all() for v in stress.values()):
        raise ValueError('invalid stress matrices')
    mean = s.mean(axis=0)
    feasible = mean <= mean.min()*(1+epsilon)+1e-10
    base = np.array([cvar(s[:,j],q) for j in range(s.shape[1])])
    worst = np.maximum.reduce([np.array([cvar(v[:,j],q) for j in range(s.shape[1])]) for v in stress.values()])
    penalty = np.maximum(0.,worst-base)
    objective = base+lam*penalty
    rank = np.where(feasible,objective,np.inf)
    chosen = int(np.argmin(rank))
    return chosen, pd.DataFrame({'candidate_id':np.arange(s.shape[1]),
        'selection_mean':mean,'selection_cvar95':base,
        'worst_stress_cvar95':worst,'shift_penalty':penalty,
        'objective':objective,'feasible':feasible})

def run_case(topology='grid',n=25,seed=201,shift='iid_gamma',budget=8,
             dev_scenarios=800,test_count=3000,stress_count=500,
             epsilon=.10,lam=.25):
    if topology not in ('grid','erdos_renyi','scale_free') or shift not in SHIFTS or budget < 2 or dev_scenarios < 100 or test_count < 100:
        raise ValueError('invalid case configuration')
    graph=graph_for(topology,n,seed)
    edges,means,_=edge_table(graph,TRAIN_CV)
    base_seed=1000003 + seed*10007+n*101 + {'grid':1,'erdos_renyi':2,'scale_free':3}[topology]
    train,cal,selection=training_pools(means,dev_scenarios,base_seed)
    paths,chosen,feasible,eps,counts,diversity=choose_methods(graph,edges,means,train,cal,selection,budget,base_seed+19)
    index={e:i for i,e in enumerate(edges)}
    sel_costs=route_costs(selection,paths,index)
    stress=stress_environments(means,stress_count,base_seed+555555)
    stress_costs={name:route_costs(matrix,paths,index) for name,matrix in stress.items()}
    selected,diag=select_shift_stress(sel_costs,stress_costs,epsilon,lam)
    chosen[EXTRA_METHOD]=selected
    held=generate_test_scenarios(means,shift,test_count,base_seed+1000000+SHIFTS.index(shift)*997)
    held_costs=route_costs(held,paths,index)
    base_idx=chosen['dijkstra']
    base_edges=set(zip(paths[base_idx][:-1],paths[base_idx][1:]))
    path_audit=[]
    train_costs=route_costs(train,paths,index)
    for j,p in enumerate(paths):
        row=diag.iloc[j]
        path_audit.append(dict(topology=topology,requested_n=n,seed=seed,shift=shift,budget=budget,
            candidate_id=j,path='->'.join(map(str,p)),hops=len(p)-1,
            edge_overlap_dijkstra=float(edge_overlap(p,paths[base_idx])),
            train_mean=float(train_costs[:,j].mean()),
            selection_mean=float(row.selection_mean),selection_cvar95=float(row.selection_cvar95),
            worst_stress_cvar95=float(row.worst_stress_cvar95),shift_penalty=float(row.shift_penalty),
            shift_objective=float(row.objective),shift_feasible=bool(row.feasible),
            test_mean=float(held_costs[:,j].mean()),test_cvar95=float(cvar(held_costs[:,j]))))
    result=[]
    for method in ALL_METHODS:
        j=chosen[method]
        result.append(dict(topology=topology,requested_n=n,actual_n=len(graph),seed=seed,
            shift=shift,budget=budget,method=method,path_id=j,path='->'.join(map(str,paths[j])),
            route_changed_vs_dijkstra=bool(j!=base_idx),
            epsilon=epsilon if method==EXTRA_METHOD else eps.get(method,np.nan),
            feasible_count=int(diag.feasible.sum()) if method==EXTRA_METHOD else feasible.get(method,np.nan),
            candidate_count=len(paths) if method==EXTRA_METHOD else counts.get(method,np.nan),
            **stats(held_costs[:,j])))
    return result,path_audit

def paired(df,baseline='cvar_fixed'):
    keys=['topology','requested_n','seed','shift','budget']
    if df.duplicated(keys+['method']).any() or set(df.method)!=set(ALL_METHODS):
        raise ValueError('duplicate/incomplete methods')
    if (df.groupby(keys).method.nunique()!=len(ALL_METHODS)).any():
        raise ValueError('incomplete case')
    ref=df[df.method==baseline][keys+['mean_cost','cvar95','variance','path_id']]
    ref=ref.rename(columns={k:'baseline_'+k for k in ('mean_cost','cvar95','variance','path_id')})
    joined=df.merge(ref,on=keys,validate='many_to_one')
    for m in ('mean_cost','cvar95','variance'):
        joined['delta_'+m]=joined[m]-joined['baseline_'+m]
    joined['route_changed_vs_baseline']=joined.path_id!=joined.baseline_path_id
    return joined

def run_benchmark(output='data/results/phase101',quick=False,dev_scenarios=800,
                  test_count=3000,stress_count=500,epsilon=.10,lam=.25,draws=2000):
    out=Path(output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Output directory not empty: {out}')
    out.mkdir(parents=True,exist_ok=True)
    sizes=(25,) if quick else (25,100)
    seeds=(201,) if quick else (201,202,203)
    shifts=('iid_gamma','heavy_lognormal') if quick else SHIFTS
    budgets=(8,) if quick else (8,24)
    rows=[]; audits=[]
    total=3*len(sizes)*len(seeds)*len(shifts)*len(budgets)
    k=0
    for topology in ('grid','erdos_renyi','scale_free'):
        for n in sizes:
            for seed in seeds:
                for shift in shifts:
                    for budget in budgets:
                        r,a=run_case(topology,n,seed,shift,budget,dev_scenarios,test_count,stress_count,epsilon,lam)
                        rows.extend(r);audits.extend(a);k+=1
                        if k%12==0 or k==total: print(f'Phase 10.1: {k}/{total} cases',flush=True)
    df=pd.DataFrame(rows); audit=pd.DataFrame(audits); matched=paired(df)
    df.to_csv(out/'algorithm_results.csv',index=False)
    audit.to_csv(out/'candidate_audit.csv',index=False)
    matched.to_csv(out/'paired_vs_fixed_cvar.csv',index=False)
    summary=matched.groupby(['shift','method'],as_index=False).agg(
        mean_cost=('mean_cost','mean'),cvar95=('cvar95','mean'),
        delta_mean_cost=('delta_mean_cost','mean'),delta_cvar95=('delta_cvar95','mean'),
        changed_vs_dijkstra=('route_changed_vs_dijkstra','mean'),
        changed_vs_fixed_cvar=('route_changed_vs_baseline','mean'))
    summary.to_csv(out/'shift_summary.csv',index=False)
    intervals=pd.concat([cluster_bootstrap(matched,m,draws) for m in ('mean_cost','cvar95','variance')],ignore_index=True)
    intervals.to_csv(out/'cluster_bootstrap_ci.csv',index=False)
    agreement=matched.groupby(['shift','method'],as_index=False).agg(
        route_change_fraction=('route_changed_vs_dijkstra','mean'))
    agreement.to_csv(out/'route_agreement.csv',index=False)
    config=dict(phase='10.1',exploratory=True,quick=quick,cases=total,records=len(df),
        methods=list(ALL_METHODS),sizes=list(sizes),seeds=list(seeds),shifts=list(shifts),budgets=list(budgets),
        dev_scenarios=dev_scenarios,test_count=test_count,stress_count=stress_count,
        epsilon=epsilon,lambda_shift=lam,baseline='cvar_fixed',
        caveats=['Candidate-restricted comparison; no proof of global optimality',
                 'Stress environments prespecified from assumed shifts; not learned from real observations',
                 'Shift-stress selection uses no held-out scenarios',
                 'Same physical graph repeated across shifts/budgets; cluster by topology/size/seed',
                 'Bootstrap intervals are descriptive and not multiplicity-adjusted',
                 'Phase 10.1 is post-hoc exploratory, not confirmatory',
                 'correlated_gamma changes marginal variance as well as correlation',
                 'Candidate-level held-out diagnostics are for audit only, never selection'])
    (out/'experiment_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    print('Phase 10.1 complete:',total,'cases,',len(df),'method records; outputs:',out)
    return df,audit,matched,summary,intervals

def main():
    p=argparse.ArgumentParser(description='Phase 10.1 route audit and exploratory shift-stress CVaR')
    p.add_argument('--quick',action='store_true')
    p.add_argument('--output',default='data/results/phase101')
    p.add_argument('--dev-scenarios',type=int,default=800)
    p.add_argument('--test-scenarios',type=int,default=3000)
    p.add_argument('--stress-scenarios',type=int,default=500)
    p.add_argument('--epsilon',type=float,default=.10)
    p.add_argument('--lambda-shift',type=float,default=.25)
    p.add_argument('--draws',type=int,default=2000)
    a=p.parse_args()
    run_benchmark(a.output,a.quick,a.dev_scenarios,a.test_scenarios,a.stress_scenarios,a.epsilon,a.lambda_shift,a.draws)

if __name__=='__main__': main()
