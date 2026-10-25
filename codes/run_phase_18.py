#!/usr/bin/env python3
"""Phases 20-26 extension for the existing five-city shortest-path experiment.

Phase 20: integrity and leakage audit of Phase 19 artifacts.
Phase 21: chance-constrained, CVaR, robust and oracle-on-candidate baselines.
Phase 22: exploratory adaptive CVaR routing with a risk-gap screening rule.
Phase 23: optional chronological observed traffic evaluation (strictly no fake data).
Phase 24: sensitivity and ablation of alpha, gamma, budgets, screening threshold.
Phase 25: city-cluster confidence intervals, exact sign flips and Holm correction.
Phase 26: wall-clock runtime, candidate counts and scalability diagnostics.

No global optimality, publication novelty, or real-traffic validation is claimed.
Candidate-set oracle is a retrospective *evaluation bound*, not a deployable method.
Requires Phase 19 CSV plus Phase 13 GraphML. All results use independent held-out
scenario streams. Use --help for options.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
import os
import platform
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import gamma as gamma_dist, norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CITIES = {c: f'phase13_{c}.graphml' for c in ('pune','hyderabad','chennai','jaipur','kolkata')}
SHIFTS = ('iid_gamma','correlated_gamma','heavy_lognormal')
METHODS = ('dijkstra','sample_mean','mean_std','cvar','chance','bs_robust','adaptive_cvar','oracle_test_only')


def seed(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str,parts)).encode()).digest()[:8],'big') % 2**32


def numeric(v):
    try:
        x=float(v)
        return x if np.isfinite(x) else None
    except (TypeError,ValueError):
        return None


def json_write(path, obj):
    Path(path).write_text(json.dumps(obj,indent=2,default=lambda x: x.item() if isinstance(x,np.generic) else str(x)),encoding='utf-8')


def cvar(x, alpha):
    x=np.asarray(x,dtype=float)
    if x.size==0 or not np.isfinite(x).all() or not 0<alpha<1: raise ValueError('invalid CVaR input')
    eta=np.quantile(x,alpha,method='inverted_cdf')
    return float(eta+np.maximum(x-eta,0).mean()/(1-alpha))


def robust_cost(mu,delta,gamma):
    if gamma<0: raise ValueError('gamma must be nonnegative')
    d=np.sort(np.asarray(delta,dtype=float))[::-1]
    k=min(int(gamma),len(d)); f=min(max(gamma-k,0),1)
    return float(np.sum(mu)+d[:k].sum()+(f*d[k] if k<len(d) else 0))


def self_test():
    assert abs(cvar([0,1,2,3],.5)-2.5)<1e-9
    assert abs(robust_cost([10,20,30],[2,5,8],1.5)-70.5)<1e-9
    assert seed('a',3)==seed('a',3)
    a=np.array([[1.,2.],[3.,4.]])
    assert np.allclose(a@np.array([[1.,0.],[0.,1.]]),a)
    print('Self-tests passed.')


def audit_phase19(root, output, expected_cities):
    root=Path(root); issues=[]; checks={}; evidence={}
    manifest=root/'experiment_manifest.json'
    summary=root/'summary.csv'
    diagnostics=root/'candidate_pool_diagnostics.csv'
    tests=root/'city_statistical_tests.csv'
    results=root/'all_method_results.csv'
    for p in (manifest,summary,diagnostics,tests,results):
        checks[p.name+'_exists']=p.exists()
        if not p.exists(): issues.append(f'MISSING: {p.name}')
    if manifest.exists():
        m=json.loads(manifest.read_text(encoding='utf-8'))
        evidence['manifest']=m
        if m.get('data_mode')!='observed': issues.append('Real traffic not established: synthetic experiment')
        if not m.get('excluded_od_file'): issues.append('Untouched Phase 13 OD pairs not established')
        if set(m.get('cities',[]))!=set(expected_cities): issues.append('Manifest city list differs from selected cities')
    if diagnostics.exists():
        d=pd.read_csv(diagnostics)
        required={'city','od','generator','budget','actual_candidates','generation_seconds'}
        if not required.issubset(d.columns): issues.append('Candidate diagnostic columns missing')
        else:
            short=d[d.actual_candidates<d.budget]
            evidence['candidate_pools']=len(d)
            evidence['undersized_candidate_pools']=len(short)
            evidence['single_route_pools']=int((d.actual_candidates==1).sum())
            evidence['invalid_candidate_counts']=int(((d.actual_candidates<1)|(d.actual_candidates>d.budget)).sum())
            short.to_csv(output/'phase20_undersized_pools.csv',index=False)
            if len(short): issues.append(f'{len(short)} candidate pools smaller than requested')
    if results.exists():
        r=pd.read_csv(results)
        required={'city','od','shift','generator','budget','method','chosen_route','test_mean','test_cvar95','dev_sha256','test_sha256'}
        if not required.issubset(r.columns): issues.append('Phase 19 results lack required audit fields')
        else:
            keys=['city','od','shift','generator','budget','method']
            evidence['phase19_records']=len(r)
            evidence['duplicate_experiment_keys']=int(r.duplicated(keys).sum())
            evidence['dev_test_hash_equal']=int((r.dev_sha256==r.test_sha256).sum())
            evidence['nonfinite_metrics']=int((~np.isfinite(r[['test_mean','test_cvar95']].to_numpy(dtype=float))).sum())
            evidence['cvar_below_mean']=int((r.test_cvar95+1e-7<r.test_mean).sum())
            if evidence['duplicate_experiment_keys']: issues.append('Duplicate experimental keys')
            if evidence['dev_test_hash_equal']: issues.append('Development and test hashes identical')
            if evidence['cvar_below_mean']: issues.append('CVaR below mean for nonnegative losses')
            group=['city','od','shift']
            hashes=r.groupby(group).agg(n_dev=('dev_sha256','nunique'),n_test=('test_sha256','nunique'))
            evidence['scenario_hash_inconsistencies']=int(((hashes.n_dev!=1)|(hashes.n_test!=1)).sum())
            if evidence['scenario_hash_inconsistencies']: issues.append('Methods or pools do not share matched scenarios')
            ref=r[r.method=='dijkstra'].set_index(['city','od','shift','generator','budget'])
            cmp=r[r.method.isin(['ru_cvar95','bs_gamma_robust'])].copy()
            if not cmp.empty:
                key=['city','od','shift','generator','budget']
                j=cmp.merge(ref.reset_index()[key+['chosen_route']],on=key,suffixes=('','_dijkstra'))
                evidence['risk_route_change_rate']=float((j.chosen_route!=j.chosen_route_dijkstra).mean())
    evidence['issues']=issues
    evidence['status']='FAIL' if any(x.startswith(('MISSING','Duplicate','Development and test','CVaR below','Methods or pools')) for x in issues) else 'WARN' if issues else 'PASS'
    json_write(output/'phase20_integrity_report.json',evidence)
    return evidence


def load_graph(path,speed):
    raw=nx.read_graphml(path); g=nx.DiGraph(); stats={'raw_nodes':raw.number_of_nodes(),'raw_edges':raw.number_of_edges(),'fallback_speed_edges':0,'skipped_edges':0}
    for u,a in raw.nodes(data=True):
        x=numeric(a.get('x')); y=numeric(a.get('y'))
        if x is None: x=numeric(a.get('lon'))
        if y is None: y=numeric(a.get('lat'))
        g.add_node(str(u),x=x,y=y)
    for u,v,a in raw.edges(data=True):
        t=numeric(a.get('travel_time'))
        if t is None or t<=0:
            length=numeric(a.get('length'))
            if length is None or length<=0:
                stats['skipped_edges']+=1;continue
            t=length/(speed/3.6);stats['fallback_speed_edges']+=1
        u,v=str(u),str(v)
        if not g.has_edge(u,v) or t<g[u][v]['mean']:g.add_edge(u,v,mean=float(t))
    if not g.number_of_edges():raise ValueError(f'No usable edges: {path}')
    g=g.subgraph(max(nx.strongly_connected_components(g),key=len)).copy()
    stats.update(nodes=g.number_of_nodes(),edges=g.number_of_edges())
    for u,v,a in g.edges(data=True):
        rng=np.random.default_rng(seed('edge',u,v))
        a['cv']=float(rng.uniform(.12,.42))
        x=g.nodes[u].get('x')
        a['zone']=int(x*10) if x is not None else 0
    return g,stats


def scenarios(mu,cv,zones,shift,n,rng):
    if shift=='iid_gamma':
        shape=1/cv**2
        return rng.gamma(shape=shape,scale=mu/shape,size=(n,len(mu)))
    if shift=='correlated_gamma':
        _,idx=np.unique(zones,return_inverse=True)
        z=.75*rng.standard_normal((n,len(np.unique(idx))))[:,idx]+np.sqrt(1-.75**2)*rng.standard_normal((n,len(mu)))
        p=np.clip(norm.cdf(z),1e-12,1-1e-12)
        shape=1/cv**2
        return gamma_dist.ppf(p,a=shape,scale=mu/shape)
    if shift=='heavy_lognormal':
        sigma=np.sqrt(np.log1p((cv*2.4)**2))
        return mu*np.exp(rng.standard_normal((n,len(mu)))*sigma-.5*sigma**2)
    raise ValueError(shift)


def route_incidence(g,paths):
    edges=sorted(set(e for p in paths for e in zip(p[:-1],p[1:])))
    lookup={e:i for i,e in enumerate(edges)}
    mat=np.zeros((len(edges),len(paths)),dtype=np.float64)
    for j,p in enumerate(paths):
        for e in zip(p[:-1],p[1:]):mat[lookup[e],j]=1
    mu=np.array([g[u][v]['mean'] for u,v in edges]);cv=np.array([g[u][v]['cv'] for u,v in edges]);zones=np.array([g[u][v]['zone'] for u,v in edges])
    return edges,mat,mu,cv,zones


def candidates(g,s,t,k):
    try:return [tuple(p) for p in itertools.islice(nx.shortest_simple_paths(g,s,t,weight='mean'),k)]
    except (nx.NetworkXNoPath,nx.NodeNotFound):return []


def load_od(root,city,g,n,seed_value,excluded):
    p=Path(root)/'phase19_od_pairs.csv'; pairs=[]
    if p.exists():
        f=pd.read_csv(p,dtype={'source':str,'target':str});f=f[f.city==city].sort_values('od')
        for row in f.itertuples():
            s,t=str(row.source),str(row.target)
            if s in g and t in g and s!=t and (s,t) not in excluded:pairs.append((s,t))
        if len(pairs)>=n:return pairs[:n], 'phase19_reused'
    valid=[u for u,a in g.nodes(data=True) if a.get('x') is not None]
    if len(valid)<20:valid=list(g)
    if len(valid)<2:raise ValueError('Too few nodes for OD sampling')
    rng=np.random.default_rng(seed_value); seen=set(excluded)|set(pairs)
    for _ in range(max(300,n*100)):
        if len(pairs)>=n:break
        i,j=rng.choice(len(valid),2,replace=False);s,t=valid[i],valid[j]
        if (s,t) in seen:continue
        seen.add((s,t));pairs.append((s,t))
    if len(pairs)<n:raise ValueError(f'{city}: could not generate enough OD pairs')
    return pairs[:n], 'mixed_or_new'


def read_exclusions(path,city):
    if not path:return set()
    f=pd.read_csv(path,dtype=str)
    if not {'city','source','target'}.issubset(f):raise ValueError('Exclusion CSV needs city,source,target')
    return set(zip(f.loc[f.city==city,'source'],f.loc[f.city==city,'target']))


def observed_matrix(path,edges,min_rows):
    f=pd.read_csv(path,dtype={'u':str,'v':str})
    needed={'timestamp','u','v','travel_seconds'}
    if not needed.issubset(f):raise ValueError(f'{path}: expected {sorted(needed)}')
    f['travel_seconds']=pd.to_numeric(f.travel_seconds,errors='coerce')
    f=f[np.isfinite(f.travel_seconds)&(f.travel_seconds>0)].copy()
    f['edge']=list(zip(f.u,f.v))
    if f.duplicated(['timestamp','edge']).any():raise ValueError('duplicate timestamp-edge observations')
    f['timestamp']=pd.to_datetime(f['timestamp'],errors='coerce',utc=True)
    if f['timestamp'].isna().any():raise ValueError('Unparseable timestamps in observed traffic')
    f=f[f.edge.isin(set(edges))]
    wide=f.pivot(index='timestamp',columns='edge',values='travel_seconds').reindex(columns=edges).dropna().sort_index()
    if len(wide)<min_rows:raise ValueError(f'Insufficient complete observed timestamps: {len(wide)} < {min_rows}')
    x=wide.to_numpy(float);cut=int(len(x)*.6);validation=cut+int(len(x)*.2)
    if cut<20 or validation-cut<10 or len(x)-validation<10:raise ValueError('Insufficient chronological splits')
    return x[:cut],x[cut:validation],x[validation:]


def select_routes(dev,val,test,mat,mu,cv,alpha,gamma,threshold,ablate):
    # All objectives use development only; validation calibrates screening,
    # and the test set is never accessed to choose deployable methods.
    means=dev.mean(axis=0);sds=dev.std(axis=0)
    scores={
        'dijkstra':mu@mat,
        'sample_mean':means,
        'mean_std':means+.4*sds,
        'cvar':np.array([cvar(dev[:,j],alpha) for j in range(dev.shape[1])]),
        'chance':np.quantile(dev,alpha,axis=0,method='higher'),
        'bs_robust':np.array([robust_cost(mu[mat[:,j]>0],mu[mat[:,j]>0]*cv[mat[:,j]>0],gamma) for j in range(mat.shape[1])]),
    }
    selected={m:int(np.argmin(v)) for m,v in scores.items()}
    # Adaptive screening: choose CVaR only if the development tail-risk
    # reduction exceeds a threshold, otherwise choose nominal Dijkstra.
    # Threshold is a pre-specified hyperparameter, not tuned on test data.
    nominal=selected['dijkstra']; risk=selected['cvar']
    gain=float(scores['cvar'][nominal]-scores['cvar'][risk])
    selected['adaptive_cvar']=nominal if ablate or gain<=threshold else risk
    # Oracle is explicitly labelled test-only and cannot be deployed.
    selected['oracle_test_only']=int(np.argmin([cvar(test[:,j],alpha) for j in range(test.shape[1])]))
    return selected,scores,gain


def run_experiments(args,out):
    rows=[]; diagnostics=[]; failures=[]; graph_stats=[]
    graph_dir=Path(args.graphs_dir)
    if args.observed_dir:
        missing=[str(Path(args.observed_dir)/f'{c}.csv') for c in args.cities if not (Path(args.observed_dir)/f'{c}.csv').exists()]
        if missing:raise FileNotFoundError('Observed mode requires each city CSV; missing: '+', '.join(missing))
    for city in args.cities:
        path=graph_dir/CITIES[city]
        if not path.exists():raise FileNotFoundError(f'Graph not found: {path}')
        g,stats=load_graph(path,args.assumed_speed_kmh);stats['city']=city;graph_stats.append(stats)
        excluded=read_exclusions(args.exclude_od,city)
        pairs,origin=load_od(args.phase19_dir,city,g,args.od_pairs,seed(args.seed,city,'od'),excluded)
        print(f'[{city}] {stats["nodes"]:,} nodes, {stats["edges"]:,} edges; OD origin: {origin}',flush=True)
        for od,(s,t) in enumerate(pairs):
            # Same maximum candidate pool for all budgets and alpha/gamma sweeps.
            started=time.perf_counter()
            try:pool=candidates(g,s,t,max(args.budgets))
            except Exception as e:
                failures.append(dict(city=city,od=od,error=f'candidate error: {e}'));continue
            gen_seconds=time.perf_counter()-started
            if not pool:
                failures.append(dict(city=city,od=od,error='no candidate path'));continue
            if any(p[0]!=s or p[-1]!=t or any(not g.has_edge(u,v) for u,v in zip(p[:-1],p[1:])) for p in pool):
                failures.append(dict(city=city,od=od,error='invalid route reconstruction'));continue
            for budget in args.budgets:
                paths=pool[:budget];edges,mat,mu,cv,zones=route_incidence(g,paths)
                if not edges:continue
                if args.observed_dir:
                    try:
                        dev_edges,val_edges,test_edges=observed_matrix(Path(args.observed_dir)/f'{city}.csv',edges,args.observed_min)
                    except Exception as e:
                        failures.append(dict(city=city,od=od,budget=budget,error=f'observed: {e}'));continue
                    datasets=[('observed_chronological',dev_edges,val_edges,test_edges)]
                else:
                    datasets=[]
                    for shift in args.shifts:
                        # Scenarios are matched for all methods at a given OD/budget/shift.
                        dev_edges=scenarios(mu,cv,zones,shift,args.dev_scenarios,np.random.default_rng(seed(args.seed,city,od,budget,shift,'dev')))
                        val_edges=scenarios(mu,cv,zones,shift,args.val_scenarios,np.random.default_rng(seed(args.seed,city,od,budget,shift,'val')))
                        test_edges=scenarios(mu,cv,zones,shift,args.test_scenarios,np.random.default_rng(seed(args.seed,city,od,budget,shift,'test')))
                        datasets.append((shift,dev_edges,val_edges,test_edges))
                for shift,dev_e,val_e,test_e in datasets:
                    dev=dev_e@mat;val=val_e@mat;test=test_e@mat
                    for alpha in args.alphas:
                        for gamma in args.gammas:
                            for threshold in args.thresholds:
                                for ablate in (False,True):
                                    tick=time.perf_counter()
                                    selected,scores,gain=select_routes(dev,val,test,mat,mu,cv,alpha,gamma,threshold,ablate)
                                    selection_seconds=time.perf_counter()-tick
                                    common=dict(city=city,od=od,source=s,target=t,budget=budget,actual_candidates=len(paths),shift=shift,alpha=alpha,gamma=gamma,threshold=threshold,ablate_screening=ablate,data_mode='observed' if args.observed_dir else 'synthetic',od_origin=origin)
                                    for method,j in selected.items():
                                        outcomes=test[:,j]
                                        rows.append(dict(**common,method=method,chosen_candidate=j,chosen_route='->'.join(paths[j]),test_mean=float(outcomes.mean()),test_cvar=float(cvar(outcomes,alpha)),test_p95=float(np.quantile(outcomes,.95)),test_std=float(outcomes.std()),dev_risk_gain=gain,gen_seconds=gen_seconds,selection_seconds=selection_seconds,wall_seconds=gen_seconds+selection_seconds,oracle_uses_test=(method=='oracle_test_only')))
                diagnostics.append(dict(city=city,od=od,budget=budget,candidates=len(paths),requested=budget,gen_seconds=gen_seconds,edges_in_pool=len(edges),graph_nodes=len(g),graph_edges=g.number_of_edges()))
            print(f'  OD {od+1}/{len(pairs)} done',flush=True)
    pd.DataFrame(graph_stats).to_csv(out/'phase20_graph_audit.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(out/'phase26_candidate_scalability.csv',index=False)
    pd.DataFrame(failures,columns=['city','od','budget','error']).to_csv(out/'experiment_failures.csv',index=False)
    if not rows:raise RuntimeError('No experiment results produced; inspect experiment_failures.csv')
    result=pd.DataFrame(rows);result.to_csv(out/'phase21_24_all_results.csv',index=False)
    return result


def holm(p):
    x=np.asarray(p,float);res=np.full(len(x),np.nan);valid=np.flatnonzero(np.isfinite(x));order=valid[np.argsort(x[valid])];running=0
    for i,j in enumerate(order):
        running=max(running,(len(order)-i)*x[j]);res[j]=min(running,1)
    return res


def signflip(values):
    x=np.asarray(values,float)
    if len(x)<2:return np.nan
    if len(x)>16:return np.nan
    obs=abs(x.mean());count=0;total=0
    for bits in itertools.product((-1,1),repeat=len(x)):
        count+=abs(np.mean(x*np.array(bits)))>=obs-1e-12;total+=1
    return count/total


def stats_phase25(r,out,bootstrap,seed_value):
    key=['city','od','budget','shift','alpha','gamma','threshold','ablate_screening']
    baseline=r[r.method=='dijkstra'][key+['test_cvar','test_mean','chosen_route']].rename(columns={'test_cvar':'base_cvar','test_mean':'base_mean','chosen_route':'base_route'})
    paired=r.merge(baseline,on=key,validate='many_to_one')
    paired['delta_cvar']=paired.test_cvar-paired.base_cvar
    paired['delta_mean']=paired.test_mean-paired.base_mean
    paired['route_changed']=paired.chosen_route!=paired.base_route
    paired.to_csv(out/'phase25_paired_effects.csv',index=False)
    group=['shift','method','alpha','gamma','threshold','ablate_screening','budget']
    report=[]
    for keys,sub in paired.groupby(group,dropna=False):
        city=sub.groupby('city',as_index=False).delta_cvar.mean();values=city.delta_cvar.to_numpy();n=len(values)
        rng=np.random.default_rng(seed(seed_value,'bootstrap',*keys));boot=np.mean(rng.choice(values,(bootstrap,n),replace=True),axis=1) if n>=2 else np.array([np.nan])
        report.append(dict(zip(group,keys))|dict(n_cities=n,n_rows=len(sub),mean_delta_cvar=float(values.mean()),median_delta_cvar=float(np.median(values)),ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975)),signflip_p=signflip(values),route_change_rate=float(sub.route_changed.mean()),mean_runtime_s=float(sub.wall_seconds.mean()),test_only_oracle=bool((sub.method=='oracle_test_only').any())))
    report=pd.DataFrame(report)
    # Holm within pre-specified primary family only; exploratory sweeps are descriptive.
    primary=(report.alpha==.95)&(report.gamma==3)&(report.threshold==0)&(~report.ablate_screening)&(report.budget==max(r.budget))&(~report.method.isin(['dijkstra','oracle_test_only']))
    report['holm_primary_p']=np.nan
    if primary.any():report.loc[primary,'holm_primary_p']=holm(report.loc[primary,'signflip_p'].to_numpy())
    report['primary_comparison']=primary
    report.to_csv(out/'phase25_cluster_inference.csv',index=False)
    return paired,report


def plots(r,paired,stats,out):
    fig,ax=plt.subplots(figsize=(10,5))
    x=stats[(stats.primary_comparison)&(stats.method.isin(['cvar','chance','bs_robust','adaptive_cvar']))]
    if not x.empty:
        x=x.groupby('method',as_index=False).mean_delta_cvar.mean()
        ax.bar(x.method,x.mean_delta_cvar);ax.axhline(0,color='black',lw=.8)
    ax.set(title='Primary comparisons: city-averaged CVaR difference',ylabel='Seconds vs Dijkstra (negative is better)');fig.tight_layout();fig.savefig(out/'phase25_primary_effects.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5))
    s=r[(r.method=='adaptive_cvar')&(~r.ablate_screening)].groupby('threshold',as_index=False).test_cvar.mean()
    if not s.empty:ax.plot(s.threshold,s.test_cvar,marker='o')
    ax.set(title='Adaptive screening sensitivity (descriptive)',xlabel='Screening threshold (seconds)',ylabel='Average test CVaR');fig.tight_layout();fig.savefig(out/'phase24_screening_sensitivity.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5))
    s=r.groupby('budget',as_index=False).wall_seconds.mean()
    if not s.empty:ax.plot(s.budget,s.wall_seconds,marker='o')
    ax.set(title='Runtime vs candidate budget',xlabel='Candidate budget',ylabel='Mean wall time (seconds)');fig.tight_layout();fig.savefig(out/'phase26_runtime_scaling.png',dpi=180);plt.close(fig)


def write_report(audit,r,stats,args,out):
    observed=bool(args.observed_dir)
    primary=stats[stats.primary_comparison].sort_values('mean_delta_cvar')
    lines=['# Phases 20–26 experimental report','','## Provenance',f'- Cities: {", ".join(args.cities)}',f'- Data mode: {"observed chronological" if observed else "synthetic on real road networks"}',f'- Records: {len(r):,}',f'- Audit status: {audit.get("status")}',f'- Phase 19 OD exclusion provided: {bool(args.exclude_od)}','','## Phase 20 integrity warnings']
    lines.extend('- '+str(i) for i in audit.get('issues',[]))
    lines+=['','## Phase 21–22 methods','- Dijkstra: nominal shortest-time candidate.','- Chance baseline: minimum development empirical alpha-quantile among candidates (not a full constrained solver).','- CVaR: minimum development empirical alpha-CVaR among candidates.','- Bertsimas–Sim: budgeted uncertainty cost over candidates.','- Adaptive CVaR: select CVaR candidate only if development CVaR improvement exceeds predeclared threshold.','- Oracle: minimizes *test* CVaR, an infeasible retrospective reference; NEVER compare as a deployable algorithm.','','## Primary comparison results (exploratory)']
    if len(primary):lines.append(primary[['shift','method','budget','mean_delta_cvar','ci_low','ci_high','signflip_p','holm_primary_p']].to_string(index=False))
    lines+=['','## Phase 23','- Observed validation executed.' if observed else '- NOT EXECUTED: supply timestamped edge traffic via --observed-dir.','', '## Limitations','- New adaptive screening rule is an exploratory proposal; novelty has not been established by literature review.','- All optimization is restricted to the same finite nominal k-shortest candidate pool.','- Candidate-set oracle uses test data; it is a non-deployable hindsight lower bound.','- City-level bootstrap intervals with only five clusters can be unstable.','- Sensitivity and ablation results are exploratory, not confirmatory significance tests.','- Synthetic edge uncertainty cannot establish real traffic performance.','- Real observed traffic requires complete aligned timestamp-by-edge coverage.','- Reused Phase 19 OD pairs are not an independent unseen OD validation.','- Timing is wall-clock and depends on hardware, system load and implementation.\n- Generation time is for the maximum candidate pool and is reused for smaller budget reports; it is not an isolated per-budget generation benchmark.\n- Validation scenarios are held out but no data-driven hyperparameter tuning is implemented; threshold is specified by the user.']
    (out/'PHASE20_26_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def parse_float_list(s):return sorted(set(float(x.strip()) for x in s.split(',') if x.strip()))
def parse_int_list(s):return sorted(set(int(x.strip()) for x in s.split(',') if x.strip()))


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--graphs-dir',default='data/real_networks');p.add_argument('--phase19-dir',default='data/results/phase17_19_full');p.add_argument('--output',default='data/results/phase20_26')
    p.add_argument('--cities',default=','.join(CITIES));p.add_argument('--od-pairs',type=int,default=3)
    p.add_argument('--budgets',default='3,6,10');p.add_argument('--alphas',default='.90,.95,.99');p.add_argument('--gammas',default='1,3,5');p.add_argument('--thresholds',default='0,5')
    p.add_argument('--shifts',default=','.join(SHIFTS));p.add_argument('--dev-scenarios',type=int,default=500);p.add_argument('--val-scenarios',type=int,default=250);p.add_argument('--test-scenarios',type=int,default=1000)
    p.add_argument('--observed-dir',default=None);p.add_argument('--observed-min',type=int,default=80);p.add_argument('--exclude-od',default=None)
    p.add_argument('--assumed-speed-kmh',type=float,default=30);p.add_argument('--bootstrap',type=int,default=1000);p.add_argument('--seed',type=int,default=20261014)
    p.add_argument('--self-test',action='store_true');p.add_argument('--audit-only',action='store_true')
    args=p.parse_args();self_test()
    if args.self_test:return
    args.cities=[x.strip() for x in args.cities.split(',') if x.strip()];args.shifts=[x.strip() for x in args.shifts.split(',') if x.strip()]
    args.budgets=parse_int_list(args.budgets);args.alphas=parse_float_list(args.alphas);args.gammas=parse_float_list(args.gammas);args.thresholds=parse_float_list(args.thresholds)
    if not args.cities or any(c not in CITIES for c in args.cities):p.error('Unknown city')
    if not args.shifts or any(s not in SHIFTS for s in args.shifts):p.error('Unknown shift')
    if not args.budgets or min(args.budgets)<1 or args.od_pairs<1 or any(not 0<a<1 for a in args.alphas) or min(args.gammas)<0 or min(args.thresholds)<0:p.error('Invalid sweep values')
    if min(args.dev_scenarios,args.val_scenarios,args.test_scenarios,args.bootstrap)<1 or args.assumed_speed_kmh<=0:p.error('Invalid scenario or speed settings')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    audit=audit_phase19(args.phase19_dir,out,args.cities)
    print('Phase 20 audit:',audit['status'],'|',len(audit['issues']),'warnings/issues')
    if args.audit_only:return
    t=time.perf_counter();r=run_experiments(args,out)
    paired,stats=stats_phase25(r,out,args.bootstrap,args.seed)
    plots(r,paired,stats,out);write_report(audit,r,stats,args,out)
    config={k:v for k,v in vars(args).items()};config.update(elapsed_seconds=time.perf_counter()-t,records=len(r),python=platform.python_version(),networkx=nx.__version__,numpy=np.__version__,pandas=pd.__version__,publication_ready=False)
    json_write(out/'phase20_26_manifest.json',config)
    print(f'Completed {len(r):,} method evaluations in {config["elapsed_seconds"]:.1f}s. Output: {out.resolve()}')
    print('Research status: exploratory; no publication or novelty claim.')


if __name__=='__main__':main()
