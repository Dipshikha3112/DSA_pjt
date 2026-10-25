"""Phase 14: reproducible paired, city-aware audit of Phase 13 outputs.
No refitting, no fresh simulation, no changes to Phase 13.
"""
import argparse, io, itertools, json, math, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

METHODS=['iid_cvar95','mixture_cvar95','worst_model_cvar95','minimax_regret_cvar95']
KEY=['area','pair','rho','dev_n','shift','method']

def read_data(path):
    path=Path(path)
    if path.is_file() and path.suffix.lower()=='.zip':
        with zipfile.ZipFile(path) as z:
            return {n:pd.read_csv(io.BytesIO(z.read(n))) for n in ['algorithm_results.csv','paired_results.csv','route_stability.csv','candidate_audit.csv']}
    return {n:pd.read_csv(path/n) for n in ['algorithm_results.csv','paired_results.csv','route_stability.csv','candidate_audit.csv']}

def audit(t):
    a,p,s,c=(t[n] for n in ['algorithm_results.csv','paired_results.csv','route_stability.csv','candidate_audit.csv'])
    issues=[]
    for name,d,keys in [('algorithm',a,KEY),('paired',p,KEY),('stability',s,['area','pair','rho','dev_n','method']),('candidate',c,['area','pair','rho','dev_n'])]:
        if d.duplicated(keys).any():issues.append(f'{name}: duplicate primary keys')
        if d.isna().any().any():issues.append(f'{name}: missing values')
    if len(a)!=22500 or len(p)!=18000 or len(s)!=7500 or len(c)!=1500:issues.append('Unexpected table row counts')
    if a['area'].nunique()!=5 or a.groupby('area')['pair'].nunique().ne(15).any():issues.append('Expected 5 areas and 15 pairs per area')
    if set(p.method)!=set(METHODS):issues.append('Unexpected comparison methods')
    if not np.isfinite(a[['mean_seconds','cvar95_seconds','p95_seconds']].to_numpy()).all():issues.append('Nonfinite algorithm metrics')
    if (a[['mean_seconds','cvar95_seconds','p95_seconds']]<=0).any().any():issues.append('Nonpositive algorithm metrics')
    if (c.feasible_count>c.candidate_count).any():issues.append('Feasible count exceeds candidate count')
    if (c.feasible_count==0).any():issues.append('Candidate set has no feasible routes')
    if not s.agrees_with_largest_n.isin([0,1]).all():issues.append('Invalid route stability flag')
    base=a[a.method=='dijkstra_development_mean'].rename(columns={'cvar95_seconds':'baseline_cvar95','mean_seconds':'baseline_mean'})
    base=base[['area','pair','rho','dev_n','shift','baseline_cvar95','baseline_mean']]
    joined=p.merge(a,on=KEY,how='left',validate='one_to_one',indicator=True)
    if (joined['_merge']!='both').any():issues.append('Paired rows missing algorithm records')
    joined=joined.merge(base,on=['area','pair','rho','dev_n','shift'],validate='many_to_one')
    if not np.allclose(joined.cvar95_seconds-joined.baseline_cvar95,joined.delta_vs_dijkstra_cvar95,atol=1e-6,rtol=1e-7):issues.append('Paired CVaR deltas do not match algorithm records')
    if not np.allclose(joined.mean_seconds-joined.baseline_mean,joined.delta_vs_dijkstra_mean,atol=1e-6,rtol=1e-7):issues.append('Paired mean deltas do not match algorithm records')
    return {'passed':not issues,'issues':issues,'counts':{k:int(len(v)) for k,v in t.items()},'areas':sorted(a.area.unique().tolist()),'shifts':sorted(a['shift'].unique().tolist()),'rho':sorted(a.rho.unique().tolist()),'dev_n':sorted(a.dev_n.unique().tolist())}

def hierarchical_ci(d,metric,rng,B=3000):
    # Resample independent geographic units first, then OD pairs within each city.
    city_arrays=[g[metric].to_numpy(dtype=float) for _,g in d.groupby('area',sort=True)]
    k=len(city_arrays)
    # Vectorize bootstrap draws to keep 240 scenario groups fast on Windows.
    city_means=np.empty((k,B),dtype=float)
    for i,x in enumerate(city_arrays):
        draws=rng.integers(0,len(x),size=(B,len(x)))
        city_means[i]=x[draws].mean(axis=1)
    city_draws=rng.integers(0,k,size=(B,k))
    boot=np.take_along_axis(city_means.T,city_draws,axis=1).mean(axis=1)
    return np.quantile(boot,[.025,.975]).tolist()

def signflip_city_p(d,metric):
    # Exact two-sided sign-flip test across 5 CITY means (32 possible flips).
    city=d.groupby('area')[metric].mean().to_numpy(dtype=float)
    obs=abs(city.mean())
    null=[abs(np.mean(city*np.array(signs))) for signs in itertools.product([-1,1],repeat=len(city))]
    return float(np.mean(np.array(null)>=obs-1e-12))

def holm(pvals):
    idx=np.argsort(pvals);out=np.empty(len(pvals));mx=0
    for rank,i in enumerate(idx):
        mx=max(mx,min(1,(len(pvals)-rank)*pvals[i]));out[i]=mx
    return out.tolist()

def summarize(d,metric,rng,B):
    x=d[metric].to_numpy(dtype=float)
    ci=hierarchical_ci(d,metric,rng,B)
    return dict(n=int(len(d)),cities=int(d.area.nunique()),mean=float(x.mean()),median=float(np.median(x)),ci_low=float(ci[0]),ci_high=float(ci[1]),improved=int((x < -1e-9).sum()),worsened=int((x > 1e-9).sum()),tied=int((abs(x)<=1e-9).sum()),city_signflip_p=signflip_city_p(d,metric))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',default='data/results/phase13_full_results.zip');ap.add_argument('--output',default='data/results/phase14');ap.add_argument('--bootstrap',type=int,default=3000);ap.add_argument('--seed',type=int,default=20261010);args=ap.parse_args()
    t=read_data(args.input);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    qa=audit(t);(out/'data_integrity_audit.json').write_text(json.dumps(qa,indent=2),encoding='utf-8')
    print('Data integrity:', 'PASS' if qa['passed'] else 'FAIL',qa['issues'],flush=True)
    if not qa['passed']:raise SystemExit('STOP: data integrity audit failed. See JSON report.')
    p=t['paired_results.csv'];rng=np.random.default_rng(args.seed)
    rows=[]
    for (rho,n,shift,method),d in p.groupby(['rho','dev_n','shift','method'],sort=True):
        stats=summarize(d,'delta_vs_dijkstra_cvar95',rng,args.bootstrap)
        rows.append(dict(rho=rho,dev_n=n,shift=shift,method=method,**stats,route_change_fraction=float(d.changed_from_dijkstra.mean()),mean_delta_travel_time=float(d.delta_vs_dijkstra_mean.mean())))
    summary=pd.DataFrame(rows);summary.to_csv(out/'all_scenarios_hierarchical_ci.csv',index=False)
    primary=summary[(np.isclose(summary.rho,.65))&(summary.dev_n==5000)].copy()
    primary['holm_p_within_shift']=np.nan
    for shift,ix in primary.groupby('shift').groups.items():primary.loc[ix,'holm_p_within_shift']=holm(primary.loc[ix,'city_signflip_p'].to_numpy())
    primary.to_csv(out/'primary_comparisons.csv',index=False)
    city=p.groupby(['area','rho','dev_n','shift','method'],as_index=False).agg(mean_delta_cvar95=('delta_vs_dijkstra_cvar95','mean'),median_delta_cvar95=('delta_vs_dijkstra_cvar95','median'),route_change_fraction=('changed_from_dijkstra','mean'),pairs=('pair','nunique'))
    city.to_csv(out/'city_level_effects.csv',index=False)
    leave=[]
    for (rho,n,shift,method),d in p.groupby(['rho','dev_n','shift','method'],sort=True):
        for city_name in sorted(d.area.unique()):
            rem=d[d.area!=city_name]
            leave.append(dict(rho=rho,dev_n=n,shift=shift,method=method,excluded_city=city_name,remaining_mean_delta=float(rem.delta_vs_dijkstra_cvar95.mean())))
    pd.DataFrame(leave).to_csv(out/'leave_one_city_out.csv',index=False)
    stab=t['route_stability.csv'].groupby(['rho','dev_n','method'],as_index=False).agg(route_agreement=('agrees_with_largest_n','mean'),observations=('pair','size'))
    stab.to_csv(out/'route_stability_summary.csv',index=False)
    candidate=t['candidate_audit.csv'].groupby('area',as_index=False).agg(mean_candidates=('candidate_count','mean'),mean_feasible=('feasible_count','mean'),mean_selected_unique=('selected_unique_routes','mean'),min_feasible=('feasible_count','min'))
    candidate.to_csv(out/'candidate_fairness_summary.csv',index=False)
    # No claim of independent 75 geographic units; 5 cities only, with 15 OD pairs nested in each.
    report={'design':'Exploratory audit of fixed Phase 13 outputs; no new simulation','primary_slice':'rho=0.65, dev_n=5000; all 3 evaluation shifts, 4 methods','interpretation':'Negative delta means lower held-out CVaR95 than Dijkstra','uncertainty':'Two-stage city/OD bootstrap, percentile 95% CI; only five cities, so intervals may be unstable','testing':'Exact two-sided city sign-flip tests; Holm adjustment over four methods within each shift; minimum attainable nonzero two-sided p with 5 cities is 0.0625','dependence':'Other rho and sample sizes reuse the same OD pairs; do not treat rows across conditions as independent','caveats':['Simulated uncertainty, not measured traffic','Different network extraction pipelines across cities','Candidate-route restriction','No formal DRO optimality guarantee','Only 5 geographic clusters; inferential power limited'],'settings':vars(args),'audit':qa}
    (out/'methodology_and_limitations.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('\nPRIMARY COMPARISONS: rho=0.65, n=5000 (negative is better)')
    print(primary[['shift','method','mean','ci_low','ci_high','improved','worsened','tied','city_signflip_p','holm_p_within_shift']].to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    print('\nSaved Phase 14 reports to',out)
if __name__=='__main__':main()
