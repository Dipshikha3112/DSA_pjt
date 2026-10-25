"""Phase 15.1: disagreement-focused maps, faithful to saved experimental metrics.
No model scores are synthesized except in the clearly labeled controlled demonstration.
"""
import argparse, io, json, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from phase15_support.benchmark_phase7 import graph_for

BASE='dijkstra_development_mean'
NAMES={'expected_dijkstra':'Dijkstra','monte_carlo_saa':'Monte Carlo SAA','robust_minimax':'Robust minimax','mean_variance':'Mean-variance',BASE:'Dijkstra','iid_cvar95':'Standard CVaR95','mixture_cvar95':'Mixture CVaR95','worst_model_cvar95':'Worst-model CVaR95','minimax_regret_cvar95':'Minimax regret CVaR95','candidate_cvar95':'Candidate CVaR95'}

def read(z, name):
    with zipfile.ZipFile(z) as f:
        matches=[n for n in f.namelist() if n.replace('\\','/').endswith(name)]
        if len(matches)!=1:raise ValueError(f'{z}: expected one {name}, found {matches}')
        return pd.read_csv(io.BytesIO(f.read(matches[0])))

def save_csv(df, dest):
    dest.parent.mkdir(parents=True,exist_ok=True);df.to_csv(dest,index=False)

def route_map(graph, pos, baseline, alternative, title, path, detail):
    # Only draw verified node/edge paths. No route fabrication.
    a=list(baseline);b=list(alternative)
    if not a or not b or a[0]!=b[0] or a[-1]!=b[-1]:raise ValueError('Different endpoints or empty paths')
    for p in (a,b):
        if not all(u in graph and v in graph and graph.has_edge(u,v) for u,v in zip(p[:-1],p[1:])):
            raise ValueError('Saved route not found in reconstructed graph')
    ae=set(zip(a[:-1],a[1:]));be=set(zip(b[:-1],b[1:]));common=ae&be
    fig,ax=plt.subplots(figsize=(10,8))
    nx.draw_networkx_edges(graph,pos,ax=ax,edge_color='#bcbcbc',width=.5,alpha=.15,arrows=False)
    for edges,color,width,label in [(common,'#525b66',4,'Shared'),(ae-be,'#1466b8',4,'Dijkstra only'),(be-ae,'#df681d',4,'Alternative only')]:
        if edges:nx.draw_networkx_edges(graph,pos,edgelist=list(edges),ax=ax,edge_color=color,width=width,arrows=False)
        ax.plot([],[],color=color,lw=4,label=label)
    ax.scatter(*pos[a[0]],s=90,c='#009c69',zorder=9,label='Origin')
    ax.scatter(*pos[a[-1]],s=90,c='#b32430',zorder=9,label='Destination')
    ax.set_title(title+'\n'+detail,fontsize=11);ax.legend(loc='best',fontsize=8);ax.axis('off')
    fig.tight_layout();path.parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,dpi=180,bbox_inches='tight');plt.close(fig)

def summary_chart(df,out,title):
    if df.empty:return
    fig,ax=plt.subplots(figsize=(9,5))
    ss=df.sort_values('delta_cvar95_seconds');ax.barh(ss.method.map(lambda m:NAMES.get(m,m)),ss.delta_cvar95_seconds,color=['#188a61' if x<0 else '#d06d43' for x in ss.delta_cvar95_seconds]);ax.axvline(0,color='black',lw=.8)
    ax.set_xlabel('Held-out CVaR95 minus Dijkstra (seconds; negative is better)');ax.set_title(title);fig.tight_layout();out.parent.mkdir(parents=True,exist_ok=True);fig.savefig(out,dpi=180);plt.close(fig)

def graph_cache(path):
    import osmnx as ox
    from src.phase11_real_networks import simplify_parallel
    g=ox.io.load_graphml(path);h=simplify_parallel(g)
    return h,{n:(float(d['x']),float(d['y'])) for n,d in g.nodes(data=True) if 'x' in d and 'y' in d}

def synthetic(args,out,report):
    df=read(args.phase7,'benchmark_results.csv');df['method_label']=df.algorithm+df.risk_lambda.apply(lambda v: f' lambda={v:g}' if pd.notna(v) else '')
    for topology in ['erdos_renyi','grid','scale_free']:
        sub=df[df.topology==topology].copy();groups=[]
        for keys,g in sub.groupby(['requested_n','seed','distribution','candidate_budget'],dropna=False):
            base=g[g.algorithm=='expected_dijkstra']
            if len(base)!=1:continue
            ref=base.iloc[0]
            for _,r in g[g.algorithm!='expected_dijkstra'].iterrows():
                changed=str(r.path)!=str(ref.path)
                groups.append(dict(topology=topology,n=int(keys[0]),seed=int(keys[1]),distribution=keys[2],budget=int(keys[3]),method=r.algorithm,method_label=r.method_label,changed=changed,delta_cvar95=float(r.cvar95-ref.cvar95),baseline_path=str(ref.path),alternative_path=str(r.path),baseline_cvar95=float(ref.cvar95),alternative_cvar95=float(r.cvar95)))
        if not groups:continue
        audit=pd.DataFrame(groups);save_csv(audit,out/'tables'/f'phase7_{topology}_disagreements.csv')
        different=audit[audit.changed].copy();report['synthetic'][topology]={'compared':len(audit),'disagreements':len(different)}
        if different.empty:
            print(f'Phase7 {topology}: no differing recorded paths; no fake disagreement map created',flush=True);continue
        # Select largest absolute observed CVaR difference among genuine disagreements.
        different['magnitude']=different.delta_cvar95.abs()
        for i,(_,r) in enumerate(different.sort_values('magnitude',ascending=False).head(args.max_maps).iterrows()):
            g=graph_for(topology,int(r.n),int(r.seed));pos=({n:(n%int(np.ceil(np.sqrt(r.n))),-n//int(np.ceil(np.sqrt(r.n)))) for n in g.nodes} if topology=='grid' else nx.spring_layout(g,seed=15))
            a=[int(v) for v in r.baseline_path.split('->')];b=[int(v) for v in r.alternative_path.split('->')]
            try:route_map(g,pos,a,b,f'Phase 7 | {topology} | {r.method_label}',out/'maps'/f'synthetic_{topology}_{i+1}.png',f'{r.distribution}; N={r.n}; ΔCVaR95={r.delta_cvar95:+.3f} cost units')
            except ValueError as e:print('SKIP synthetic map:',e)

def phase11(args,out,report):
    df=read(args.phase11,'algorithm_results.csv');audit=read(args.phase11,'candidate_audit.csv')
    for area in ['urban','suburban','constrained']:
        b=df[(df.area==area)&(df.method=='dijkstra_development_mean')][['pair','shift','route_id','cvar95_seconds']].rename(columns={'route_id':'baseline_route','cvar95_seconds':'baseline_cvar95'})
        alt=df[(df.area==area)&(df.method=='candidate_cvar95')].merge(b,on=['pair','shift'],validate='one_to_one')
        alt=alt.copy();alt['changed']=alt.route_id!=alt.baseline_route;alt['delta_cvar95_seconds']=alt.cvar95_seconds-alt.baseline_cvar95
        save_csv(alt,out/'tables'/f'phase11_{area}_disagreements.csv');diff=alt[alt.changed].copy()
        report['phase11'][area]={'comparisons':len(alt),'disagreements':len(diff)}
        if diff.empty:continue
        f=Path(args.cache)/f'{area}.graphml'
        if not f.exists():print('SKIP road maps; missing',f);continue
        from src.phase11_real_networks import candidates
        h,pos=graph_cache(f);diff['magnitude']=diff.delta_cvar95_seconds.abs()
        for i,(_,r) in enumerate(diff.sort_values('magnitude',ascending=False).head(args.max_maps).iterrows()):
            a=audit[(audit.area==area)&(audit.pair==r.pair)]
            if len(a)!=1:continue
            a=a.iloc[0];lookup={str(n):n for n in h.nodes};s=lookup.get(str(a.source));t=lookup.get(str(a.target))
            if s is None or t is None:continue
            pool=candidates(h,s,t,int(a.candidate_count));bi=int(r.baseline_route);ai=int(r.route_id)
            if max(bi,ai)>=len(pool):print('SKIP: candidate pool mismatch',area,r.pair);continue
            try:route_map(h,pos,pool[bi],pool[ai],f'Phase 11 | {area} | pair {r.pair}',out/'maps'/f'road_{area}_{i+1}.png',f'{r["shift"]}; ΔCVaR95={r.delta_cvar95_seconds:+.3f} s')
            except ValueError as e:print('SKIP road map:',e)

def phase13(args,out,report):
    df=read(args.phase13,'algorithm_results.csv');df=df[np.isclose(df.rho,args.rho)&(df.dev_n==args.dev)&(df['shift']==args.shift)].copy()
    if df.empty:raise ValueError('No Phase 13 records match rho/dev/shift')
    baseline=df[df.method==BASE][['area','pair','route_id','cvar95_seconds','mean_seconds']].rename(columns={'route_id':'baseline_route','cvar95_seconds':'baseline_cvar95','mean_seconds':'baseline_mean'})
    alt=df[df.method!=BASE].merge(baseline,on=['area','pair'],validate='many_to_one');alt['changed']=alt.route_id!=alt.baseline_route;alt['delta_cvar95_seconds']=alt.cvar95_seconds-alt.baseline_cvar95;alt['delta_mean_seconds']=alt.mean_seconds-alt.baseline_mean
    save_csv(alt,out/'tables'/'phase13_all_disagreements.csv')
    stats=alt.groupby(['area','method'],as_index=False).agg(pairs=('pair','size'),changed=('changed','sum'),mean_delta_cvar95_seconds=('delta_cvar95_seconds','mean'),mean_delta_mean_seconds=('delta_mean_seconds','mean'))
    save_csv(stats,out/'tables'/'phase13_disagreement_summary.csv')
    from src.phase11_real_networks import od_pairs,candidates
    areas=['pune','hyderabad','chennai','jaipur','kolkata']
    for area_index,area in enumerate(areas):
        different=alt[(alt.area==area)&alt.changed].copy()
        report['phase13'][area]={'comparisons':int((alt.area==area).sum()),'disagreements':len(different)}
        summary_chart(stats[stats.area==area].rename(columns={'mean_delta_cvar95_seconds':'delta_cvar95_seconds'}),out/'figures'/f'{area}_method_deltas.png',f'{area.title()} | {args.shift} | rho={args.rho} | N={args.dev}')
        if different.empty:print(f'Phase13 {area}: no disagreement in selected scenario');continue
        f=Path(args.cache)/f'phase13_{area}.graphml'
        if not f.exists():print('SKIP maps; missing',f);continue
        h,pos=graph_cache(f);pairs=od_pairs(h,15,args.seed+area_index*701)
        different['magnitude']=different.delta_cvar95_seconds.abs()
        for i,(_,r) in enumerate(different.sort_values('magnitude',ascending=False).head(args.max_maps).iterrows()):
            if int(r.pair)>=len(pairs):continue
            s,t=pairs[int(r.pair)];pool=candidates(h,s,t,args.budget)
            bi=int(r.baseline_route);ai=int(r.route_id)
            if max(bi,ai)>=len(pool):print('SKIP: candidate pool mismatch',area,r.pair);continue
            try:route_map(h,pos,pool[bi],pool[ai],f'Phase 13 | {area.title()} | OD {r.pair} | {NAMES.get(r.method,r.method)}',out/'maps'/f'phase13_{area}_{i+1}.png',f'{args.shift}; ΔCVaR95={r.delta_cvar95_seconds:+.3f} s; Δmean={r.delta_mean_seconds:+.3f} s')
            except ValueError as e:print('SKIP phase13 map:',e)

def demonstration(out,report):
    # Controlled and fully disclosed example: same OD, two routes.
    # A: 10s with prob .94; 100s with prob .06 => E=15.4, CVaR95=100.
    # B: always 18s => E=18, CVaR95=18.
    g=nx.DiGraph();g.add_edges_from([('S','A'),('A','T'),('S','B'),('B','T')]);pos={'S':(0,0),'A':(1,1),'B':(1,-1),'T':(2,0)}
    route_map(g,pos,['S','A','T'],['S','B','T'],'CONTROLLED SYNTHETIC DEMONSTRATION | not an empirical result',out/'maps'/'controlled_risk_tradeoff.png','Dijkstra: mean 15.4s, CVaR95 100s | CVaR route: mean 18s, CVaR95 18s')
    tab=pd.DataFrame([{'method':'Dijkstra (expected cost)','route':'S-A-T','mean_seconds':15.4,'cvar95_seconds':100.0},{'method':'CVaR95 minimization','route':'S-B-T','mean_seconds':18.0,'cvar95_seconds':18.0}]);save_csv(tab,out/'tables'/'controlled_demo.csv');report['controlled_demo']='Two-route exact distribution; separate from empirical results.'

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase7',default='phase15_support/phase7_results.zip');p.add_argument('--phase11',default='phase15_support/phase11_results.zip');p.add_argument('--phase13',default='data/results/phase13_full_results.zip');p.add_argument('--cache',default='data/real_networks');p.add_argument('--output',default='data/results/phase151');p.add_argument('--rho',type=float,default=.65);p.add_argument('--dev',type=int,default=5000);p.add_argument('--shift',choices=['iid_gamma','correlated_gamma','heavy_lognormal'],default='heavy_lognormal');p.add_argument('--max-maps',type=int,default=3);p.add_argument('--seed',type=int,default=1301);p.add_argument('--budget',type=int,default=12);args=p.parse_args()
    if args.max_maps<1:raise ValueError('--max-maps must be positive')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    report={'phase':'15.1','synthetic':{},'phase11':{},'phase13':{},'warnings':['Route maps are reconstructed using original graph caches and deterministic candidate generation; do not interpret mismatched caches as original routes.','Lowest held-out CVaR95 is scenario-specific, not universal superiority.','Phase 7 costs are abstract units; road networks use seconds.','Phase 11/13 and synthetic examples are separate studies.','Correlated Gamma rho is a mixing weight, not a Pearson correlation.']}
    synthetic(args,out,report);phase11(args,out,report);phase13(args,out,report);demonstration(out,report)
    (out/'audit_summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k in ('synthetic','phase11','phase13')},indent=2));print('Phase 15.1 complete:',out)
if __name__=='__main__':main()
