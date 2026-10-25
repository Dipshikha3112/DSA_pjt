"""Phase 15: evidence-grounded route maps and publication figures.
Reads saved Phase 7, 11, 13 and 14 results; does not rerun Monte Carlo.
"""
import argparse, ast, csv, io, json, math, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from phase15_support.benchmark_phase7 import graph_for

METHOD_NAMES = {
 'expected_dijkstra':'Dijkstra (expected cost)', 'monte_carlo_saa':'Monte Carlo SAA',
 'robust_minimax':'Robust minimax', 'mean_variance':'Mean-variance',
 'dijkstra_development_mean':'Dijkstra (dev. mean)', 'candidate_cvar95':'Candidate CVaR95',
 'iid_cvar95':'Standard CVaR95','mixture_cvar95':'Mixture CVaR95',
 'worst_model_cvar95':'Worst-model CVaR95','minimax_regret_cvar95':'Minimax CVaR regret'}
COLORS = ['#1368aa','#d64d35','#188b68','#8e58a5','#b88b13','#555555','#e378bd']

def read_zip_csv(zip_path, filename):
 with zipfile.ZipFile(zip_path) as z:
  matches=[n for n in z.namelist() if n.endswith(filename)]
  if not matches: raise FileNotFoundError(f'{filename} absent from {zip_path}')
  return pd.read_csv(io.BytesIO(z.read(matches[0])))

def write_table(df, path):
 path.parent.mkdir(parents=True,exist_ok=True);df.to_csv(path,index=False)

def format_method(m, lam=None):
 if m=='mean_variance' and pd.notna(lam): return f'Mean-variance (λ={lam:g})'
 return METHOD_NAMES.get(m,m)

def best_rows(df, metric):
 a=df.copy();a['method_label']=[format_method(m,l) for m,l in zip(a.algorithm,a.risk_lambda)] if 'algorithm' in a else a.method.map(METHOD_NAMES).fillna(a.method)
 return a.sort_values([metric,'method_label']).drop_duplicates('method_label')

def plot_route_panel(g, positions, entries, title, path, metric, subtitle='', is_multi=False):
 """Draw the SAME topology with each method's selected path; observed metric labels."""
 entries=list(entries)
 if not entries:return
 n=len(entries); cols=min(3,n); rows=math.ceil(n/cols)
 fig,axs=plt.subplots(rows,cols,figsize=(6.2*cols,5.0*rows),squeeze=False)
 vals=[float(e['value']) for e in entries];best=min(vals)
 for i,entry in enumerate(entries):
  ax=axs.flat[i]; route=entry['path'];val=float(entry['value'])
  nx.draw_networkx_edges(g,positions,ax=ax,width=.65,alpha=.20,edge_color='#888888',arrows=False)
  if route and all(x in g for x in route) and all(g.has_edge(u,v) for u,v in zip(route[:-1],route[1:])):
   nx.draw_networkx_edges(g,positions,edgelist=list(zip(route[:-1],route[1:])),ax=ax,
    edge_color=COLORS[i%len(COLORS)],width=3.2,alpha=.95,arrows=False)
   nx.draw_networkx_nodes(g,positions,nodelist=[route[0]],node_color='#16a085',node_size=55,ax=ax)
   nx.draw_networkx_nodes(g,positions,nodelist=[route[-1]],node_color='#c0392b',node_size=55,ax=ax)
  ax.set_title(f"{entry['label']}\n{metric}: {val:.3f}"+('  ★ lowest' if abs(val-best)<1e-9 else ''),fontsize=10)
  ax.axis('off')
 for ax in axs.flat[n:]:ax.axis('off')
 fig.suptitle(title+'\n'+subtitle,fontsize=13,y=.995)
 fig.tight_layout(rect=[0,0,1,.94]);path.parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,dpi=170,bbox_inches='tight');plt.close(fig)

def parse_route(path):
 return [int(x) for x in str(path).split('->')]

def phase7(zpath,out):
 df=read_zip_csv(zpath,'benchmark_results.csv'); df['risk_lambda']=pd.to_numeric(df.risk_lambda,errors='coerce')
 # A single recorded experiment for each topology, selected without looking at outcomes.
 for topo in ('erdos_renyi','grid','scale_free'):
  sub=df[(df.topology==topo)&(df.requested_n==25)&(df.seed==1)&(df.distribution=='gamma')&(df.candidate_budget==8)].copy()
  if sub.empty:continue
  g=graph_for(topo,25,1)
  pos=nx.spring_layout(g,seed=15) if topo!='grid' else {i:(i%5,-i//5) for i in g.nodes}
  entries=[]
  for _,r in sub.iterrows():
   entries.append(dict(label=format_method(r.algorithm,r.risk_lambda),path=parse_route(r.path),value=r.cvar95))
  plot_route_panel(g,pos,entries,f'Phase 7 | {topo.replace("_"," ").title()}',out/'maps'/f'synthetic_{topo}.png',
   'held-out CVaR95 (cost units)','Gamma uncertainty; seed 1; 25 nodes; candidate budget 8')
 # Aggregated descriptive comparison, no invalid pooling of incompatible scales.
 summary=df.groupby(['topology','algorithm'],as_index=False).agg(mean_cvar95=('cvar95','mean'),cases=('cvar95','size'))
 write_table(summary,out/'tables'/'phase7_topology_comparison.csv')
 for topo,sub in summary.groupby('topology'):
  fig,ax=plt.subplots(figsize=(9,4.8));ax.barh(sub.algorithm,sub.mean_cvar95)
  ax.set(xlabel='Mean held-out CVaR95 (Phase 7 cost units)',title=f'Phase 7: {topo} — descriptive means')
  fig.tight_layout();fig.savefig(out/'figures'/f'phase7_{topo}_comparison.png',dpi=180);plt.close(fig)
 return df

def graph_from_cache(path):
 import osmnx as ox
 g=ox.io.load_graphml(path)
 from src.phase11_real_networks import simplify_parallel
 h=simplify_parallel(g)
 # GraphML coords may be strings after loading; OSMnx usually converts them.
 pos={n:(float(d['x']),float(d['y'])) for n,d in g.nodes(data=True) if 'x' in d and 'y' in d}
 return h,pos

def phase11(zpath,cache,out):
 df=read_zip_csv(zpath,'algorithm_results.csv');audit=read_zip_csv(zpath,'candidate_audit.csv')
 from src.phase11_real_networks import candidates
 for area in ('urban','suburban','constrained'):
  file=cache/f'{area}.graphml'
  if not file.exists(): print('SKIP Phase 11 map (cache missing):',file);continue
  h,pos=graph_from_cache(file)
  aa=audit[(audit.area==area)&(audit.pair==0)]
  if aa.empty:continue
  a=aa.iloc[0]
  # Node IDs may be numeric or strings. Resolve against graph nodes.
  lookup={str(n):n for n in h.nodes};s=lookup.get(str(a.source));t=lookup.get(str(a.target))
  if s is None or t is None:print('SKIP Phase 11 map: OD nodes not found',area);continue
  pool=candidates(h,s,t,int(a.candidate_count))
  sub=df[(df.area==area)&(df.pair==0)&(df['shift']=='correlated_gamma')]
  entries=[]
  for _,r in sub.iterrows():
   idx=int(r.route_id)
   if idx>=len(pool):raise RuntimeError(f'Phase11 route index {idx} missing in {area}')
   entries.append(dict(label=METHOD_NAMES.get(r.method,r.method),path=pool[idx],value=r.cvar95_seconds))
  plot_route_panel(h,pos,entries,f'Phase 11 | {area.title()} road network',out/'maps'/f'road_{area}.png',
   'held-out CVaR95 (seconds)','Correlated Gamma evaluation; pair 0; green=start, red=destination')
 summary=df.groupby(['area','shift','method'],as_index=False).agg(mean_cvar95_seconds=('cvar95_seconds','mean'),pairs=('pair','nunique'))
 write_table(summary,out/'tables'/'phase11_road_comparison.csv')
 return df

def phase13(zpath,cache,out,rho,dev,shift,seed=1301,budget=12,pair=0):
 from src.phase11_real_networks import od_pairs,candidates
 df=read_zip_csv(zpath,'algorithm_results.csv')
 areas=['pune','hyderabad','chennai','jaipur','kolkata']
 selected=df[(np.isclose(df.rho,rho))&(df.dev_n==dev)&(df['shift']==shift)]
 if selected.empty:raise ValueError('No Phase 13 rows match requested rho, dev and shift')
 summary=selected.groupby(['area','method'],as_index=False).agg(mean_cvar95_seconds=('cvar95_seconds','mean'),mean_travel_seconds=('mean_seconds','mean'),pairs=('pair','nunique'))
 write_table(summary,out/'tables'/'phase13_city_method_comparison.csv')
 baseline=selected[selected.method=='dijkstra_development_mean'][['area','pair','cvar95_seconds','mean_seconds']].rename(columns={'cvar95_seconds':'baseline_cvar95','mean_seconds':'baseline_mean'})
 merged=selected.merge(baseline,on=['area','pair'],validate='many_to_one')
 merged['delta_cvar95_seconds']=merged.cvar95_seconds-merged.baseline_cvar95
 merged['delta_mean_seconds']=merged.mean_seconds-merged.baseline_mean
 paired=merged[merged.method!='dijkstra_development_mean']
 effects=paired.groupby(['area','method'],as_index=False).agg(mean_delta_cvar95_seconds=('delta_cvar95_seconds','mean'),mean_delta_mean_seconds=('delta_mean_seconds','mean'),pairs=('pair','nunique'))
 write_table(effects,out/'tables'/'phase13_city_paired_effects.csv')
 for area,sub in effects.groupby('area'):
  fig,ax=plt.subplots(figsize=(9,4.6));ax.barh(sub.method.map(METHOD_NAMES),sub.mean_delta_cvar95_seconds)
  ax.axvline(0,color='black',lw=1);ax.set(xlabel='Δ CVaR95 vs Dijkstra (seconds; negative is better)',title=f'{area.title()} | {shift} | ρ={rho} | N={dev}')
  fig.tight_layout();fig.savefig(out/'figures'/f'phase13_{area}_paired_effects.png',dpi=180);plt.close(fig)
 for ai,area in enumerate(areas):
  file=cache/f'phase13_{area}.graphml'
  if not file.exists():print('SKIP Phase 13 map (cache missing):',file);continue
  h,pos=graph_from_cache(file)
  od=od_pairs(h,15,seed+ai*701)
  if pair>=len(od):print('SKIP Phase 13 map (pair unavailable):',area);continue
  s,t=od[pair];pool=candidates(h,s,t,budget)
  sub=selected[(selected.area==area)&(selected.pair==pair)]
  entries=[]
  for _,r in sub.iterrows():
   idx=int(r.route_id)
   if idx>=len(pool):raise RuntimeError(f'Phase13 {area}: recorded route {idx} exceeds reconstructed pool {len(pool)}; check graph and seed')
   entries.append(dict(label=METHOD_NAMES.get(r.method,r.method),path=pool[idx],value=r.cvar95_seconds))
  plot_route_panel(h,pos,entries,f'Phase 13 | {area.title()} road network',out/'maps'/f'phase13_{area}_{shift}.png',
   'held-out CVaR95 (seconds)',f'{shift}; mixing weight ρ={rho}; N={dev}; pair={pair}; green=start, red=destination')
 return selected

def phase14(zpath,out):
 df=read_zip_csv(zpath,'primary_comparisons.csv');write_table(df,out/'tables'/'phase14_primary_comparisons.csv')
 fig,ax=plt.subplots(figsize=(11,6));x=np.arange(len(df));ax.errorbar(x,df['mean'],yerr=[df['mean']-df['ci_low'],df['ci_high']-df['mean']],fmt='o',capsize=3)
 ax.axhline(0,color='black',lw=1);ax.set_xticks(x,labels=[f'{s}\n{m}' for s,m in zip(df['shift'],df['method'])],rotation=75,ha='right',fontsize=8)
 ax.set(ylabel='Δ CVaR95 vs Dijkstra (seconds)',title='Phase 14: exploratory hierarchical bootstrap intervals (not significance claims)')
 fig.tight_layout();fig.savefig(out/'figures'/'phase14_primary_intervals.png',dpi=180);plt.close(fig)
 return df

def main():
 p=argparse.ArgumentParser();p.add_argument('--phase7',default='phase15_support/phase7_results.zip');p.add_argument('--phase11',default='phase15_support/phase11_results.zip');p.add_argument('--phase13',default='data/results/phase13_full_results.zip');p.add_argument('--phase14',default='data/results/phase14_results.zip');p.add_argument('--cache',default='data/real_networks');p.add_argument('--output',default='data/results/phase15');p.add_argument('--rho',type=float,default=.65);p.add_argument('--dev',type=int,default=5000);p.add_argument('--shift',default='correlated_gamma');p.add_argument('--pair',type=int,default=0)
 a=p.parse_args();out=Path(a.output);(out/'maps').mkdir(parents=True,exist_ok=True);(out/'figures').mkdir(exist_ok=True);(out/'tables').mkdir(exist_ok=True)
 report={'phase':15,'scope':'visualization of recorded held-out metrics','maps':[],'warnings':['Best means lowest recorded held-out CVaR95 in the selected case; it does not imply general superiority.','Phase 7 costs are abstract cost units, not seconds.','Phase 11 and Phase 13 are different experiments; do not pool them.','Reconstructed routes require unchanged graph caches and matching candidate order.','Monte Carlo SAA is a route selection method; Monte Carlo is not a physical route.']}
 for label,func in [('Phase 7',lambda:phase7(Path(a.phase7),out)),('Phase 11',lambda:phase11(Path(a.phase11),Path(a.cache),out)),('Phase 13',lambda:phase13(Path(a.phase13),Path(a.cache),out,a.rho,a.dev,a.shift,pair=a.pair)),('Phase 14',lambda:phase14(Path(a.phase14),out))]:
  try:func();print(label,'COMPLETE',flush=True)
  except (FileNotFoundError,ModuleNotFoundError) as e: print(label,'SKIPPED:',e,flush=True);report['warnings'].append(f'{label} skipped: {e}')
 report['maps']=[x.name for x in sorted((out/'maps').glob('*.png'))]
 (out/'research_summary.json').write_text(json.dumps(report,indent=2),encoding='utf8')
 print(f"Saved {len(report['maps'])} route maps to {out/'maps'}",flush=True)
 print('NOTE: Phase 7 visualizations use the saved paths; Phase 11/13 use reconstructed candidates.',flush=True)
if __name__=='__main__':main()
