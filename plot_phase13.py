"""Post-run Phase 13 scientific figures; never alters benchmark results."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.phase13_validation import FRESH_AREAS, uncertain_costs
from src.phase11_real_networks import load_graph, simplify_parallel, od_pairs, candidates, cvar


def rows(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def save(fig, path):
    fig.savefig(path, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print('Saved:', path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='data/results/phase13_full')
    ap.add_argument('--area', default='pune', choices=list(FRESH_AREAS))
    ap.add_argument('--pair', type=int, default=0)
    ap.add_argument('--rho', type=float, default=.65)
    ap.add_argument('--shift', default='correlated_gamma', choices=['iid_gamma','heavy_lognormal','correlated_gamma'])
    ap.add_argument('--samples', type=int, default=10000)
    args = ap.parse_args()
    root=Path(args.results)
    report=json.loads((root/'research_report.json').read_text(encoding='utf-8'))
    cfg=report['parameters']
    audits=rows(root/'candidate_audit.csv')
    area_audits=[r for r in audits if r['area']==args.area and int(r['pair'])==args.pair and abs(float(r['rho'])-args.rho)<1e-8]
    if not area_audits: raise ValueError('Requested area/pair/rho not in Phase 13 results')
    audit=max(area_audits,key=lambda r:int(r['dev_n']))
    network_order=list(FRESH_AREAS)
    ai=network_order.index(args.area)
    g,_=load_graph('phase13_'+args.area,FRESH_AREAS[args.area],cfg['cache'],int(cfg['dist']),False)
    h=simplify_parallel(g)
    pairs=od_pairs(h,int(cfg['pairs']),int(cfg['seed'])+ai*701)
    s,t=pairs[args.pair]
    paths=candidates(h,s,t,int(cfg['budget']))
    dj=int(audit['route_dijkstra_development_mean'])
    cv=int(audit['route_iid_cvar95'])
    if max(dj,cv)>=len(paths):raise RuntimeError('Candidate reconstruction mismatch')
    folder=root/'figures';folder.mkdir(exist_ok=True)
    prefix=f'{args.area}_pair{args.pair}_rho{args.rho:g}'

    # Plot real graph using OSMnx-provided x/y coordinates, if available.
    positions={node:(float(d['x']),float(d['y'])) for node,d in g.nodes(data=True) if 'x' in d and 'y' in d}
    if len(positions)>0:
        fig,ax=plt.subplots(figsize=(9,8))
        for u,v in h.edges:
            if u in positions and v in positions:
                ax.plot([positions[u][0],positions[v][0]],[positions[u][1],positions[v][1]],color='#b8c0c8',linewidth=.35,zorder=1)
        for path,color,name in [(paths[dj],'#e67e22','Dijkstra'),(paths[cv],'#2064ad','Standard CVaR95')]:
            coords=[positions[x] for x in path if x in positions]
            if len(coords)>1:
                ax.plot([p[0] for p in coords],[p[1] for p in coords],color=color,linewidth=2.5,label=name,zorder=3)
        for node,color,name in [(s,'green','Origin'),(t,'red','Destination')]:
            if node in positions:ax.scatter(*positions[node],c=color,s=70,label=name,zorder=5)
        ax.set_title(f'{args.area.title()}: road network and selected routes\nPair {args.pair}, rho={args.rho:g} (mixing weight)')
        ax.set_aspect('equal');ax.axis('off');ax.legend(loc='best')
        save(fig,folder/(prefix+'_routes.png'))
    else:print('No OSM x/y coordinates; skipping geographic plot')

    # Fresh visualization-only draws, never used to select a route or claim benchmark estimates.
    vals=uncertain_costs(h,paths,args.samples,seed=202613+ai*100+args.pair,model=args.shift,rho=args.rho)
    fig,ax=plt.subplots(figsize=(9,5))
    for j,name,color in [(dj,'Dijkstra','#e67e22'),(cv,'Standard CVaR95','#2064ad')]:
        ax.hist(vals[:,j],bins=65,alpha=.38,density=True,label=f'{name}: CVaR95={cvar(vals[:,j]):.1f}s',color=color)
    ax.set(title=f'Monte Carlo route travel times — {args.area.title()} ({args.shift})',xlabel='Travel time (seconds)',ylabel='Probability density')
    ax.legend();save(fig,folder/(prefix+'_monte_carlo_hist.png'))

    fig,ax=plt.subplots(figsize=(9,5))
    for j,name,color in [(dj,'Dijkstra','#e67e22'),(cv,'Standard CVaR95','#2064ad')]:
        sample=vals[:,j]; threshold=float(np.quantile(sample,.95))
        ax.hist(sample[sample>=threshold],bins=35,alpha=.42,density=False,color=color,label=f'{name} worst 5%')
        ax.axvline(threshold,color=color,linestyle='--',linewidth=1.5)
    ax.set(title=f'Worst 5% Monte Carlo outcomes — {args.area.title()}',xlabel='Travel time (seconds)',ylabel='Scenario count')
    ax.legend();save(fig,folder/(prefix+'_cvar_tail.png'))

    # Convergence of an estimator for a FIXED route; distinct from decision stability CSV.
    sizes=[n for n in [100,250,500,1000,2000,5000,args.samples] if n<=args.samples]
    sizes=sorted(set(sizes))
    fig,ax=plt.subplots(figsize=(9,5))
    for j,name,color in [(dj,'Dijkstra','#e67e22'),(cv,'Standard CVaR95','#2064ad')]:
        ax.plot(sizes,[cvar(vals[:n,j]) for n in sizes],marker='o',color=color,label=name)
    ax.set(title='Monte Carlo CVaR95 convergence (fixed selected routes)',xlabel='Number of scenarios',ylabel='Estimated CVaR95 (seconds)')
    ax.legend();ax.grid(alpha=.25);save(fig,folder/(prefix+'_convergence.png'))
    print('All figures saved in',folder)

if __name__=='__main__':main()
