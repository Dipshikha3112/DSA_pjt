"""Phase 11: road topology + explicitly simulated travel-time uncertainty."""
import argparse,csv,json,math,time,zipfile
from pathlib import Path
import numpy as np
import networkx as nx

PLACES={'urban':'Connaught Place, New Delhi, India','suburban':'Whitefield, Bengaluru, India','constrained':'Fort Kochi, Kochi, India'}

def demo_graph():
    g=nx.MultiDiGraph()
    for u,v,t in [('s','a',5),('a','t',5),('s','b',6),('b','t',6),('a','b',2),('b','a',2),('s','c',7),('c','t',7)]:
        g.add_edge(u,v,length=t*100,speed_kph=36,travel_time=float(t*10),highway='residential')
    return g

def load_graph(label, place, cache, dist=1200, offline=False):
    """Load cached OSM graph or download it with bounded endpoint retries.

    Only a fully processed graph is cached. The caller's experiment parameters,
    candidate generation, and random-number generation are unchanged.
    """
    if offline:
        return demo_graph(), 'synthetic_demo'
    try:
        import osmnx as ox
    except ImportError as exc:
        raise RuntimeError('Install osmnx: python -m pip install "osmnx>=2,<3"') from exc

    import os
    import random

    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    file = cache / f'{label}.graphml'

    if file.exists():
        print(f'Loading cached OSM graph: {file}', flush=True)
        g = ox.io.load_graphml(file)
    else:
        # Override with PHASE13_OVERPASS_URLS, separated by commas, if needed.
        configured = os.environ.get('PHASE13_OVERPASS_URLS', '').strip()
        endpoints = ([x.strip() for x in configured.split(',') if x.strip()]
                     if configured else [
                         'https://overpass.kumi.systems/api',
                         'https://overpass-api.de/api',
                         'https://overpass.nchc.org.tw/api',
                     ])
        ox.settings.use_cache = True
        ox.settings.requests_timeout = 120
        failures = []
        g = None
        for endpoint in endpoints:
            if not endpoint.startswith('https://'):
                raise ValueError(f'Invalid Overpass endpoint: {endpoint!r}')
            ox.settings.overpass_url = endpoint
            for attempt in range(1, 3):
                try:
                    print(f'OSM download for {label}: {endpoint} (attempt {attempt}/2)', flush=True)
                    g = ox.graph.graph_from_address(
                        place, dist=dist, network_type='drive', simplify=True
                    )
                    break
                except Exception as exc:
                    # Only connectivity/service errors should trigger fallback.
                    import requests
                    network_error = isinstance(exc, requests.exceptions.RequestException)
                    msg = str(exc).lower()
                    server_error = any(x in msg for x in (
                        'overpass', 'timeout', 'timed out', 'connection',
                        'too many requests', '429', '502', '503', '504'
                    ))
                    if not (network_error or server_error):
                        raise
                    failures.append(f'{endpoint} attempt {attempt}: {type(exc).__name__}: {exc}')
                    print(f'Network/server issue: {type(exc).__name__}. Trying again/fallback.', flush=True)
                    if attempt == 1:
                        time.sleep(5 + random.uniform(0, 2))
            if g is not None:
                break
        if g is None:
            raise RuntimeError(
                f'Unable to download OSM graph for {label}. '
                'Your Phase 13 checkpoint is unchanged. Attempts:\n' +
                '\n'.join(failures)
            )
        g = ox.routing.add_edge_speeds(g, fallback=35)
        g = ox.routing.add_edge_travel_times(g)
        ox.io.save_graphml(g, file)
        print(f'Cached OSM graph: {file}', flush=True)

    if not all('travel_time' in d for _, _, _, d in g.edges(keys=True, data=True)):
        g = ox.routing.add_edge_speeds(g, fallback=35)
        g = ox.routing.add_edge_travel_times(g)
    return g, 'openstreetmap'

def simplify_parallel(g):
    """Keep the fastest directed parallel edge. Documented approximation."""
    h=nx.DiGraph()
    for u,v,d in g.edges(data=True):
        t=float(d.get('travel_time',0))
        if t<=0 or not math.isfinite(t):continue
        if not h.has_edge(u,v) or t<h[u][v]['base']:
            h.add_edge(u,v,base=t,highway=str(d.get('highway','unknown')))
    return h

def od_pairs(h,n,seed):
    rng=np.random.default_rng(seed);nodes=list(h.nodes)
    if len(nodes)<2:return []
    pairs=[];used=set()
    for _ in range(max(100,n*60)):
        s,t=rng.choice(len(nodes),size=2,replace=False)
        u,v=nodes[int(s)],nodes[int(t)]
        if (u,v) in used:continue
        if nx.has_path(h,u,v):
            used.add((u,v));pairs.append((u,v))
            if len(pairs)==n:break
    return pairs

def candidates(h,s,t,k=12):
    try:
        it=nx.shortest_simple_paths(h,s,t,weight='base')
        import itertools
        return [tuple(p) for p in itertools.islice(it,k)]
    except (nx.NetworkXNoPath,nx.NodeNotFound):return []

def scenario_matrix(h,paths,n,seed,shift):
    """Positive simulated edge costs, same draws across competing paths."""
    rng=np.random.default_rng(seed)
    edges=sorted({(u,v) for p in paths for u,v in zip(p[:-1],p[1:])},key=str)
    idx={e:i for i,e in enumerate(edges)}
    base=np.array([h[u][v]['base'] for u,v in edges]); m=len(edges)
    if shift=='iid_gamma':
        mult=rng.gamma(shape=16,scale=1/16,size=(n,m))
    elif shift=='heavy_lognormal':
        sigma=.65; mult=np.exp(rng.normal(-sigma*sigma/2,sigma,size=(n,m)))
    elif shift=='correlated_gamma':
        shared=rng.gamma(4,1/4,size=(n,1))
        private=rng.gamma(16,1/16,size=(n,m))
        mult=.65*shared+.35*private
    else:raise ValueError('Unknown shift')
    costs=mult*base[None,:]
    matrix=np.zeros((n,len(paths)))
    for j,p in enumerate(paths):
        for e in zip(p[:-1],p[1:]):matrix[:,j]+=costs[:,idx[e]]
    return matrix

def cvar(values,alpha=.95):
    a=np.sort(np.asarray(values,dtype=float));tail=max(1,int(math.ceil((1-alpha)*len(a))))
    return float(a[-tail:].mean())

def select(dev,epsilon=.1,alpha=.95):
    means=dev.mean(axis=0);anchor=int(np.argmin(means));eligible=np.flatnonzero(means <= (1+epsilon)*means[anchor]+1e-9)
    risk=int(min(eligible,key=lambda j:(cvar(dev[:,j],alpha),means[j],j)))
    return anchor,risk,int(len(eligible))

def bootstrap(values,clusters,seed=42,reps=1000):
    unique=sorted(set(clusters));rng=np.random.default_rng(seed)
    means=np.array([np.mean([v for v,c in zip(values,clusters) if c==u]) for u in unique])
    if not len(means):return [None,None]
    boot=np.mean(means[rng.integers(0,len(means),size=(reps,len(means)))],axis=1)
    return [float(x) for x in np.quantile(boot,[.025,.975])]

def write_csv(file,rows):
    if not rows:return
    with open(file,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def run(args):
    out=Path(args.output)
    if out.exists() and any(out.iterdir()):raise FileExistsError(f'Output directory not empty: {out}')
    out.mkdir(parents=True,exist_ok=True)
    rows=[];audits=[];meta=[]
    places={'demo':'offline demo'} if args.demo else {key:PLACES[key] for key in args.areas.split(',')}
    for a,(label,place) in enumerate(places.items()):
        g,source=load_graph(label,place,args.cache,args.dist,args.demo)
        h=simplify_parallel(g)
        pairs=od_pairs(h,args.pairs,args.seed+100*a)
        meta.append(dict(area=label,place=place,source=source,nodes=h.number_of_nodes(),edges=h.number_of_edges(),pairs=len(pairs)))
        print(f'{label}: {h.number_of_nodes()} nodes, {h.number_of_edges()} directed edges, {len(pairs)} pairs',flush=True)
        for pi,(s,t) in enumerate(pairs):
            start=time.perf_counter();paths=candidates(h,s,t,args.budget);runtime=time.perf_counter()-start
            if not paths:continue
            # One development draw reused across shifts, with no test leakage.
            dev=scenario_matrix(h,paths,args.dev,args.seed+100000*a+pi,'iid_gamma')
            di,ri,feasible=select(dev,args.epsilon)
            audits.append(dict(area=label,pair=pi,source=str(s),target=str(t),candidate_count=len(paths),feasible_count=feasible,route_changed=int(di!=ri),candidate_generation_seconds=runtime))
            for shift_id,shift in enumerate(['iid_gamma','heavy_lognormal','correlated_gamma']):
                test=scenario_matrix(h,paths,args.test,args.seed+500000+100000*a+100*pi+shift_id,shift)
                for method,j in [('dijkstra_development_mean',di),('candidate_cvar95',ri)]:
                    rows.append(dict(area=label,pair=pi,shift=shift,method=method,route_id=j,mean_seconds=float(test[:,j].mean()),cvar95_seconds=cvar(test[:,j]),p95_seconds=float(np.quantile(test[:,j],.95)),candidate_count=len(paths)))
    write_csv(out/'algorithm_results.csv',rows);write_csv(out/'candidate_audit.csv',audits)
    paired=[]
    index={(r['area'],r['pair'],r['shift'],r['method']):r for r in rows}
    for area,pair,shift in sorted({(r['area'],r['pair'],r['shift']) for r in rows}):
        d=index[area,pair,shift,'dijkstra_development_mean'];c=index[area,pair,shift,'candidate_cvar95']
        paired.append(dict(area=area,pair=pair,shift=shift,delta_cvar95_seconds=c['cvar95_seconds']-d['cvar95_seconds'],delta_mean_seconds=c['mean_seconds']-d['mean_seconds'],route_changed=int(c['route_id']!=d['route_id'])))
    write_csv(out/'paired_results.csv',paired)
    summary=[]
    for shift in ['iid_gamma','heavy_lognormal','correlated_gamma']:
        rr=[r for r in paired if r['shift']==shift]
        if rr:
            delta=[r['delta_cvar95_seconds'] for r in rr]
            cluster=[f"{r['area']}:{r['pair']}" for r in rr]
            summary.append(dict(shift=shift,pairs=len(rr),mean_delta_cvar95_seconds=float(np.mean(delta)),ci_low=bootstrap(delta,cluster,args.seed)[0],ci_high=bootstrap(delta,cluster,args.seed)[1],mean_delta_mean_seconds=float(np.mean([r['delta_mean_seconds'] for r in rr])),changed_routes=sum(r['route_changed'] for r in rr)))
    write_csv(out/'shift_summary.csv',summary)
    report=dict(phase=11,design='real OSM road topology with simulated travel-time uncertainty' if not args.demo else 'synthetic offline smoke test, NOT real-road evidence',network_metadata=meta,parameters=vars(args),pairs=sum(m['pairs'] for m in meta),method_records=len(rows),limitations=['Travel-time distributions are simulated, not measured traffic data','Parallel edges collapsed to minimum free-flow time; turn restrictions not modeled','Candidate-restricted CVaR is not globally optimal','IID Gamma development model; distribution shifts only in held-out evaluation','Source-destination pairs are sampled from the directed network; clusters are OD pairs nested within area','Only three areas: do not interpret bootstrap CI as geographic generalization','Bootstrap intervals are exploratory; multiple comparisons not adjusted'])
    (out/'research_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    zip_path=out.parent/(out.name+'_results.zip')
    with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
        for p in out.iterdir():z.write(p,p.name)
    print(f'Phase 11 complete: {len(meta)} networks, {report["pairs"]} OD pairs, {len(rows)} records. Results ZIP: {zip_path}')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--demo',action='store_true',help='Offline synthetic smoke test (not real OSM)')
    p.add_argument('--areas',default='urban,suburban,constrained')
    p.add_argument('--dist',type=int,default=1200,help='Radius in meters around place address')
    p.add_argument('--pairs',type=int,default=20)
    p.add_argument('--budget',type=int,default=12)
    p.add_argument('--dev',type=int,default=1200)
    p.add_argument('--test',type=int,default=3000)
    p.add_argument('--epsilon',type=float,default=.10)
    p.add_argument('--seed',type=int,default=1101)
    p.add_argument('--cache',default='data/real_networks')
    p.add_argument('--output',default='data/results/phase11_full')
    a=p.parse_args()
    if a.dev<20 or a.test<20 or a.budget<1 or a.pairs<1 or a.dist<100: p.error('Invalid experiment parameters')
    run(a)
if __name__=='__main__':main()
