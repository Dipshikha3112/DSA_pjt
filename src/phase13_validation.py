"""Phase 13: independent topology validation and Monte Carlo sensitivity.

All routes selected using development scenarios; evaluation sample is fixed,
independent, and shared by every method and development sample size.
"""
import argparse
import csv
import json
import math
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from src.phase11_real_networks import load_graph, simplify_parallel, od_pairs, candidates, cvar, write_csv
from src.phase12_robust import select_methods, METHODS, SHIFTS

# Fresh OSM study regions; none of the three Phase 11/12 locations are reused.
FRESH_AREAS = {
    'pune': 'Shivajinagar, Pune, India',
    'hyderabad': 'Gachibowli, Hyderabad, India',
    'chennai': 'Adyar, Chennai, India',
    'jaipur': 'Civil Lines, Jaipur, India',
    'kolkata': 'Salt Lake Sector V, Kolkata, India',
}


def uncertain_costs(graph, paths, n, seed, model, rho=.65):
    """Generate shared edge-level random travel times for all candidate paths.

    For correlated Gamma, rho is a mixing weight, not Pearson correlation.
    """
    if not 0 <= rho <= 1:
        raise ValueError('rho must be between zero and one')
    if n < 1 or not paths:
        raise ValueError('positive sample size and nonempty paths required')
    rng = np.random.default_rng(seed)
    edges = sorted({(u, v) for p in paths for u, v in zip(p[:-1], p[1:])}, key=str)
    edge_ids = {e: i for i, e in enumerate(edges)}
    base = np.array([graph[u][v]['base'] for u, v in edges], dtype=float)
    if model == 'iid_gamma':
        multiplier = rng.gamma(16, 1 / 16, size=(n, len(edges)))
    elif model == 'heavy_lognormal':
        sigma = .65
        multiplier = np.exp(rng.normal(-sigma * sigma / 2, sigma, size=(n, len(edges))))
    elif model == 'correlated_gamma':
        shared = rng.gamma(4, 1 / 4, size=(n, 1))
        private = rng.gamma(16, 1 / 16, size=(n, len(edges)))
        multiplier = rho * shared + (1 - rho) * private
    else:
        raise ValueError('unknown model')
    costs = multiplier * base[None, :]
    incidence = np.zeros((len(edges), len(paths)), dtype=np.float64)
    for j, p in enumerate(paths):
        for e in zip(p[:-1], p[1:]):
            incidence[edge_ids[e], j] += 1
    return costs @ incidence


def nested_development(graph, paths, max_n, seed, rho):
    return {model: uncertain_costs(graph, paths, max_n, seed + 5003 * k, model, rho)
            for k, model in enumerate(SHIFTS)}


def paired_ci(rows, field, seed, reps=2000):
    """Hierarchical bootstrap: resample areas, then OD pairs within areas.

    Inference is exploratory with very few geographic areas.
    """
    by_area = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_area[row['area']][row['pair']].append(float(row[field]))
    areas = sorted(by_area)
    if not areas:
        return None, None
    rng = np.random.default_rng(seed)
    estimates = np.empty(reps)
    for r in range(reps):
        draws = []
        for a in rng.choice(areas, size=len(areas), replace=True):
            od = by_area[a]
            ids = sorted(od)
            for p in rng.choice(ids, size=len(ids), replace=True):
                draws.append(np.mean(od[p]))
        estimates[r] = np.mean(draws)
    return tuple(float(x) for x in np.quantile(estimates, [.025, .975]))


def run(args):
    out = Path(args.output)
    checkpoint = out / 'checkpoint.json'
    if out.exists() and any(out.iterdir()):
        if not getattr(args, 'resume', False) or not checkpoint.is_file():
            raise FileExistsError(f'Output directory not empty: {out}; use --resume if a checkpoint exists')
    out.mkdir(parents=True, exist_ok=True)
    sizes = sorted({int(x) for x in args.dev_sizes.split(',')})
    if not sizes or sizes[0] < 20 or args.test < 100 or args.pairs < 1 or args.budget < 1 or args.bootstrap < 100:
        raise ValueError('invalid sample sizes or experiment parameters')
    rhos = sorted({float(x) for x in args.rhos.split(',')})
    if any(not 0 <= x <= 1 for x in rhos):
        raise ValueError('rho values must be in [0,1]')
    if args.demo:
        from src.phase11_real_networks import demo_graph
        area_specs = {'demo': 'offline synthetic demonstration'}
    else:
        area_specs = {k: FRESH_AREAS[k] for k in args.areas.split(',')}
    records, audits, networks = [], [], []
    completed = []
    if getattr(args, 'resume', False) and checkpoint.exists():
        saved = json.loads(checkpoint.read_text(encoding='utf-8'))
        if saved['config'] != {k:v for k,v in vars(args).items() if k not in ('resume',)}:
            raise ValueError('Checkpoint configuration differs; use identical experiment options for --resume')
        records, audits, networks, completed = (saved[k] for k in ('records', 'audits', 'networks', 'completed'))
        print('Resuming after completed areas:', ', '.join(completed), flush=True)
    for ai, (area, place) in enumerate(area_specs.items()):
        if area in completed:
            continue
        if args.demo:
            graph, source = demo_graph(), 'synthetic_demo'
        else:
            # Fresh cache labels avoid inadvertently reusing Phase 11/12 networks.
            graph, source = load_graph('phase13_' + area, place, args.cache, args.dist, False)
        h = simplify_parallel(graph)
        pairs = od_pairs(h, args.pairs, args.seed + ai * 701)
        networks.append(dict(area=area, place=place, source=source, nodes=h.number_of_nodes(), edges=h.number_of_edges(), pairs=len(pairs)))
        print(f'{area}: {len(pairs)} pairs; {h.number_of_nodes()} nodes', flush=True)
        for pi, (s, t) in enumerate(pairs):
            start = time.perf_counter()
            paths = candidates(h, s, t, args.budget)
            generation_s = time.perf_counter() - start
            if not paths:
                continue
            for rho in rhos:
                dev = nested_development(h, paths, max(sizes), args.seed + ai * 10000000 + pi * 100000 + int(rho * 10000), rho)
                # Same held-out matrices reused across development sample sizes and methods.
                test = {model: uncertain_costs(h, paths, args.test,
                        args.seed + 500000000 + ai * 10000000 + pi * 100000 + int(rho * 10000) + 5003 * k,
                        model, rho) for k, model in enumerate(SHIFTS)}
                for n in sizes:
                    chosen, feasible, _ = select_methods({model: dev[model][:n] for model in SHIFTS}, args.epsilon, args.alpha)
                    audits.append(dict(area=area, pair=pi, rho=rho, dev_n=n, candidate_count=len(paths), feasible_count=feasible,
                                       selected_unique_routes=len(set(chosen.values())), candidate_generation_seconds=generation_s,
                                       **{'route_' + m: int(chosen[m]) for m in METHODS}))
                    for model in SHIFTS:
                        for method in METHODS:
                            j = chosen[method]
                            vals = test[model][:, j]
                            records.append(dict(area=area, pair=pi, rho=rho, dev_n=n, shift=model, method=method, route_id=int(j),
                                                mean_seconds=float(np.mean(vals)), cvar95_seconds=cvar(vals, args.alpha),
                                                p95_seconds=float(np.quantile(vals, args.alpha))))
        completed.append(area)
        # Save atomically after each complete area, so interruption loses at most the active area.
        payload = dict(config={k:v for k,v in vars(args).items() if k != 'resume'},
                       records=records, audits=audits, networks=networks, completed=completed)
        temp = checkpoint.with_suffix('.tmp')
        temp.write_text(json.dumps(payload), encoding='utf-8')
        temp.replace(checkpoint)
        print(f'Checkpoint saved after {area}: {len(records)} method records', flush=True)
    write_csv(out/'algorithm_results.csv', records)
    write_csv(out/'candidate_audit.csv', audits)
    indexed = {(r['area'], r['pair'], r['rho'], r['dev_n'], r['shift'], r['method']): r for r in records}
    paired = []
    for (area, pair, rho, n, shift, method), r in indexed.items():
        if method == 'dijkstra_development_mean':
            continue
        base = indexed[area, pair, rho, n, shift, 'dijkstra_development_mean']
        std = indexed[area, pair, rho, n, shift, 'iid_cvar95']
        paired.append(dict(area=area, pair=pair, rho=rho, dev_n=n, shift=shift, method=method,
                           delta_vs_dijkstra_cvar95=r['cvar95_seconds'] - base['cvar95_seconds'],
                           delta_vs_standard_cvar95=r['cvar95_seconds'] - std['cvar95_seconds'],
                           delta_vs_dijkstra_mean=r['mean_seconds'] - base['mean_seconds'],
                           changed_from_dijkstra=int(r['route_id'] != base['route_id'])))
    write_csv(out/'paired_results.csv', paired)
    summary = []
    for rho in rhos:
        for n in sizes:
            for shift in SHIFTS:
                for method in METHODS[1:]:
                    sub = [r for r in paired if r['rho'] == rho and r['dev_n'] == n and r['shift'] == shift and r['method'] == method]
                    if not sub:
                        continue
                    ci = paired_ci(sub, 'delta_vs_dijkstra_cvar95', args.seed + 37, args.bootstrap)
                    ci_std = paired_ci(sub, 'delta_vs_standard_cvar95', args.seed + 47, args.bootstrap)
                    summary.append(dict(rho=rho, dev_n=n, shift=shift, method=method, observations=len(sub), areas=len({r['area'] for r in sub}),
                                        mean_delta_vs_dijkstra=float(np.mean([r['delta_vs_dijkstra_cvar95'] for r in sub])),
                                        ci_low=ci[0], ci_high=ci[1],
                                        mean_delta_vs_standard=float(np.mean([r['delta_vs_standard_cvar95'] for r in sub])),
                                        standard_ci_low=ci_std[0], standard_ci_high=ci_std[1],
                                        mean_delta_mean=float(np.mean([r['delta_vs_dijkstra_mean'] for r in sub])),
                                        changed_routes=sum(r['changed_from_dijkstra'] for r in sub)))
    write_csv(out/'sensitivity_summary.csv', summary)
    # Route-selection stability against the largest development sample, not ground-truth optimality.
    audit_idx = {(r['area'], r['pair'], r['rho'], r['dev_n']): r for r in audits}
    stability = []
    for r in audits:
        reference = audit_idx[r['area'], r['pair'], r['rho'], max(sizes)]
        for method in METHODS:
            stability.append(dict(area=r['area'], pair=r['pair'], rho=r['rho'], dev_n=r['dev_n'], method=method,
                                  agrees_with_largest_n=int(r['route_' + method] == reference['route_' + method])))
    write_csv(out/'route_stability.csv', stability)
    report = dict(phase=13, title='Fresh-network validation and Monte Carlo convergence sensitivity',
                  design='OFFLINE SYNTHETIC DEMO, NOT REAL ROAD EVIDENCE' if args.demo else 'Fresh OSM topology with simulated uncertain travel times',
                  networks=networks, pairs=sum(x['pairs'] for x in networks), method_records=len(records),
                  parameters=vars(args), development_sample_sizes=sizes, correlated_gamma_shared_mixing_weights=rhos,
                  primary_comparison='iid_cvar95 minus dijkstra_development_mean on correlated_gamma at rho=0.65 and largest development N; CVaR95, held-out',
                  limitations=['All traffic uncertainty is simulated; no observed travel times',
                               'The correlation parameter is a shared-factor mixing weight, not Pearson correlation',
                               'Development sample sizes are nested; comparisons across N are dependent',
                               'Held-out samples are fixed across development sizes and methods for paired evaluation',
                               'All model families are known at development for multi-model baselines',
                               'A small number of geographic areas limits population-level inference',
                               'Bootstrap intervals exploratory; many secondary comparisons unadjusted',
                               'Route candidates limited by shortest-simple-path budget; not global CVaR optimality',
                               'Road networks approximate road travel without turn restrictions'])
    (out/'research_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    archive = out.parent / (out.name + '_results.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()):
            z.write(p, p.name)
    print(f'Phase 13 complete: {len(records)} records; results: {archive}', flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--demo', action='store_true')
    p.add_argument('--resume', action='store_true', help='Resume an interrupted run after a completed-area checkpoint')
    p.add_argument('--areas', default='pune,hyderabad,chennai,jaipur,kolkata')
    p.add_argument('--dist', type=int, default=1200)
    p.add_argument('--pairs', type=int, default=15)
    p.add_argument('--budget', type=int, default=12)
    p.add_argument('--dev-sizes', default='250,500,1000,2000,5000')
    p.add_argument('--test', type=int, default=10000)
    p.add_argument('--rhos', default='0.0,0.35,0.65,0.9')
    p.add_argument('--epsilon', type=float, default=.10)
    p.add_argument('--alpha', type=float, default=.95)
    p.add_argument('--seed', type=int, default=1301)
    p.add_argument('--bootstrap', type=int, default=1000)
    p.add_argument('--cache', default='data/real_networks')
    p.add_argument('--output', default='data/results/phase13_full')
    a = p.parse_args()
    if not a.demo and any(x not in FRESH_AREAS for x in a.areas.split(',')):
        p.error(f'Unknown area; choose from {list(FRESH_AREAS)}')
    if not (0 < a.alpha < 1 and a.epsilon >= 0 and a.dist >= 100):
        p.error('Invalid alpha, epsilon or distance')
    run(a)

if __name__ == '__main__':
    main()
