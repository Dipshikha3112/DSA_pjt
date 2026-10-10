"""Phase 12: candidate-restricted multi-model risk routing on OSM topology.

All methods share one candidate pool. No held-out outcomes influence selection.
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
from src.phase11_real_networks import (
    PLACES, bootstrap, candidates, cvar, demo_graph, load_graph, od_pairs,
    scenario_matrix, simplify_parallel, write_csv,
)

SHIFTS = ('iid_gamma', 'heavy_lognormal', 'correlated_gamma')
METHODS = ('dijkstra_development_mean', 'iid_cvar95', 'mixture_cvar95', 'worst_model_cvar95', 'minimax_regret_cvar95')


def select_methods(development, epsilon=0.10, alpha=0.95):
    """Return selected candidate indices and feasible count.

    The anchor and feasibility bound use ONLY iid Gamma development means.
    For robustness, all models receive an independent development sample.
    Worst-model CVaR uses raw model-wise tail costs; regret subtracts the
    feasible model-wise minimum (no out-of-sample information).
    """
    if set(development) != set(SHIFTS):
        raise ValueError('Development scenarios must include exactly the three specified models')
    npaths = next(iter(development.values())).shape[1]
    if npaths < 1 or any(x.ndim != 2 or x.shape[1] != npaths or x.shape[0] < 20 for x in development.values()):
        raise ValueError('Invalid development scenario matrices')
    base_mean = development['iid_gamma'].mean(axis=0)
    anchor = int(np.argmin(base_mean))
    feasible = np.flatnonzero(base_mean <= (1 + epsilon) * base_mean[anchor] + 1e-9)
    risk = np.stack([[cvar(development[m][:, j], alpha) for j in range(npaths)] for m in SHIFTS])
    # Equal-weight mixture across models, with equal scenario counts per model.
    mixture = np.vstack([development[m] for m in SHIFTS])
    mixture_risk = np.array([cvar(mixture[:, j], alpha) for j in range(npaths)])
    model_best = risk[:, feasible].min(axis=1)
    scores = {
        'iid_cvar95': risk[0],
        'mixture_cvar95': mixture_risk,
        'worst_model_cvar95': risk.max(axis=0),
        'minimax_regret_cvar95': (risk - model_best[:, None]).max(axis=0),
    }
    chosen = {'dijkstra_development_mean': anchor}
    for method, values in scores.items():
        chosen[method] = int(min(feasible, key=lambda j: (values[j], base_mean[j], int(j))))
    return chosen, int(len(feasible)), risk


def clustered_interval(paired, seed, reps=1500):
    """Cluster by OD pair, retaining each pair's observations across shifts."""
    rng = np.random.default_rng(seed)
    clusters = defaultdict(list)
    for r in paired:
        clusters[(r['area'], r['pair'])].append(float(r['delta_cvar95_seconds']))
    means = np.array([np.mean(v) for v in clusters.values()])
    if not len(means):
        return (None, None)
    sampled = means[rng.integers(0, len(means), (reps, len(means)))].mean(axis=1)
    return tuple(float(v) for v in np.quantile(sampled, [.025, .975]))


def run(args):
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Output directory not empty: {out}')
    out.mkdir(parents=True, exist_ok=True)
    records, audits, networks = [], [], []
    places = {'demo': 'offline synthetic demo'} if args.demo else {name: PLACES[name] for name in args.areas.split(',')}
    for area_index, (area, place) in enumerate(places.items()):
        graph, source = load_graph(area, place, args.cache, args.dist, args.demo)
        h = simplify_parallel(graph)
        pairs = od_pairs(h, args.pairs, args.seed + 100 * area_index)
        networks.append(dict(area=area, source=source, nodes=h.number_of_nodes(), edges=h.number_of_edges(), pairs=len(pairs)))
        print(f'{area}: {len(pairs)} OD pairs, {h.number_of_nodes()} nodes', flush=True)
        for pair_index, (s, t) in enumerate(pairs):
            start = time.perf_counter()
            paths = candidates(h, s, t, args.budget)
            generation_seconds = time.perf_counter() - start
            if not paths:
                continue
            # Different, deterministic development seeds per model, pair and area.
            development = {
                shift: scenario_matrix(h, paths, args.dev, args.seed + 100000 * area_index + 1000 * pair_index + 17 * k, shift)
                for k, shift in enumerate(SHIFTS)
            }
            chosen, feasible, _ = select_methods(development, args.epsilon, args.alpha)
            audits.append(dict(area=area, pair=pair_index, source=str(s), target=str(t), candidate_count=len(paths), feasible_count=feasible, selected_unique_routes=len(set(chosen.values())), generation_seconds=generation_seconds, **{f'route_{m}': j for m, j in chosen.items()}))
            for k, shift in enumerate(SHIFTS):
                heldout = scenario_matrix(h, paths, args.test, args.seed + 900000000 + 100000 * area_index + 1000 * pair_index + 17 * k, shift)
                for method in METHODS:
                    j = chosen[method]
                    records.append(dict(area=area, pair=pair_index, shift=shift, method=method, route_id=j, mean_seconds=float(heldout[:, j].mean()), cvar95_seconds=cvar(heldout[:, j], args.alpha), p95_seconds=float(np.quantile(heldout[:, j], args.alpha)), candidate_count=len(paths), feasible_count=feasible))
    write_csv(out/'algorithm_results.csv', records)
    write_csv(out/'candidate_audit.csv', audits)
    indexed = {(r['area'], r['pair'], r['shift'], r['method']): r for r in records}
    paired = []
    for area, pair, shift in sorted({(r['area'], r['pair'], r['shift']) for r in records}):
        base = indexed[area, pair, shift, 'dijkstra_development_mean']
        iid = indexed[area, pair, shift, 'iid_cvar95']
        for method in METHODS[1:]:
            current = indexed[area, pair, shift, method]
            paired.append(dict(area=area, pair=pair, shift=shift, method=method,
                               delta_cvar95_seconds=current['cvar95_seconds'] - base['cvar95_seconds'],
                               delta_mean_seconds=current['mean_seconds'] - base['mean_seconds'],
                               delta_vs_iid_cvar95_seconds=current['cvar95_seconds'] - iid['cvar95_seconds'],
                               changed_from_dijkstra=int(current['route_id'] != base['route_id']),
                               changed_from_iid_cvar=int(current['route_id'] != iid['route_id'])))
    write_csv(out/'paired_results.csv', paired)
    summary = []
    for shift in SHIFTS + ('all_shifts',):
        for method in METHODS[1:]:
            subset = [r for r in paired if r['method'] == method and (r['shift'] == shift or shift == 'all_shifts')]
            if not subset:
                continue
            lo, hi = clustered_interval(subset, args.seed + METHODS.index(method) + (SHIFTS.index(shift) if shift in SHIFTS else 5), args.bootstrap)
            summary.append(dict(shift=shift, method=method, observations=len(subset), od_clusters=len({(r['area'], r['pair']) for r in subset}),
                                mean_delta_cvar95_seconds=float(np.mean([r['delta_cvar95_seconds'] for r in subset])),
                                ci_low=lo, ci_high=hi,
                                mean_delta_mean_seconds=float(np.mean([r['delta_mean_seconds'] for r in subset])),
                                changed_routes=sum(r['changed_from_dijkstra'] for r in subset)))
    write_csv(out/'shift_summary.csv', summary)
    report = dict(phase=12, experiment='candidate-restricted multi-model CVaR robustness benchmark',
                  design='OFFLINE SYNTHETIC DEMO; NOT REAL ROAD DATA' if args.demo else 'OSM road topology; simulated travel-time uncertainty',
                  networks=networks, od_pairs=sum(n['pairs'] for n in networks), method_records=len(records), parameters=vars(args),
                  methods=list(METHODS), uncertainty_models=list(SHIFTS),
                  limitations=[
                      'Multi-model and worst-model CVaR are candidate-restricted heuristic baselines, not distributionally robust optimization over a formal ambiguity set',
                      'All three uncertainty models are specified in development: this is known-model robustness, not robustness to unseen model families',
                      'IID Gamma development means define the expected-cost feasibility constraint for all methods',
                      'OSM road topology is real, but edge uncertainty is simulated; not measured traffic',
                      'Road network is simplified by choosing fastest parallel directed edge; turn restrictions are ignored',
                      'Bootstrap resamples OD pairs, not independent cities; only three areas and repeated observations within each area',
                      'Intervals are exploratory and not adjusted for multiple comparisons',
                      'Candidate budget and model family selected after prior-phase exploration; validate confirmatorily on fresh networks',
                  ])
    (out/'research_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    archive = out.parent / (out.name + '_results.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for file in sorted(out.iterdir()):
            z.write(file, file.name)
    print(f'Phase 12 complete: {report["od_pairs"]} OD pairs, {len(records)} method records; {archive}', flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--demo', action='store_true', help='Offline synthetic demonstration, not real OSM evidence')
    p.add_argument('--areas', default='urban,suburban,constrained')
    p.add_argument('--dist', type=int, default=1200)
    p.add_argument('--pairs', type=int, default=20)
    p.add_argument('--budget', type=int, default=12)
    p.add_argument('--dev', type=int, default=1200)
    p.add_argument('--test', type=int, default=3000)
    p.add_argument('--epsilon', type=float, default=.10)
    p.add_argument('--alpha', type=float, default=.95)
    p.add_argument('--seed', type=int, default=1201)
    p.add_argument('--bootstrap', type=int, default=1500)
    p.add_argument('--cache', default='data/real_networks')
    p.add_argument('--output', default='data/results/phase12_full')
    args = p.parse_args()
    if not (0 < args.alpha < 1 and args.epsilon >= 0 and args.dev >= 20 and args.test >= 20 and args.budget >= 1 and args.pairs >= 1 and args.bootstrap >= 100 and args.dist >= 100):
        p.error('Invalid experiment parameters')
    if not args.demo and (not args.areas or any(x not in PLACES for x in args.areas.split(','))):
        p.error(f'Areas must be selected from: {list(PLACES)}')
    run(args)

if __name__ == '__main__':
    main()
