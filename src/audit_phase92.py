"""Phase 9.2: post-hoc statistical audit of Phase 9.1 results.

Descriptive and exploratory inference only. Cluster by topology, requested_n, seed.
No retraining, route selection, or access to individual test scenarios.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

KEYS = ['topology', 'requested_n', 'seed', 'distribution', 'budget']
CLUSTER = ['topology', 'requested_n', 'seed']
METRICS = ['mean_cost', 'cvar95', 'variance']
BASELINE = 'cvar_fixed'


def validate(df):
    required = set(KEYS + ['method', 'path_id', 'feasible_count', 'candidate_diversity'] + METRICS)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    if df.empty or df[KEYS + ['method']].isna().any().any():
        raise ValueError('Empty dataset or missing case/method keys')
    if df.duplicated(KEYS + ['method']).any():
        raise ValueError('Duplicate case/method records')
    for col in METRICS:
        if not np.isfinite(pd.to_numeric(df[col], errors='coerce')).all():
            raise ValueError(f'Nonfinite/non-numeric {col}')
    method_sets = df.groupby(KEYS, dropna=False)['method'].agg(lambda x: frozenset(x))
    if method_sets.nunique() != 1:
        raise ValueError('Inconsistent methods across cases')
    methods = set(df.method)
    if BASELINE not in methods or 'dijkstra' not in methods:
        raise ValueError('Missing fixed CVaR or Dijkstra baseline')
    return sorted(methods)


def paired(df, reference=BASELINE):
    if reference not in set(df.method):
        raise ValueError('Unknown reference')
    base = df.loc[df.method == reference, KEYS + METRICS + ['path_id']].rename(
        columns={c: f'ref_{c}' for c in METRICS + ['path_id']})
    out = df.merge(base, on=KEYS, how='left', validate='many_to_one')
    if out[[f'ref_{c}' for c in METRICS]].isna().any().any():
        raise ValueError('Missing reference for one or more cases')
    for c in METRICS:
        out[f'delta_{c}'] = out[c] - out[f'ref_{c}']
    out['route_changed'] = out.path_id != out.ref_path_id
    return out


def graph_clusters(paired_df, metric):
    """Average over dependent distributions/budgets within each graph instance."""
    if metric not in METRICS:
        raise ValueError('Unsupported metric')
    return (paired_df.groupby(CLUSTER + ['method'], as_index=False)
            [f'delta_{metric}'].mean().rename(columns={f'delta_{metric}': 'difference'}))


def signflip_pvalue(x, rng, permutations=20000):
    """Two-sided cluster sign-flip test; exact for <= 15 nonzero clusters."""
    a = np.asarray(x, float)
    a = a[np.abs(a) > 1e-12]
    if not len(a):
        return 1.0
    observed = abs(a.mean())
    if len(a) <= 15:
        bits = ((np.arange(1 << len(a))[:, None] >> np.arange(len(a))) & 1)
        null = abs(((2 * bits - 1) * a).mean(axis=1))
        return float(np.mean(null >= observed - 1e-12))
    extreme = 0
    for _ in range(permutations):
        signs = rng.choice((-1, 1), size=len(a))
        extreme += abs(np.mean(signs * a)) >= observed - 1e-12
    return (extreme + 1) / (permutations + 1)


def cluster_bootstrap_ci(x, rng, iterations=4000):
    a = np.asarray(x, float)
    if not len(a):
        raise ValueError('Empty clusters')
    means = np.empty(iterations)
    for i in range(iterations):
        means[i] = a[rng.integers(0, len(a), len(a))].mean()
    return tuple(map(float, np.quantile(means, [.025, .975])))


def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    if np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError('Invalid p values')
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    running = 0.
    for rank, idx in enumerate(order):
        running = max(running, min(1., p[idx] * (len(p) - rank)))
        adjusted[idx] = running
    return adjusted


def inference(df, reference=BASELINE, seed=9201, iterations=4000, permutations=20000):
    validate(df)
    comparisons = paired(df, reference)
    rng = np.random.default_rng(seed)
    rows = []
    for metric in METRICS:
        clusters = graph_clusters(comparisons, metric)
        for method, g in clusters.groupby('method', sort=True):
            if method == reference:
                continue
            values = g.difference.to_numpy(float)
            low, high = cluster_bootstrap_ci(values, rng, iterations)
            p = signflip_pvalue(values, rng, permutations)
            rows.append(dict(reference=reference, method=method, metric=metric,
                             n_graph_clusters=len(values), mean_cluster_difference=float(values.mean()),
                             ci95_low=low, ci95_high=high, p_unadjusted=p))
    out = pd.DataFrame(rows)
    # Adjust within each metric across all alternative methods.
    out['p_holm'] = np.nan
    for metric, idx in out.groupby('metric').groups.items():
        out.loc[idx, 'p_holm'] = holm(out.loc[idx, 'p_unadjusted'].to_numpy())
    return out


def diagnostics(df):
    validate(df)
    p = paired(df, 'dijkstra')
    rows = []
    for (topology, method), g in p.groupby(['topology', 'method']):
        feas = pd.to_numeric(g.feasible_count, errors='coerce')
        rows.append(dict(topology=topology, method=method, cases=len(g),
                         mean_cost=float(g.mean_cost.mean()), cvar95=float(g.cvar95.mean()),
                         mean_delta_cost=float(g.delta_mean_cost.mean()),
                         mean_delta_cvar95=float(g.delta_cvar95.mean()),
                         route_change_rate=float(g.route_changed.mean()),
                         mean_feasible=float(feas.mean()) if feas.notna().any() else np.nan,
                         one_feasible_rate=float((feas == 1).mean()) if feas.notna().any() else np.nan,
                         mean_candidate_diversity=float(g.candidate_diversity.mean())))
    return pd.DataFrame(rows)


def method_audit(df):
    """Structural, not code-level, audit: flag design limitations from CSV."""
    methods = validate(df)
    clusters = df[CLUSTER].drop_duplicates()
    budgets = sorted(df.budget.unique().tolist())
    results = {
        'records': int(len(df)), 'cases': int(df[KEYS].drop_duplicates().shape[0]),
        'graph_clusters': int(len(clusters)), 'methods': methods,
        'candidate_budgets': budgets,
        'same_graph_repeated_across_distributions_or_budgets': bool(
            df[KEYS].drop_duplicates().groupby(CLUSTER).size().max() > 1),
        'scenario_level_costs_available_in_csv': False,
        'method_runtime_isolated': False,
        'cvar_fixed_is_mean_plus_half_cvar_training_objective': True,
        'adaptive_budget_selection_uses_calibration_halves': True,
        'adaptive_budget_score_uses_fixed_mean_penalty_weight_2': True,
        'generator_comparison_has_different_candidate_subsets': True,
        'interpretation': 'Exploratory, not preregistered; CSV cannot independently verify split isolation or fair tuning.'
    }
    return results


def run(input_csv='data/results/phase91/algorithm_results.csv', output='data/results/phase92',
        iterations=4000, permutations=20000, seed=9201):
    source = Path(input_csv)
    if not source.is_file():
        raise FileNotFoundError(f'Phase 9.1 CSV not found: {source}')
    if iterations < 100 or permutations < 100:
        raise ValueError('Need >=100 bootstrap and permutation draws')
    df = pd.read_csv(source)
    validate(df)
    dest = Path(output); dest.mkdir(parents=True, exist_ok=True)
    paired(df, BASELINE).to_csv(dest/'paired_vs_fixed_cvar.csv', index=False)
    paired(df, 'dijkstra').to_csv(dest/'paired_vs_dijkstra.csv', index=False)
    diagnostics(df).to_csv(dest/'topology_diagnostics.csv', index=False)
    result = inference(df, BASELINE, seed, iterations, permutations)
    result.to_csv(dest/'cluster_inference.csv', index=False)
    audit = method_audit(df)
    audit.update(input_csv=str(source), bootstrap_iterations=iterations,
                 signflip_permutations=permutations, rng_seed=seed,
                 familywise_correction='Holm separately within each metric')
    (dest/'design_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    # Explicit notes avoid misinterpreting significance or claims of novelty.
    report = ['# Phase 9.2: Statistical validation and audit', '',
              'Exploratory, post-hoc analysis. Lower mean cost and CVaR95 are preferred.',
              'Differences are method minus fixed-CVaR, averaged per independent graph cluster',
              '(topology, requested_n, seed), with dependent budget/distribution cells averaged first.',
              f'Graph clusters: {audit["graph_clusters"]}; cases: {audit["cases"]}.', '',
              '## CVaR95 comparisons', '',
              '```\n' + result[result.metric=='cvar95'].round(5).to_string(index=False) + '\n```', '',
              '## Caveats', '',
              '- Graph seeds are independent only as modeled; topologies and graph sizes are fixed strata.',
              '- CVaR95 is estimated from finite test scenarios; 800 total gives only 160 test scenarios.',
              '- A two-sided sign-flip test assumes symmetric cluster-difference null distributions.',
              '- Bootstrap percentile intervals and Holm-adjusted sign-flip p-values are different procedures; they may disagree.',
              '- Averages over graph clusters are equally weighted, not weighted by scenario count.',
              '- All algorithm choices were influenced by earlier exploratory phases; do not claim confirmatory significance.',
              '- The CSV does not contain scenario-level data, so route leakage and paired scenario alignment cannot be independently verified.',
              '- Original and diverse generators use different candidate subsets; attribution requires careful ablation.',
              '- Fixed-CVaR here is the implemented mean + 0.5*CVaR training objective, not necessarily a pure CVaR optimizer.',
              '- The Phase 9.1 runtime field repeats whole-case runtime for every method; no per-method speed comparison.',
              '- Candidate diversity is structural; it does not guarantee near-optimal expected cost.', '']
    (dest/'statistical_report.md').write_text('\n'.join(report), encoding='utf-8')
    print(f'PHASE 9.2 COMPLETED: {len(df)} records; {audit["graph_clusters"]} graph clusters')
    print('Outputs:', dest)
    print(result[result.metric=='cvar95'][['method','mean_cluster_difference','ci95_low','ci95_high','p_holm']].round(4).to_string(index=False))
    return result


def main():
    parser=argparse.ArgumentParser(description='Phase 9.2 statistical audit')
    parser.add_argument('--input',default='data/results/phase91/algorithm_results.csv')
    parser.add_argument('--output',default='data/results/phase92')
    parser.add_argument('--bootstrap',type=int,default=4000)
    parser.add_argument('--permutations',type=int,default=20000)
    parser.add_argument('--seed',type=int,default=9201)
    args=parser.parse_args()
    run(args.input,args.output,args.bootstrap,args.permutations,args.seed)

if __name__=='__main__':
    main()
