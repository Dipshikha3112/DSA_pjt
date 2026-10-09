"""Phase 8: cluster-aware paired comparisons of Phase 7 aggregate results."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

KEYS = ['topology','actual_n','seed','distribution','candidate_budget']
CLUSTER = ['topology','actual_n','seed']
METRICS = ['mean_cost','cvar95','variance']
REQUIRED = KEYS + ['algorithm','risk_lambda','path'] + METRICS


def load_results(path):
    df = pd.read_csv(path)
    missing = sorted(set(REQUIRED) - set(df.columns))
    if missing:
        raise ValueError(f'Phase 7 CSV missing required columns: {missing}')
    if df.empty:
        raise ValueError('Empty benchmark CSV')
    for metric in METRICS:
        df[metric] = pd.to_numeric(df[metric], errors='raise')
        if not np.isfinite(df[metric].to_numpy()).all():
            raise ValueError(f'Nonfinite values in {metric}')
    if (df[METRICS] < 0).any().any():
        raise ValueError('Cost/variance metrics must be nonnegative')
    df['method'] = df['algorithm'].astype(str)
    mv = df['algorithm'].eq('mean_variance')
    if df.loc[mv, 'risk_lambda'].isna().any():
        raise ValueError('Mean-variance rows require risk_lambda')
    df.loc[mv, 'method'] = df.loc[mv, 'risk_lambda'].map(lambda x: f'mean_variance_lambda_{float(x):g}')
    if df.duplicated(KEYS + ['method']).any():
        raise ValueError('Duplicate experiment/algorithm records')
    if not df['method'].eq('expected_dijkstra').any():
        raise ValueError('Missing expected_dijkstra baseline')
    return df


def paired_cases(df):
    base = df[df.method == 'expected_dijkstra'].set_index(KEYS)
    output = []
    for method in sorted(set(df.method) - {'expected_dijkstra'}):
        alt = df[df.method == method].set_index(KEYS)
        joined = base[METRICS + ['path']].join(alt[METRICS + ['path']], how='inner', lsuffix='_base', rsuffix='_alt')
        if len(joined) != len(base) or len(joined) != len(alt):
            raise ValueError(f'Unmatched cases for {method}: baseline={len(base)}, alternative={len(alt)}, matched={len(joined)}')
        for metric in METRICS:
            temp = joined.reset_index()[KEYS].copy()
            temp['method'] = method
            temp['metric'] = metric
            temp['baseline_value'] = joined[f'{metric}_base'].to_numpy()
            temp['alternative_value'] = joined[f'{metric}_alt'].to_numpy()
            temp['difference_alt_minus_base'] = temp['alternative_value'] - temp['baseline_value']
            temp['same_path'] = (joined['path_base'].to_numpy() == joined['path_alt'].to_numpy())
            output.append(temp)
    return pd.concat(output, ignore_index=True)


def cluster_differences(cases):
    # Average repeated distribution and candidate-budget conditions *within* graph-seed clusters.
    return (cases.groupby(CLUSTER + ['method','metric'], as_index=False)
            .agg(difference_alt_minus_base=('difference_alt_minus_base','mean'),
                 baseline_value=('baseline_value','mean'),
                 alternative_value=('alternative_value','mean'),
                 same_path_rate=('same_path','mean'),
                 conditions=('difference_alt_minus_base','size')))


def bootstrap_ci(values, rng, n_boot=3000):
    x = np.asarray(values, dtype=float)
    if len(x) < 2:
        return np.nan, np.nan
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = x[rng.integers(0, len(x), size=len(x))].mean()
    return tuple(np.quantile(means, [0.025, 0.975]))


def signflip_p(values, rng, permutations=10000):
    """Two-sided paired sign-flip randomization p-value; cluster-level exchangeability assumed."""
    x = np.asarray(values, dtype=float)
    x = x[np.abs(x) > 1e-12]
    if len(x) == 0:
        return 1.0
    observed = abs(x.mean())
    count = 0
    for _ in range(permutations):
        signs = rng.choice([-1.0, 1.0], size=len(x))
        count += abs((x * signs).mean()) >= observed - 1e-12
    return (count + 1) / (permutations + 1)


def holm_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    out = np.empty(len(p))
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, p[idx] * (len(p) - rank)))
        out[idx] = running
    return out


def summarize(clusters, n_boot=3000, permutations=10000, seed=2026):
    if n_boot < 100 or permutations < 100:
        raise ValueError('At least 100 bootstrap replicates and permutations required')
    rng = np.random.default_rng(seed)
    rows = []
    for scope, group in [('overall', clusters)] + [(f'topology:{name}', x) for name,x in clusters.groupby('topology')]:
        for (method, metric), subset in group.groupby(['method','metric']):
            values = subset['difference_alt_minus_base'].to_numpy()
            low, high = bootstrap_ci(values, rng, n_boot)
            p = signflip_p(values, rng, permutations)
            denom = subset['baseline_value'].mean()
            rows.append(dict(scope=scope, method=method, metric=metric, n_independent_graph_clusters=len(values),
                             n_repeated_conditions=int(subset.conditions.sum()),
                             mean_difference=float(values.mean()), ci95_low=low, ci95_high=high,
                             baseline_mean=float(denom), alternative_mean=float(subset.alternative_value.mean()),
                             relative_difference_pct=float(100*values.mean()/denom) if denom else np.nan,
                             paired_cohens_dz=float(values.mean()/values.std(ddof=1)) if len(values)>1 and values.std(ddof=1)>1e-12 else np.nan,
                             wins=int((values < -1e-9).sum()), losses=int((values > 1e-9).sum()), ties=int((abs(values)<=1e-9).sum()),
                             same_path_rate=float(subset.same_path_rate.mean()), p_raw=p))
    result = pd.DataFrame(rows)
    # Family: all comparisons within one scope and metric (predeclared).
    result['p_holm'] = np.nan
    for _, ix in result.groupby(['scope','metric']).groups.items():
        result.loc[ix,'p_holm'] = holm_adjust(result.loc[ix,'p_raw'].to_numpy())
    result['significant_at_0_05'] = result.p_holm < .05
    return result


def plot_results(summary, output):
    output.mkdir(parents=True, exist_ok=True)
    for metric in METRICS:
        s = summary[(summary.scope=='overall') & (summary.metric==metric)].sort_values('mean_difference')
        fig, ax = plt.subplots(figsize=(10, max(3.5, .55*len(s)+1.3)))
        y = np.arange(len(s))
        ax.errorbar(s.mean_difference, y, xerr=[s.mean_difference-s.ci95_low, s.ci95_high-s.mean_difference], fmt='o', capsize=3)
        ax.axvline(0, color='gray', linestyle='--', linewidth=1)
        ax.set_yticks(y, s.method)
        ax.set_xlabel(f'{metric}: alternative minus Dijkstra (lower is better)')
        ax.set_title('Graph-cluster bootstrap 95% intervals (descriptive)')
        fig.tight_layout()
        fig.savefig(output / f'paired_{metric}_ci.png', dpi=180)
        plt.close(fig)


def run(input_path, output_dir, plot_dir, n_boot=3000, permutations=10000, seed=2026):
    df = load_results(input_path)
    cases = paired_cases(df)
    clusters = cluster_differences(cases)
    summary = summarize(clusters, n_boot=n_boot, permutations=permutations, seed=seed)
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    cases.to_csv(output_dir/'paired_case_differences.csv', index=False)
    clusters.to_csv(output_dir/'graph_cluster_differences.csv', index=False)
    summary.to_csv(output_dir/'statistical_summary.csv', index=False)
    summary[summary.scope=='overall'].to_csv(output_dir/'overall_comparisons.csv', index=False)
    summary[summary.scope!='overall'].to_csv(output_dir/'topology_comparisons.csv', index=False)
    audit = dict(source=str(input_path), phase7_records=len(df), distinct_cases=int(df[KEYS].drop_duplicates().shape[0]),
                 graph_clusters=int(df[CLUSTER].drop_duplicates().shape[0]), methods=sorted(df.method.unique().tolist()),
                 n_boot=n_boot, permutations=permutations, random_seed=seed,
                 inference_unit='graph topology + node count + seed; conditions averaged within cluster',
                 note='CVaR intervals are across graph clusters, NOT scenario-level CVaR confidence intervals. Graph clusters may share seed values and are not guaranteed independent. Sign-flip p-values are exploratory.',
                 runtime_note='Phase 7 total_case_seconds is case-wide, repeated across algorithms; not per-algorithm runtime.' )
    (output_dir/'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    plot_results(summary, Path(plot_dir))
    best = summary[(summary.scope=='overall') & (summary.metric=='cvar95')][['method','mean_difference','ci95_low','ci95_high','p_holm']]
    lines = ['# Phase 8 Statistical Report', '', f'Phase 7 records: {len(df)}; distinct cases: {audit["distinct_cases"]}; graph clusters: {audit["graph_clusters"]}.', '',
             'Comparisons are paired against expected-weight Dijkstra. Negative differences favor the alternative.',
             'Repeated candidate budgets and uncertainty distributions are averaged within topology × graph size × seed clusters.',
             '95% percentile bootstrap intervals resample graph clusters. Sign-flip tests are exploratory and Holm-adjusted within scope × metric.',
             'Cluster independence and sign-symmetry are assumptions, not established facts; avoid definitive significance claims.',
             'Only aggregate Phase 7 CSVs are available: scenario-level bootstrap intervals, tail dependence and distribution-shift claims require raw held-out scenario costs.',
             'Same paths can occur across methods; interpret ties and differences accordingly.', '', '## Overall CVaR95 comparisons', '',
             best.to_string(index=False), '', '## Limitations', '',
             '- The Phase 7 minimax algorithm is candidate-restricted, not a globally optimal robust solver.',
             '- Repeated scenarios across candidate budgets and possibly related random seeds limit effective independence.',
             '- These are synthetic networks and synthetic cost distributions.',
             '- Phase 5–6 outputs are not pooled because experimental designs differ.',
             '- Statistical significance does not establish practical importance or novel algorithmic contribution.', '']
    (output_dir/'statistical_report.md').write_text('\n'.join(lines), encoding='utf-8')
    return audit, summary
