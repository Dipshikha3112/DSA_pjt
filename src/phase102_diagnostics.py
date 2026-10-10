"""Phase 10.2: post-hoc candidate diagnostics and controlled risk-trade-off tests.

Never tune on held-out test data. This is an exploratory diagnostic, not a
confirmatory claim of superiority or a novel algorithm.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.adaptive_routing_phase9 import cvar

KEYS = ['topology', 'requested_n', 'seed', 'shift', 'budget']
REQUIRED = set(KEYS + ['candidate_id', 'selection_mean', 'selection_cvar95',
    'worst_stress_cvar95', 'shift_penalty', 'shift_objective', 'shift_feasible',
    'test_mean', 'test_cvar95'])


def read_audit(path):
    frame = pd.read_csv(path)
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError('Missing candidate audit columns: ' + ', '.join(sorted(missing)))
    if frame.duplicated(KEYS + ['candidate_id']).any():
        raise ValueError('Duplicate candidate within a case')
    if frame[KEYS].isna().any().any():
        raise ValueError('Missing case identifiers')
    return frame


def analyze_candidates(audit):
    """Test labels are used ONLY for retrospective diagnostics, never selection."""
    rows = []
    for key, group in audit.groupby(KEYS, sort=True):
        g = group.sort_values('candidate_id').reset_index(drop=True)
        if not (g.candidate_id == 0).any():
            raise ValueError('Candidate 0 missing')
        d = g[g.candidate_id == 0].iloc[0]
        feasible = g[g.shift_feasible.astype(str).str.lower().isin(('true','1'))]
        if feasible.empty:
            raise ValueError('No feasible candidates')
        alternatives = g[g.candidate_id != 0]
        f_alt = feasible[feasible.candidate_id != 0]
        chosen = feasible.sort_values(['shift_objective','candidate_id']).iloc[0]
        best_dev = feasible.sort_values(['selection_cvar95','candidate_id']).iloc[0]
        best_test = feasible.sort_values(['test_cvar95','candidate_id']).iloc[0]
        min_mean = float(g.selection_mean.min())
        max_eps_needed = float((g.selection_mean / min_mean - 1).max())
        rows.append(dict(zip(KEYS, key), candidate_count=len(g), feasible_count=len(feasible),
            has_feasible_alternative=bool(len(f_alt)),
            has_test_better_alternative=bool((f_alt.test_cvar95 < d.test_cvar95-1e-9).any()),
            has_dev_better_alternative=bool((f_alt.selection_cvar95 < d.selection_cvar95-1e-9).any()),
            chosen_id=int(chosen.candidate_id), best_dev_cvar_id=int(best_dev.candidate_id),
            best_test_cvar_id=int(best_test.candidate_id),
            chosen_test_cvar95=float(chosen.test_cvar95),
            best_feasible_test_cvar95=float(best_test.test_cvar95),
            test_oracle_gap=float(chosen.test_cvar95-best_test.test_cvar95),
            dev_ranking_correct=bool(best_dev.candidate_id==best_test.candidate_id),
            max_candidate_epsilon_needed=max_eps_needed))
    return pd.DataFrame(rows)


def sweep_objectives(audit, epsilons=(0.,.05,.10,.20,.40), lambdas=(0.,.10,.25,.5,1.)):
    """Re-rank stored candidates; held-out costs are diagnostics only.

    This does NOT retrain candidate generation and is NOT a new held-out test.
    It uses precomputed selection CVaR and stress estimates.
    """
    rows = []
    for key, g in audit.groupby(KEYS, sort=True):
        g = g.sort_values('candidate_id')
        mu = g.selection_mean.to_numpy(dtype=float)
        base = g.selection_cvar95.to_numpy(dtype=float)
        penalty = g.shift_penalty.to_numpy(dtype=float)
        if np.any(mu <= 0) or not np.isfinite(mu).all():
            raise ValueError('Invalid means')
        ids = g.candidate_id.to_numpy(dtype=int)
        for eps in epsilons:
            if eps < 0: raise ValueError('epsilon must be nonnegative')
            feasible = mu <= mu.min()*(1+eps)+1e-10
            for lam in lambdas:
                if lam < 0: raise ValueError('lambda must be nonnegative')
                objective = base + lam*penalty
                rank = np.where(feasible,objective,np.inf)
                j = int(np.argmin(rank))
                row = g.iloc[j]
                rows.append(dict(zip(KEYS,key), epsilon=float(eps),lambda_shift=float(lam),
                    feasible_count=int(feasible.sum()), chosen_id=int(ids[j]),
                    selection_mean=float(mu[j]),selection_cvar95=float(base[j]),
                    shift_penalty=float(penalty[j]), objective=float(objective[j]),
                    # Test field is descriptive only; do not optimize eps/lambda using it.
                    diagnostic_test_cvar95=float(row.test_cvar95)))
    return pd.DataFrame(rows)


def pareto_candidates(audit):
    """Non-dominated on development mean and development CVaR (not test)."""
    rows=[]
    for key,g in audit.groupby(KEYS,sort=True):
        mu=g.selection_mean.to_numpy(dtype=float)
        risk=g.selection_cvar95.to_numpy(dtype=float)
        for i,(_,r) in enumerate(g.iterrows()):
            dominated=bool(np.any(((mu <= mu[i]+1e-10) & (risk <= risk[i]+1e-10)) &
                ((mu < mu[i]-1e-10) | (risk < risk[i]-1e-10))))
            rows.append(dict(zip(KEYS,key),candidate_id=int(r.candidate_id),
                selection_mean=float(mu[i]),selection_cvar95=float(risk[i]),
                pareto_efficient=not dominated))
    return pd.DataFrame(rows)


def controlled_tradeoff(train_count=6000, test_count=30000, seed=501, q=.95):
    """A transparent two-route counterexample with exact mean/risk conflict.

    Route A: 9.5 deterministic units (1 edge).
    Route B: 8 with probability .90, 16 with probability .10 (1 edge).
    B has mean 8.8 < 9.5 but CVaR95(B)=16 > 9.5.
    Independent training/test Bernoulli streams, no tuning on test.
    """
    if train_count < 100 or test_count < 100 or not .5 < q < 1:
        raise ValueError('Invalid controlled experiment configuration')
    rng=np.random.default_rng(seed)
    tr=np.column_stack([np.full(train_count,9.5),
        np.where(rng.random(train_count)<.10,16.,8.)])
    te=np.column_stack([np.full(test_count,9.5),
        np.where(rng.random(test_count)<.10,16.,8.)])
    # Both routes are available under a 10% mean-cost slack.
    mean=tr.mean(axis=0)
    cv=np.array([cvar(tr[:,j],q) for j in range(2)])
    methods={'expected_cost':int(np.argmin(mean)),
             'cvar_only':int(np.argmin(cv)),
             'mean_plus_half_cvar':int(np.argmin(mean+.5*cv))}
    results=[]
    for name,j in methods.items():
        results.append(dict(method=name,chosen_route=('A_safe' if j==0 else 'B_risky'),
            train_mean=float(mean[j]),train_cvar95=float(cv[j]),
            test_mean=float(te[:,j].mean()),test_cvar95=float(cvar(te[:,j],q))))
    return pd.DataFrame(results)


def cluster_summary(case_df):
    """Physical graph clusters are not independent across shifts/budgets."""
    cluster_cols=['topology','requested_n','seed']
    return case_df.groupby(cluster_cols,as_index=False).agg(
        cases=('chosen_id','size'),mean_test_oracle_gap=('test_oracle_gap','mean'),
        fraction_feasible_alternatives=('has_feasible_alternative','mean'),
        fraction_dev_better_alternatives=('has_dev_better_alternative','mean'),
        fraction_test_better_alternatives=('has_test_better_alternative','mean'))


def save_plots(sweep, pareto, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Fraction of cases selecting candidate 0 by epsilon and lambda.
    tab=sweep.groupby(['epsilon','lambda_shift']).apply(
        lambda g: float((g.chosen_id==0).mean()),include_groups=False).unstack()
    fig,ax=plt.subplots(figsize=(8,5))
    for lam in tab.columns: ax.plot(tab.index,tab[lam],marker='o',label=f'lambda={lam:g}')
    ax.set(xlabel='Mean-cost slack epsilon',ylabel='Fraction selecting candidate 0',ylim=(-.03,1.03),
           title='Exploratory parameter sensitivity (stored candidates)')
    ax.legend(fontsize=8);fig.tight_layout()
    fig.savefig(out/'candidate0_sensitivity.png',dpi=160);plt.close(fig)
    # An illustrative development Pareto scatter (not held-out).
    keys=pareto[KEYS].drop_duplicates().iloc[0].to_dict()
    g=pareto
    for k,v in keys.items():g=g[g[k]==v]
    fig,ax=plt.subplots(figsize=(7,5))
    ax.scatter(g.selection_mean,g.selection_cvar95,label='All candidates')
    efficient=g[g.pareto_efficient]
    ax.scatter(efficient.selection_mean,efficient.selection_cvar95,marker='x',s=70,label='Non-dominated')
    ax.set(xlabel='Development mean',ylabel='Development CVaR95',title='Example development Pareto frontier')
    ax.legend();fig.tight_layout();fig.savefig(out/'example_pareto.png',dpi=160);plt.close(fig)


def main(argv=None):
    p=argparse.ArgumentParser(description='Phase 10.2 diagnostic analysis; never uses test for route selection')
    p.add_argument('--input',default='data/results/phase101_repeat1/candidate_audit.csv')
    p.add_argument('--output',default='data/results/phase102')
    p.add_argument('--seed',type=int,default=501)
    p.add_argument('--no-plots',action='store_true')
    a=p.parse_args(argv)
    out=Path(a.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Output directory not empty: {out}; choose a new --output')
    audit=read_audit(a.input)
    cases=analyze_candidates(audit)
    sweep=sweep_objectives(audit)
    pareto=pareto_candidates(audit)
    clusters=cluster_summary(cases)
    controlled=controlled_tradeoff(seed=a.seed)
    out.mkdir(parents=True,exist_ok=True)
    for name,frame in [('case_diagnostics',cases),('parameter_sweep',sweep),
                       ('development_pareto',pareto),('graph_cluster_summary',clusters),
                       ('controlled_tradeoff',controlled)]:
        frame.to_csv(out/f'{name}.csv',index=False)
    if not a.no_plots:save_plots(sweep,pareto,out)
    report={'phase':'10.2','design':'retrospective exploratory diagnostic',
            'cases':int(len(cases)),'physical_graph_clusters':int(len(clusters)),
            'candidates':int(len(audit)),
            'cases_one_feasible':int((cases.feasible_count==1).sum()),
            'cases_feasible_alternative_test_better':int(cases.has_test_better_alternative.sum()),
            'cases_feasible_alternative_dev_better':int(cases.has_dev_better_alternative.sum()),
            'cases_dev_and_test_best_agree':int(cases.dev_ranking_correct.sum()),
            'sweep_warning':'Sweep is exploratory; test labels must not be used to tune parameters.',
            'limitations':['Candidate pool fixed from Phase 10.1; not a candidate-generation intervention',
                'Phase 10.1 test outcomes already inspected: cannot treat sweep as confirmatory',
                'Controlled example establishes objective disagreement, not real-world superiority',
                'No global optimality proof or novelty claim']}
    (out/'diagnostic_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(f"Phase 10.2 complete: {len(cases)} cases, {len(audit)} candidates, {len(clusters)} graph clusters")
    print('Outputs:',out)
    print('Cases with one feasible candidate:',report['cases_one_feasible'])
    print('Feasible alternatives better on test (audit only):',report['cases_feasible_alternative_test_better'])
    return report

if __name__=='__main__':main()
