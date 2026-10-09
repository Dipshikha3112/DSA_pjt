import numpy as np
import pandas as pd
import pytest
from src.statistical_analysis_phase8 import (load_results, paired_cases, cluster_differences,
    bootstrap_ci, signflip_p, holm_adjust, summarize, run)


def fixture_csv(tmp_path):
    rows=[]
    for topology in ['grid','scale_free']:
        for n in [25,49]:
            for seed in [1,2,3]:
                for dist in ['normal','gamma']:
                    for budget in [8,24]:
                        key=dict(topology=topology,actual_n=n,seed=seed,distribution=dist,candidate_budget=budget)
                        for algo, lam, delta in [('expected_dijkstra',np.nan,0),('robust_minimax',np.nan,-2),('mean_variance',0.5,1)]:
                            rows.append(dict(**key,algorithm=algo,risk_lambda=lam,path='0->1' if delta==0 else '0->2',
                                             mean_cost=40+delta, cvar95=60+delta, variance=10+delta))
    path=tmp_path/'benchmark_results.csv'
    pd.DataFrame(rows).to_csv(path,index=False)
    return path


def test_load_and_methods(tmp_path):
    d=load_results(fixture_csv(tmp_path))
    assert len(d)==144
    assert 'mean_variance_lambda_0.5' in d.method.unique()


def test_pairing_and_clustering(tmp_path):
    d=load_results(fixture_csv(tmp_path)); cases=paired_cases(d); clusters=cluster_differences(cases)
    assert len(cases)==96*3
    assert len(clusters)==12*2*3
    assert (clusters[clusters.method=='robust_minimax'].difference_alt_minus_base==-2).all()


def test_summary_direction(tmp_path):
    clusters=cluster_differences(paired_cases(load_results(fixture_csv(tmp_path))))
    s=summarize(clusters, n_boot=100, permutations=100, seed=1)
    r=s[(s.scope=='overall')&(s.method=='robust_minimax')&(s.metric=='cvar95')].iloc[0]
    assert r.mean_difference==-2 and r.wins==12 and r.losses==0
    assert r.n_independent_graph_clusters==12


def test_holm():
    assert np.allclose(holm_adjust([.01,.03,.04]), [.03,.06,.06])


def test_bootstrap_reproducible():
    a=bootstrap_ci([1,2,3],np.random.default_rng(7),100)
    b=bootstrap_ci([1,2,3],np.random.default_rng(7),100)
    assert a==b


def test_signflip_ties():
    assert signflip_p([0,0,0],np.random.default_rng(1),100)==1


def test_duplicate_rejected(tmp_path):
    p=fixture_csv(tmp_path); d=pd.read_csv(p); pd.concat([d,d.iloc[:1]]).to_csv(p,index=False)
    with pytest.raises(ValueError,match='Duplicate'): load_results(p)


def test_missing_rejected(tmp_path):
    p=fixture_csv(tmp_path); d=pd.read_csv(p).drop(columns='cvar95'); d.to_csv(p,index=False)
    with pytest.raises(ValueError,match='missing'): load_results(p)


def test_unmatched_rejected(tmp_path):
    p=fixture_csv(tmp_path); d=pd.read_csv(p); d=d.drop(d[d.algorithm=='robust_minimax'].index[0]); d.to_csv(p,index=False)
    with pytest.raises(ValueError,match='Unmatched'): paired_cases(load_results(p))


def test_full_integration(tmp_path):
    p=fixture_csv(tmp_path)
    audit,s=run(p,tmp_path/'results',tmp_path/'plots',n_boot=100,permutations=100)
    assert audit['graph_clusters']==12
    assert (tmp_path/'results'/'statistical_report.md').exists()
    assert (tmp_path/'plots'/'paired_cvar95_ci.png').exists()
    assert len(s)==3*2*3
