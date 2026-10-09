import numpy as np
import pandas as pd
import pytest
from src.audit_phase92 import (validate, paired, graph_clusters, signflip_pvalue,
                               cluster_bootstrap_ci, holm, inference, diagnostics, method_audit, run)


def fixture():
    rows=[]
    for seed in (101,102,103):
        for budget in (8,24):
            for method, extra in [('dijkstra',0.),('cvar_fixed',-.2),('adaptive_diverse',-.1)]:
                rows.append(dict(topology='grid',requested_n=25,seed=seed,distribution='gamma',
                                 budget=budget,method=method,path_id=int(extra*10),
                                 feasible_count=1. if method=='adaptive_diverse' else np.nan,
                                 candidate_diversity=.5,mean_cost=10+extra,
                                 cvar95=20+extra,variance=3+extra))
    return pd.DataFrame(rows)


def test_validate():
    assert len(validate(fixture())) == 3


def test_missing_column():
    with pytest.raises(ValueError): validate(fixture().drop(columns='cvar95'))


def test_duplicates():
    d=fixture(); with_duplicate=pd.concat([d,d.iloc[:1]],ignore_index=True)
    with pytest.raises(ValueError): validate(with_duplicate)


def test_missing_method():
    d=fixture(); d=d[~((d.seed==101)&(d.budget==8)&(d.method=='dijkstra'))]
    with pytest.raises(ValueError): validate(d)


def test_paired():
    p=paired(fixture()); a=p[p.method=='adaptive_diverse']
    assert np.allclose(a.delta_cvar95,.1)


def test_cluster_repeated_budget():
    c=graph_clusters(paired(fixture()),'cvar95')
    assert len(c)==9


def test_signflip_ties():
    assert signflip_pvalue(np.zeros(4),np.random.default_rng(1))==1.


def test_signflip_exact():
    p=signflip_pvalue(np.ones(4),np.random.default_rng(1))
    assert p==.125


def test_bootstrap_constant():
    assert cluster_bootstrap_ci(np.ones(3),np.random.default_rng(1),100)==(1.,1.)


def test_holm():
    assert np.allclose(holm([.01,.03,.02]),[.03,.04,.04])


def test_inference_reproducible():
    a=inference(fixture(),iterations=100,permutations=100,seed=9)
    b=inference(fixture(),iterations=100,permutations=100,seed=9)
    pd.testing.assert_frame_equal(a,b)


def test_diagnostics():
    assert len(diagnostics(fixture()))==3


def test_audit():
    assert method_audit(fixture())['graph_clusters']==3


def test_run(tmp_path):
    inp=tmp_path/'in.csv'; fixture().to_csv(inp,index=False)
    result=run(inp,tmp_path/'out',iterations=100,permutations=100)
    assert len(result)==6
    assert (tmp_path/'out'/'statistical_report.md').exists()
    assert (tmp_path/'out'/'design_audit.json').exists()


def test_bad_draws(tmp_path):
    inp=tmp_path/'in.csv'; fixture().to_csv(inp,index=False)
    with pytest.raises(ValueError): run(inp,tmp_path/'out',iterations=20)
