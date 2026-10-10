import numpy as np
import pytest
from src.benchmark_phase7 import graph_for, edge_table
from src.phase103_risk_benchmark import heterogeneous_scenarios
from src.phase104_fairness import make_candidates, paired_ci, evaluate_graph, run


def test_candidate_generation_reproducible():
    g=graph_for('grid',25,901)
    edges,nominal,_=edge_table(g,.4)
    dev=heterogeneous_scenarios(nominal,np.zeros(len(nominal),bool),120,4)
    a,ia=make_candidates(g,edges,dev,dev.mean(axis=0),8,10,'conventional')
    b,ib=make_candidates(g,edges,dev,dev.mean(axis=0),8,10,'conventional')
    assert a==b and len(a)<=8 and ia['attempts']<=64


def test_candidate_modes():
    g=graph_for('grid',25,901)
    edges,nominal,_=edge_table(g,.4)
    dev=heterogeneous_scenarios(nominal,np.zeros(len(nominal),bool),120,4)
    for mode in ('conventional','risk_informed'):
        paths,info=make_candidates(g,edges,dev,dev.mean(axis=0),4,9,mode,8)
        assert 1<=len(paths)<=4 and info['unique_count']==len(paths)


def test_invalid_mode():
    with pytest.raises(ValueError): make_candidates(None,[],np.zeros((1,1)),[1],3,1,'unknown')


def test_bootstrap_zero():
    m,ci=paired_ci(np.zeros(10))
    assert m==0 and ci==[0.,0.]


def test_bootstrap_sign():
    m,ci=paired_ci(np.full(10,-2.))
    assert m==-2 and ci==[-2.,-2.]


def test_bootstrap_requires_clusters():
    with pytest.raises(ValueError): paired_ci([1.])


def test_graph_rows_paired():
    rows,audit=evaluate_graph('grid',25,901,(4,),('iid_gamma',),120,120,.1,2)
    assert len(rows)==4 and len(audit)==2
    assert rows[0]['test_cvar95']==rows[2]['test_cvar95']
    assert all(a['unique_attempts']<=8 for a in audit)


def test_quick_outputs(tmp_path):
    out=tmp_path/'quick'
    report=run(out,True,120,120,.1,(4,8),2)
    assert report['graph_clusters']==3
    assert report['method_records']==12
    assert (out/'paired_dijkstra_cvar.csv').exists()
    assert (out/'runtime_candidate_audit.csv').exists()


def test_prevent_overwrite(tmp_path):
    p=tmp_path/'used';p.mkdir();(p/'keep').write_text('x')
    with pytest.raises(FileExistsError): run(p,True,120,120,.1,(4,),2)


def test_invalid_budgets(tmp_path):
    with pytest.raises(ValueError): run(tmp_path/'out',True,120,120,.1,(4,4),2)
