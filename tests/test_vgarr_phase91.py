import numpy as np
import pytest
from src.vgarr_phase91 import (split_four,edge_overlap,diversity,candidates_91,
    feasible_indices,select_budget,select_route,run_case)
from src.benchmark_phase7 import graph_for,edge_table

def test_four_way_disjoint():
    (parts,indices)=split_four(np.arange(400).reshape(100,4),11)
    assert list(map(len,parts))==[50,15,15,20]
    assert len(set(sum((list(x) for x in indices),[])))==100

def test_four_way_reproducible():
    a=np.arange(400).reshape(100,4)
    assert all(np.array_equal(x,y) for x,y in zip(split_four(a,3)[0],split_four(a,3)[0]))

def test_four_way_invalid():
    with pytest.raises(ValueError):split_four(np.ones((10,3)),1)

def test_overlap():
    assert edge_overlap((0,1,2),(0,1,2))==1
    assert edge_overlap((0,1,2),(0,3,2))==0

def test_diversity():
    assert diversity([(0,1,2)])==0
    assert diversity([(0,1,2),(0,3,2)])==1

def test_candidate_uniqueness():
    g=graph_for('grid',25,101);edges,means,_=edge_table(g,.4)
    paths=candidates_91(g,edges,means,np.tile(means,(20,1)),0,24,8,7)
    assert len(paths)==len(set(paths)) and 1<=len(paths)<=8

def test_candidate_original_ablation():
    g=graph_for('grid',25,101);edges,means,_=edge_table(g,.4)
    p=candidates_91(g,edges,means,np.tile(means,(20,1)),0,24,8,7,'original')
    assert 1<=len(p)<=8

def test_candidate_invalid():
    g=graph_for('grid',25,101);edges,means,_=edge_table(g,.4)
    with pytest.raises(ValueError):candidates_91(g,edges,means,np.tile(means,(20,1)),0,24,0,7)

def test_feasible():
    a=np.tile([10.,10.3,12.],(20,1))
    assert feasible_indices(a,.05).tolist()==[0,1]

def test_budget_calibration():
    a=np.tile([10.,10.3,12.],(40,1))
    eps,scores=select_budget(a)
    assert eps in (.02,.05,.10,.15) and len(scores)==4

def test_budget_invalid():
    with pytest.raises(ValueError):select_budget(np.ones((3,2)))

def test_route_selection():
    a=np.tile([10.,10.3,12.],(20,1))
    idx,count=select_route(a,.05)
    assert idx in (0,1) and count==2

def test_case_deterministic():
    a=run_case(scenarios=100,budget=8)
    b=run_case(scenarios=100,budget=8)
    assert [r['path'] for r in a]==[r['path'] for r in b]
    assert [r['cvar95'] for r in a]==[r['cvar95'] for r in b]

def test_case_methods():
    a=run_case(topology='scale_free',n=25,seed=101,distribution='lognormal',scenarios=100,budget=8)
    assert len(a)==8
    assert all((r['n_train'],r['n_cal'],r['n_select'],r['n_test'])==(50,15,15,20) for r in a)

def test_bad_case():
    with pytest.raises(ValueError):run_case(scenarios=30)
