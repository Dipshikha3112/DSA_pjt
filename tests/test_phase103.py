import numpy as np
import networkx as nx
import pytest
from src.phase103_risk_benchmark import (tail_mean, heterogeneous_scenarios,
    path_cost_matrix, choose, pool, evaluate_case)


def test_tail_mean():
    assert tail_mean(np.arange(1,101)) == pytest.approx(98.0)


def test_tail_bad():
    with pytest.raises(ValueError): tail_mean([np.nan])


def test_scenarios_reproducible():
    a=heterogeneous_scenarios([8.,10.],[True,False],120,42)
    b=heterogeneous_scenarios([8.,10.],[True,False],120,42)
    assert np.array_equal(a,b) and (a>0).all()


def test_scenarios_invalid():
    with pytest.raises(ValueError): heterogeneous_scenarios([8],[True],30,1)


def test_path_costs():
    x=np.array([[1.,2.],[3.,4.]])
    assert np.array_equal(path_cost_matrix(x,[(0,1,2)],{(0,1):0,(1,2):1}).ravel(),[3.,7.])


def test_selection_prefers_tail_with_slack():
    x=np.column_stack([np.where(np.arange(100)<10,20.,8.),np.full(100,10.)])
    j,n,_,_=choose(x,.2)
    assert j==1 and n==2


def test_selection_invalid():
    with pytest.raises(ValueError): choose(np.ones((100,2)),-.1)


def test_no_heldout_in_signature():
    import inspect
    from src.phase103_risk_benchmark import pool
    assert 'test' not in inspect.signature(pool).parameters


def test_quick_case():
    a,b=evaluate_case('grid',25,701,8,'iid_gamma',120,120,.1)
    assert len(a)==4 and len(b)==2
    assert all(row['candidate_count']<=8 for row in a)
    assert a[0]['route']==a[2]['route']
