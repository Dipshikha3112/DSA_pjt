import numpy as np
import pytest
from src.adaptive_routing_phase9 import (split_scenarios,cvar,stats,choose_routes,
                                           diverse_candidates,run_case)
from src.benchmark_phase7 import graph_for,edge_table

def test_split_disjoint():
    a=np.arange(300).reshape(100,3)
    t,v,h=split_scenarios(a,42)
    assert len(t)==50 and len(v)==25 and len(h)==25
    assert set(t[:,0]).isdisjoint(v[:,0]) and set(v[:,0]).isdisjoint(h[:,0])

def test_split_reproducible():
    a=np.arange(120).reshape(40,3)
    assert np.array_equal(split_scenarios(a,3)[0],split_scenarios(a,3)[0])

def test_split_invalid():
    with pytest.raises(ValueError):split_scenarios(np.ones((4,3)),2)

def test_cvar_tail():
    assert cvar(np.arange(100.),.95)==96.5

def test_cvar_invalid():
    with pytest.raises(ValueError):cvar([1.])

def test_stats():
    s=stats(np.arange(100.))
    assert s['mean_cost']==49.5 and s['cvar95']==96.5

def test_candidates():
    g=graph_for('grid',25,1)
    edges,means,_=edge_table(g,.4)
    paths=diverse_candidates(g,edges,means,np.tile(means,(30,1)),0,24,8,1)
    assert 1<=len(paths)<=8 and len(paths)==len(set(paths))

def test_choose_no_test_input():
    t=np.array([[1,2,3],[2,2,3],[1,2,4],[2,2,4]],float)
    v=np.array([[1,2,3],[2,2,3],[1,2,4],[2,2,4]],float)
    a,m=choose_routes(t,v,[(0,1),(0,2),(0,3)])
    assert set(a)=={'dijkstra','saa','mean_variance','minimax','cvar_fixed','vgarr'}
    assert m['feasible_count']>=1

def test_choose_invalid():
    with pytest.raises(ValueError):choose_routes(np.ones((4,2)),np.ones((4,3)),[(0,1),(0,2)])

def test_case_reproducible():
    a=run_case(scenarios=80,budget=8)
    b=run_case(scenarios=80,budget=8)
    for x,y in zip(a,b):
        assert x['path']==y['path'] and x['cvar95']==y['cvar95']

def test_case_all_methods():
    a=run_case(topology='erdos_renyi',n=25,seed=2,distribution='lognormal',scenarios=80,budget=8)
    assert len(a)==6 and all(r['n_train']==40 and r['n_valid']==20 and r['n_test']==20 for r in a)

def test_bad_config():
    with pytest.raises(ValueError):run_case(scenarios=10)
