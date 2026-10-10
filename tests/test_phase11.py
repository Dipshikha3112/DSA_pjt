import numpy as np
import networkx as nx
from src.phase11_real_networks import demo_graph,simplify_parallel,candidates,scenario_matrix,cvar,select,od_pairs,bootstrap

def test_graph():
 h=simplify_parallel(demo_graph());assert h.number_of_edges()==8

def test_candidates():
 h=simplify_parallel(demo_graph());assert len(candidates(h,'s','t',12))>=2

def test_scenarios_reproducible():
 h=simplify_parallel(demo_graph());p=candidates(h,'s','t');assert np.array_equal(scenario_matrix(h,p,50,5,'iid_gamma'),scenario_matrix(h,p,50,5,'iid_gamma'))

def test_shifts():
 h=simplify_parallel(demo_graph());p=candidates(h,'s','t');assert all(np.all(scenario_matrix(h,p,30,2,x)>0) for x in ['iid_gamma','heavy_lognormal','correlated_gamma'])

def test_cvar():assert cvar([1,2,3,100],.75)==100

def test_selection():
 dev=np.array([[2,5],[2,5],[20,5],[2,5],[2,5],[2,5],[2,5],[2,5],[2,5],[2,5]],dtype=float)
 a,b,n=select(dev,1.0);assert a==0 and b==1 and n==2

def test_od():assert len(od_pairs(simplify_parallel(demo_graph()),2,5))==2

def test_bootstrap():assert len(bootstrap([1,2,3],['a','b','c'],seed=1,reps=20))==2

def test_parallel():
 g=nx.MultiDiGraph();g.add_edge(1,2,travel_time=8);g.add_edge(1,2,travel_time=3)
 assert simplify_parallel(g)[1][2]['base']==3
