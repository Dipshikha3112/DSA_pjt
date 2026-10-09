import numpy as np
import pandas as pd
import pytest
from src.algorithms_phase5 import (split_scenarios, candidates_from_weights,
                                   choose_routes, compare_family, run_phase5)

@pytest.fixture
def tiny():
    edges = pd.DataFrame({'source':[0,1,0,2], 'target':[1,3,2,3],
                          'mean_cost':[1.,1.,2.,2.]})
    samples = np.array([[1,1,2,2],[1.2,1.1,2,2],[2,2,1,1],
                        [1,1,3,3],[2,2,1,1],[1,1,2,2]], dtype=float)
    return edges, samples

def test_split_reproducible(tiny):
    _, samples = tiny
    a,b = split_scenarios(samples, .5, 5)
    c,d = split_scenarios(samples, .5, 5)
    assert np.array_equal(a,c) and np.array_equal(b,d)
    assert set(a).isdisjoint(b) and len(a)+len(b)==len(samples)

def test_split_bad(tiny):
    with pytest.raises(ValueError): split_scenarios(tiny[1], 1)

def test_candidate_paths(tiny):
    edges,_ = tiny
    paths = candidates_from_weights(edges, 0, 3, [edges.mean_cost.to_numpy()], k=3)
    assert (0,1,3) in paths and (0,2,3) in paths

def test_four_routes(tiny):
    edges, samples = tiny
    routes, count = choose_routes(edges,samples,0,3,k=3)
    assert len(routes)==4 and count>=2
    assert all(path[0]==0 and path[-1]==3 for path,_ in routes.values())

def test_heldout_regret(tiny):
    edges,samples=tiny
    table,scenarios,split=compare_family(edges,samples,0,3,k=3)
    assert len(table)==4
    assert (scenarios.regret>=-1e-9).all()
    assert set(split['train_indices']).isdisjoint(split['test_indices'])
    assert (table.test_size==2).all()

def test_invalid_lambda(tiny):
    with pytest.raises(ValueError): choose_routes(tiny[0],tiny[1],0,3,risk_lambda=-1)

def test_full_integration(tmp_path):
    root=tmp_path
    edges,samples=tiny_data()
    d=root/'data'/'synthetic'; d.mkdir(parents=True)
    edges.to_csv(d/'toy_edges.csv',index=False)
    (d/'toy_metadata.json').write_text('{"source":0,"target":3}')
    p=root/'data'/'results'/'phase3';p.mkdir(parents=True)
    np.savez_compressed(p/'toy_normal_samples.npz', samples=samples, edge_ids=edges.edge_id.to_numpy())
    result=run_phase5(dataset='toy',families=('normal',),project_root=root,k=3)
    assert len(result)==4
    assert (root/'data'/'results'/'phase5'/'toy_algorithm_comparison.csv').exists()
    assert (root/'outputs'/'plots'/'phase5'/'toy_mean_cost.png').exists()

def tiny_data():
    edges=pd.DataFrame({'edge_id':[1,2,3,4], 'source':[0,1,0,2], 'target':[1,3,2,3],
                        'mean_cost':[1.,1.,2.,2.], 'std_cost':[.2,.2,.4,.4], 'distribution':['normal']*4})
    rng=np.random.default_rng(5)
    samples=np.maximum(.001,rng.normal(edges.mean_cost.to_numpy(),edges.std_cost.to_numpy(),size=(30,4)))
    return edges,samples
