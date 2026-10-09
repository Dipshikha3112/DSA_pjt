import numpy as np
import pandas as pd
import pytest
from src.sensitivity_phase6 import (scale_uncertainty, candidate_pool, evaluate_setting,
                                    controlled_benchmark, run_phase6)


def toy():
    edges = pd.DataFrame({'edge_id':[1,2,3,4], 'source':[0,1,0,2], 'target':[1,3,2,3],
                          'mean_cost':[2.,2.,3.,3.], 'std_cost':[.4,.4,.6,.6], 'distribution':['normal']*4})
    rng = np.random.default_rng(12)
    samples = np.maximum(0., rng.normal(edges.mean_cost.to_numpy(),
                                        edges.std_cost.to_numpy(), size=(80,4)))
    return edges, samples


def test_uncertainty_zero():
    edges, samples = toy()
    out = scale_uncertainty(samples, edges.mean_cost.to_numpy(), .2, 0.)
    assert np.allclose(out, edges.mean_cost.to_numpy())


def test_uncertainty_bad():
    edges, samples = toy()
    with pytest.raises(ValueError):
        scale_uncertainty(samples, edges.mean_cost.to_numpy(), 0, .4)


def test_candidates():
    edges, samples = toy()
    paths = candidate_pool(edges, samples, 0, 3, k=3)
    assert (0,1,3) in paths and (0,2,3) in paths


def test_split_and_metrics():
    edges, samples = toy()
    frame, split = evaluate_setting(edges, samples, 0, 3, lambdas=(0.,1.), k=3)
    assert len(frame) == 5
    assert len(set(split['train_indices']) & set(split['test_indices'])) == 0
    assert frame.mean_regret.ge(-1e-9).all()
    assert frame.reliability.between(0,1).all()


def test_controlled_switches_route():
    frame = controlled_benchmark()
    assert frame.iloc[0].selected_route == 'A_risky'
    assert frame.iloc[-1].selected_route == 'B_safe'
    assert frame.iloc[-1].cvar95 < frame.iloc[0].cvar95


def test_invalid_lambda():
    edges, samples = toy()
    with pytest.raises(ValueError):
        evaluate_setting(edges, samples, 0, 3, lambdas=(-1,), k=3)


def test_integration(tmp_path):
    edges, samples = toy()
    d = tmp_path/'data'/'synthetic'; d.mkdir(parents=True)
    edges.to_csv(d/'toy_edges.csv', index=False)
    (d/'toy_metadata.json').write_text('{"source":0,"target":3}')
    p = tmp_path/'data'/'results'/'phase3'; p.mkdir(parents=True)
    np.savez_compressed(p/'toy_normal_samples.npz', samples=samples,
                        edge_ids=edges.edge_id.to_numpy())
    frame = run_phase6(dataset='toy', families=('normal',), project_root=tmp_path,
                       lambdas=(0.,1.), uncertainties=(.2,.4), k=3)
    assert len(frame) == 10
    assert (tmp_path/'data'/'results'/'phase6'/'toy_sensitivity.csv').exists()
    assert (tmp_path/'outputs'/'plots'/'phase6'/'controlled_tradeoff.png').exists()
