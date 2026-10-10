import argparse
import numpy as np
import pytest
from src.phase11_real_networks import simplify_parallel, demo_graph
from src.phase13_validation import uncertain_costs, nested_development, paired_ci, run
from src.phase12_robust import select_methods, METHODS, SHIFTS


def setup():
    h = simplify_parallel(demo_graph())
    return h, [('s', 'a', 't'), ('s', 'b', 't')]


def test_shape_and_positive():
    h, paths = setup()
    for model in SHIFTS:
        x = uncertain_costs(h, paths, 200, 1, model)
        assert x.shape == (200, 2) and np.all(x > 0)


def test_reproducibility():
    h, paths = setup()
    assert np.array_equal(uncertain_costs(h, paths, 30, 1, 'iid_gamma'), uncertain_costs(h, paths, 30, 1, 'iid_gamma'))


def test_nested_prefix():
    h, paths = setup()
    x = nested_development(h, paths, 500, 10, .65)
    y = nested_development(h, paths, 500, 10, .65)
    assert np.array_equal(x['heavy_lognormal'][:250], y['heavy_lognormal'][:250])


def test_invalid_rho():
    h, paths = setup()
    with pytest.raises(ValueError):
        uncertain_costs(h, paths, 20, 1, 'correlated_gamma', 1.2)


def test_shared_draws_for_same_path():
    h, paths = setup()
    x = uncertain_costs(h, [paths[0], paths[0]], 40, 1, 'correlated_gamma')
    assert np.array_equal(x[:, 0], x[:, 1])


def test_select_all_methods():
    h, paths = setup()
    d = nested_development(h, paths, 200, 10, .65)
    selected, feasible, _ = select_methods(d)
    assert set(selected) == set(METHODS) and feasible >= 1


def test_ci():
    rows = [{'area': a, 'pair': p, 'x': v} for a, v in [('a', -1), ('b', 1)] for p in range(3)]
    lo, hi = paired_ci(rows, 'x', 4, 100)
    assert lo <= hi


def test_demo_end_to_end(tmp_path):
    args = argparse.Namespace(demo=True, areas='pune', dist=1200, pairs=2, budget=4,
                              dev_sizes='20,40', test=120, rhos='0.0,0.65', epsilon=.1,
                              alpha=.95, seed=11, bootstrap=100, cache=str(tmp_path/'cache'),
                              output=str(tmp_path/'run'))
    report = run(args)
    assert report['method_records'] == report['pairs'] * 2 * 2 * 3 * 5
    assert (tmp_path/'run_results.zip').exists()
    assert (tmp_path/'run'/'route_stability.csv').exists()
