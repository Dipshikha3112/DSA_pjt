import argparse
import json
import zipfile
import numpy as np
import pytest
from src.phase12_robust import METHODS, SHIFTS, clustered_interval, run, select_methods


def dev_data():
    rng = np.random.default_rng(8)
    return {shift: np.column_stack([rng.normal(10, 1, 200), rng.normal(12, 1, 200)]) for shift in SHIFTS}


def test_selection_all_methods_and_feasible():
    chosen, feasible, risks = select_methods(dev_data(), epsilon=.25)
    assert set(chosen) == set(METHODS)
    assert feasible == 2
    assert risks.shape == (3, 2)
    assert all(v in (0, 1) for v in chosen.values())


def test_only_one_feasible():
    chosen, feasible, _ = select_methods(dev_data(), epsilon=.01)
    assert feasible == 1
    assert set(chosen.values()) == {0}


def test_worst_model_selects_lower_worst_tail():
    rng = np.random.default_rng(3)
    d = {s: np.column_stack([rng.normal(10, .1, 500), rng.normal(10.3, .1, 500)]) for s in SHIFTS}
    d['heavy_lognormal'][:, 0] += 7
    chosen, _, _ = select_methods(d, epsilon=.1)
    assert chosen['worst_model_cvar95'] == 1
    assert chosen['dijkstra_development_mean'] == 0


def test_reject_missing_models():
    with pytest.raises(ValueError):
        select_methods({'iid_gamma': np.ones((100, 2))})


def test_reject_incompatible_shapes():
    d = dev_data()
    d['iid_gamma'] = np.ones((100, 3))
    with pytest.raises(ValueError):
        select_methods(d)


def test_clustered_ci_constant():
    rows = [dict(area='x', pair=i, delta_cvar95_seconds=2.0) for i in range(5)]
    assert clustered_interval(rows, 5, 100) == (2., 2.)


def test_offline_end_to_end(tmp_path):
    args = argparse.Namespace(demo=True, areas='urban,suburban,constrained', dist=1200,
        pairs=2, budget=4, dev=50, test=70, epsilon=.1, alpha=.95,
        seed=1201, bootstrap=100, cache=str(tmp_path/'cache'), output=str(tmp_path/'demo'))
    report = run(args)
    assert report['method_records'] == report['od_pairs'] * 3 * len(METHODS)
    assert report['od_pairs'] > 0
    zpath = tmp_path/'demo_results.zip'
    assert zpath.exists()
    with zipfile.ZipFile(zpath) as z:
        assert len(z.namelist()) == 5
        assert json.loads(z.read('research_report.json'))['phase'] == 12
    with pytest.raises(FileExistsError):
        run(args)
