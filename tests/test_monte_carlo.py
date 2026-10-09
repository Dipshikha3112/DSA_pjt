import json
import numpy as np
import pandas as pd
import pytest

from src.monte_carlo import (build_network, load_inputs, route_columns, run_phase4,
                             shortest_path, simulate, summarize)


def toy_edges():
    return pd.DataFrame({
        'edge_id': [1, 2, 3], 'source': [0, 1, 0], 'target': [1, 2, 2],
        'mean_cost': [2., 2., 7.], 'std_cost': [0.4, 0.4, 1.4],
        'distribution': ['normal', 'normal', 'normal'],
    })


def test_shortest_path():
    adj, _ = build_network(toy_edges())
    assert shortest_path(adj, np.array([2., 2., 7.]), 0, 2) == (4., (0, 1, 2))


def test_route_columns():
    _, lookup = build_network(toy_edges())
    assert route_columns((0, 1, 2), lookup).tolist() == [0, 1]


def test_scenario_optimum_and_regret():
    samples = np.array([[2., 2., 7.], [5., 5., 3.], [1., 1., 6.]])
    scenarios, paths, result = simulate(toy_edges(), samples, 0, 2)
    assert scenarios['fixed_cost'].tolist() == [4., 10., 2.]
    assert scenarios['oracle_cost'].tolist() == [4., 3., 2.]
    assert scenarios['regret'].tolist() == [0., 7., 0.]
    assert result['fixed_optimal_fraction'] == pytest.approx(2 / 3)
    assert paths['count'].sum() == 3


def test_nonnegative_regret():
    rng = np.random.default_rng(42)
    samples = rng.uniform(0.5, 10, (100, 3))
    scenarios, _, _ = simulate(toy_edges(), samples, 0, 2)
    assert (scenarios['regret'] >= 0).all()
    assert (scenarios['oracle_cost'] <= scenarios['fixed_cost'] + 1e-8).all()


def test_bad_weights():
    adj, _ = build_network(toy_edges())
    with pytest.raises(ValueError):
        shortest_path(adj, np.array([-1., 2., 3.]), 0, 2)


def test_no_route():
    adj, _ = build_network(toy_edges())
    with pytest.raises(ValueError):
        shortest_path(adj, np.array([1., 1., 1.]), 2, 0)


def test_summary():
    result = summarize(np.array([1., 2., 3., 4.]))
    assert result['mean'] == 2.5
    assert result['var95'] == pytest.approx(3.85)
    assert result['cvar95'] == 4.


def test_npz_edge_alignment(tmp_path):
    root = tmp_path
    folder = root / 'data' / 'synthetic'
    folder.mkdir(parents=True)
    toy_edges().to_csv(folder / 'toy_edges.csv', index=False)
    (folder / 'toy_metadata.json').write_text(json.dumps({'source': 0, 'target': 2}))
    out = root / 'data' / 'results' / 'phase3'
    out.mkdir(parents=True)
    np.savez_compressed(out / 'toy_normal_samples.npz', samples=np.ones((3, 3)),
                        edge_ids=np.array([2, 1, 3]))
    with pytest.raises(ValueError, match='Edge IDs'):
        load_inputs(root, 'toy', 'normal')


def test_integration(tmp_path):
    folder = tmp_path / 'data' / 'synthetic'
    folder.mkdir(parents=True)
    toy_edges().to_csv(folder / 'toy_edges.csv', index=False)
    (folder / 'toy_metadata.json').write_text(json.dumps({'source': 0, 'target': 2}))
    out = tmp_path / 'data' / 'results' / 'phase3'
    out.mkdir(parents=True)
    np.savez_compressed(out / 'toy_normal_samples.npz',
                        samples=np.array([[2., 2., 7.], [5., 5., 3.]]),
                        edge_ids=np.array([1, 2, 3]))
    table = run_phase4('toy', families=('normal',), project_root=tmp_path)
    assert len(table) == 1
    assert (tmp_path / 'data/results/phase4/toy_normal_scenarios.csv').exists()
    assert (tmp_path / 'outputs/plots/phase4/toy_normal_convergence.png').exists()
