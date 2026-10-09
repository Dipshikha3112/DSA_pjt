import numpy as np
import pytest
from src.benchmark_phase7 import (graph_for, edge_table, sample_scenarios,
                                  candidates, pareto_indices, metrics, evaluate_case, run)

@pytest.mark.parametrize('topology', ['grid', 'erdos_renyi', 'scale_free'])
def test_graph_connected(topology):
    g = graph_for(topology, 25, 1)
    import networkx as nx
    assert nx.is_strongly_connected(g)
    assert g.number_of_nodes() == 25

@pytest.mark.parametrize('distribution', ['normal', 'lognormal', 'gamma', 'uniform'])
def test_samples(distribution):
    x = sample_scenarios(np.array([10., 20.]), np.array([2., 4.]), distribution, 300, np.random.default_rng(5))
    assert x.shape == (300, 2)
    assert np.all(x >= 0)


def test_pareto():
    assert pareto_indices(np.array([1., 2., 3.]), np.array([4., 3., 5.])) == [0, 1]


def test_metrics():
    m = metrics(np.arange(1., 101.))
    assert m['cvar95'] >= m['var95']


def test_invalid_budget():
    g = graph_for('grid', 25, 1)
    e, m, _ = edge_table(g, .2)
    with pytest.raises(ValueError):
        candidates(g, e, m, np.tile(m, (10, 1)), 0, 24, 0, 1)


def test_case_reproducible():
    a, f = evaluate_case('grid', 25, 1, 'gamma', 12, scenarios=100)
    b, _ = evaluate_case('grid', 25, 1, 'gamma', 12, scenarios=100)
    assert [r['path'] for r in a] == [r['path'] for r in b]
    assert np.allclose([r['mean_cost'] for r in a], [r['mean_cost'] for r in b])
    assert len(f) >= 1


def test_minimax_training_objective():
    rows, _ = evaluate_case('grid', 25, 1, 'normal', 12, scenarios=100)
    mm = next(r for r in rows if r['algorithm'] == 'robust_minimax')
    assert mm['train_worst'] <= min(r['train_worst'] for r in rows) + 1e-8


def test_integration(tmp_path):
    df = run(output_root=tmp_path/'results', plot_root=tmp_path/'plots',
             sizes=(25,), seeds=(1,), distributions=('normal',), budgets=(8,), scenarios=60)
    assert len(df) == 21
    assert (tmp_path/'results'/'benchmark_results.csv').exists()
    assert (tmp_path/'plots'/'cvar95.png').exists()
