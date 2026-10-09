import numpy as np
import pandas as pd
import pytest

from src.probability_engine import make_distribution, sample_edges, load_edges, run_phase3


def example_edges():
    return pd.DataFrame({'edge_id': [0, 1], 'source': [0, 1], 'target': [1, 2],
                         'mean_cost': [10., 20.], 'std_cost': [2., 4.]})


@pytest.mark.parametrize('kind', ['normal', 'lognormal', 'gamma', 'uniform'])
def test_nonnegative_and_moments(kind):
    dist = make_distribution(kind, 10, 2)
    draws = dist.rvs(size=10000, random_state=np.random.default_rng(123))
    assert np.min(draws) >= 0
    target_mean = float(dist.mean())
    assert abs(np.mean(draws) - target_mean) < 0.15


def test_bad_parameters():
    with pytest.raises(ValueError):
        make_distribution('gamma', 0, 2)


def test_bad_distribution():
    with pytest.raises(ValueError):
        make_distribution('poisson', 10, 2)


def test_bad_uniform_support():
    with pytest.raises(ValueError):
        make_distribution('uniform', 1, 2)


def test_shape():
    samples, summary = sample_edges(example_edges(), 'gamma', 100, 42)
    assert samples.shape == (100, 2)
    assert len(summary) == 2


def test_reproducibility():
    a, _ = sample_edges(example_edges(), 'normal', 50, 42)
    b, _ = sample_edges(example_edges(), 'normal', 50, 42)
    assert np.array_equal(a, b)


def test_load_edges(tmp_path):
    path = tmp_path / 'edges.csv'
    example_edges().to_csv(path, index=False)
    assert len(load_edges(path)) == 2


def test_missing_columns(tmp_path):
    path = tmp_path / 'bad.csv'
    pd.DataFrame({'source': [0]}).to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_edges(path)


def test_full_export(tmp_path):
    folder = tmp_path / 'data' / 'synthetic'
    folder.mkdir(parents=True)
    example_edges().to_csv(folder / 'grid_25_edges.csv', index=False)
    run_phase3(n_samples=100, project_root=tmp_path)
    path = tmp_path / 'data' / 'results' / 'phase3' / 'grid_25_gamma_samples.npz'
    with np.load(path) as data:
        assert data['samples'].shape == (100, 2)
    assert (tmp_path / 'outputs' / 'plots' / 'phase3' / 'grid_25_ecdf_comparison.png').exists()
