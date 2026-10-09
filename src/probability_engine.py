"""Probability distributions and Phase 3 runner for stochastic shortest paths."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

DISTRIBUTIONS = ('normal', 'lognormal', 'gamma', 'uniform')


def make_distribution(kind: str, mean: float, std: float):
    """Return a frozen scipy distribution with nonnegative support.

    For 'normal', mean/std specify the underlying untruncated normal.
    For all other families, they are the actual distribution moments.
    """
    kind = str(kind).strip().lower()
    mean, std = float(mean), float(std)
    if not np.isfinite(mean) or not np.isfinite(std) or mean <= 0 or std <= 0:
        raise ValueError('mean and std must be finite positive numbers')
    if kind == 'normal':
        return stats.truncnorm(a=-mean / std, b=np.inf, loc=mean, scale=std)
    if kind == 'lognormal':
        shape2 = np.log1p((std / mean) ** 2)
        return stats.lognorm(s=np.sqrt(shape2), scale=np.exp(np.log(mean) - shape2 / 2))
    if kind == 'gamma':
        return stats.gamma(a=(mean / std) ** 2, scale=std ** 2 / mean)
    if kind == 'uniform':
        half_width = np.sqrt(3) * std
        if mean < half_width:
            raise ValueError('uniform lower bound would be negative')
        return stats.uniform(loc=mean - half_width, scale=2 * half_width)
    raise ValueError(f'Unsupported distribution: {kind}')


def load_edges(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Phase 2 dataset missing: {path}. Run python main.py first.')
    frame = pd.read_csv(path)
    required = {'source', 'target', 'mean_cost', 'std_cost'}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f'Missing columns in {path}: {sorted(missing)}')
    if frame.empty:
        raise ValueError('Edge dataset is empty')
    frame = frame.copy()
    for col in ('mean_cost', 'std_cost'):
        frame[col] = pd.to_numeric(frame[col], errors='raise')
        if not np.isfinite(frame[col]).all() or (frame[col] <= 0).any():
            raise ValueError(f'{col} must contain finite positive values')
    if frame[['source', 'target']].isna().any().any():
        raise ValueError('Source and target cannot be missing')
    if frame.duplicated(subset=['source', 'target']).any():
        raise ValueError('Duplicate directed edges found')
    if 'edge_id' not in frame:
        frame['edge_id'] = np.arange(len(frame))
    if frame['edge_id'].isna().any() or frame['edge_id'].duplicated().any():
        raise ValueError('edge_id must be unique and nonmissing')
    return frame.reset_index(drop=True)


def sample_edges(edges: pd.DataFrame, kind: str, n_samples: int = 10000,
                 seed: int = 42) -> tuple[np.ndarray, pd.DataFrame]:
    if not isinstance(n_samples, (int, np.integer)) or n_samples <= 0:
        raise ValueError('n_samples must be a positive integer')
    rng = np.random.default_rng(seed)
    samples = np.empty((n_samples, len(edges)), dtype=np.float64)
    records = []
    for col, edge in enumerate(edges.itertuples(index=False)):
        mean = float(edge.mean_cost)
        std = float(edge.std_cost)
        dist = make_distribution(kind, mean, std)
        draws = dist.rvs(size=n_samples, random_state=rng)
        samples[:, col] = draws
        theoretical_mean, theoretical_var = dist.stats(moments='mv')
        records.append({
            'edge_id': edge.edge_id, 'source': edge.source, 'target': edge.target,
            'distribution': kind, 'input_mean': mean, 'input_std': std,
            'theoretical_mean': float(theoretical_mean),
            'theoretical_variance': float(theoretical_var),
            'sample_mean': float(np.mean(draws)),
            'sample_variance': float(np.var(draws, ddof=0)),
            'sample_min': float(np.min(draws)),
            'sample_max': float(np.max(draws)),
            'sample_p95': float(np.quantile(draws, 0.95)),
        })
    if not np.isfinite(samples).all() or (samples < 0).any():
        raise RuntimeError('Generated invalid stochastic costs')
    return samples, pd.DataFrame.from_records(records)


def plot_distributions(edges: pd.DataFrame, sampled: dict[str, np.ndarray],
                       output_dir: str | Path, dataset: str) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    representative_edge = edges.iloc[0]
    mu, sigma = float(representative_edge.mean_cost), float(representative_edge.std_cost)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, kind in zip(axes.flat, DISTRIBUTIONS):
        vals = sampled[kind][:, 0]
        dist = make_distribution(kind, mu, sigma)
        x = np.linspace(max(0, np.quantile(vals, 0.001) * 0.9), np.quantile(vals, 0.999) * 1.1, 300)
        ax.hist(vals, bins=50, density=True, alpha=0.5, label='Empirical')
        ax.plot(x, dist.pdf(x), linewidth=2, label='Theoretical')
        ax.set(title=kind.title(), xlabel='Edge cost', ylabel='Density')
        ax.legend()
    fig.suptitle(f'{dataset}: distribution comparison (first edge)')
    fig.tight_layout()
    fig.savefig(output_dir / f'{dataset}_distribution_comparison.png', dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for kind in DISTRIBUTIONS:
        vals = np.sort(sampled[kind][:, 0])
        ax.plot(vals, np.arange(1, len(vals) + 1) / len(vals), label=kind.title())
    ax.set(title=f'{dataset}: empirical CDF (first edge)', xlabel='Edge cost', ylabel='Cumulative probability')
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / f'{dataset}_ecdf_comparison.png', dpi=160)
    plt.close(fig)


def run_phase3(dataset: str = 'grid_25', n_samples: int = 10000, seed: int = 42,
               project_root: str | Path | None = None) -> None:
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[1]
    edges = load_edges(root / 'data' / 'synthetic' / f'{dataset}_edges.csv')
    results_dir = root / 'data' / 'results' / 'phase3'
    plots_dir = root / 'outputs' / 'plots' / 'phase3'
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    print(f'Dataset: {dataset}_edges.csv | directed edges: {len(edges)}')
    sampled = {}
    for kind in DISTRIBUTIONS:
        samples, summary = sample_edges(edges, kind, n_samples=n_samples, seed=seed)
        sampled[kind] = samples
        prefix = f'{dataset}_{kind}'
        np.savez_compressed(results_dir / f'{prefix}_samples.npz',
                            samples=samples, edge_ids=edges['edge_id'].to_numpy())
        summary.to_csv(results_dir / f'{prefix}_summary.csv', index=False)
        metadata = {
            'dataset': dataset, 'distribution': kind, 'seed': seed,
            'n_samples': n_samples, 'n_edges': len(edges),
            'matrix_shape': list(samples.shape),
            'column_order': 'Same as edge_ids in NPZ and edge CSV rows',
            'normal_parameter_note': 'Normal mean/std are pre-truncation parameters',
            'independence_assumption': 'Independent directed edge costs',
        }
        (results_dir / f'{prefix}_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        relative_error = np.abs(summary['sample_mean'] - summary['theoretical_mean']) / summary['theoretical_mean']
        print(f'{kind:<10} samples={samples.shape} max_relative_mean_error={relative_error.max():.3%}')
    plot_distributions(edges, sampled, plots_dir, dataset)
    print('PHASE 3 COMPLETED SUCCESSFULLY')


if __name__ == '__main__':
    run_phase3()
