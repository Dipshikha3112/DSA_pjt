# Phase 4 — Monte Carlo Shortest-Path Engine

## Installation in existing Windows project

Copy only these files (do not replace Phase 1–3 files):

- `run_phase4.py` -> project root
- `src/monte_carlo.py` -> existing `src` directory
- `tests/test_monte_carlo.py` -> existing `tests` directory

Requires Phase 2 `data/synthetic/grid_25_edges.csv`, `grid_25_metadata.json`, and Phase 3 `data/results/phase3/grid_25_{normal,lognormal,gamma,uniform}_samples.npz`.

## Commands (from activated venv and project root)

```cmd
python run_phase4.py
python -m pytest tests/test_monte_carlo.py -v
python -m pytest tests -v
```

## Outputs

- `data/results/phase4/grid_25_<family>_scenarios.csv`: per-scenario fixed cost, oracle cost, regret, optimality, oracle path
- `data/results/phase4/grid_25_<family>_path_frequencies.csv`: scenario-optimal path frequencies
- `data/results/phase4/grid_25_<family>_summary.json`: mean, std, variance, VaR95, worst-5% CVaR95, reliability, regret
- `data/results/phase4/grid_25_comparison.csv`: four-distribution comparison
- `outputs/plots/phase4/grid_25_<family>_{cost_histogram,regret_histogram,convergence}.png`

The NPZ loader accepts both `costs` (the user's Phase 3 fixed package) and `samples` array keys. It validates the NPZ `edge_ids` order against Phase 2 CSV rows before running.

**Interpretation**: the oracle observes all edge weights before selecting a route; it is an offline lower bound, not a real-time deployable policy. The fixed route is selected once using Phase 2 mean_cost values. Reliability uses the illustrative threshold 1.2 × baseline expected cost. Simulation time measures scenario Dijkstra only, excluding loading and plotting.
