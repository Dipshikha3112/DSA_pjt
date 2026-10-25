import argparse
from pathlib import Path
from src.statistical_analysis_phase8 import run

p = argparse.ArgumentParser(description='Phase 8 paired statistical analysis')
p.add_argument('--input', default='data/results/phase7/benchmark_results.csv')
p.add_argument('--output', default='data/results/phase8')
p.add_argument('--plots', default='outputs/plots/phase8')
p.add_argument('--bootstrap', type=int, default=3000)
p.add_argument('--permutations', type=int, default=10000)
p.add_argument('--seed', type=int, default=2026)
a = p.parse_args()
if not Path(a.input).exists():
    p.error(f'Missing input CSV: {a.input}. Run Phase 7 first.')
audit, summary = run(a.input, a.output, a.plots, a.bootstrap, a.permutations, a.seed)
print(f'PHASE 8 COMPLETED: {audit["phase7_records"]} records, {audit["distinct_cases"]} cases, {audit["graph_clusters"]} graph clusters')
print('\nOverall CVaR95 paired differences (alternative minus Dijkstra):')
print(summary[(summary.scope=='overall') & (summary.metric=='cvar95')][['method','mean_difference','ci95_low','ci95_high','p_holm']].round(4).to_string(index=False))
print(f'Outputs: {a.output} and {a.plots}')
