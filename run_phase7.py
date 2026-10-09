import argparse
from src.benchmark_phase7 import run

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--quick', action='store_true', help='Fast validation on all three topologies')
    p.add_argument('--scenarios', type=int, default=600)
    args = p.parse_args()
    if args.quick:
        run(sizes=(25,), seeds=(1,), distributions=('normal', 'gamma'), budgets=(8, 24), scenarios=200)
    else:
        run(scenarios=args.scenarios)
