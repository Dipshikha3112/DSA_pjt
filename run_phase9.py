import argparse
from src.adaptive_routing_phase9 import run_benchmark

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--quick',action='store_true')
    p.add_argument('--scenarios',type=int,default=800)
    a=p.parse_args()
    result=run_benchmark(quick=a.quick,scenarios=a.scenarios)
    print(f'PHASE 9 COMPLETED: {len(result)} algorithm records')
    print('Outputs: data/results/phase9')
