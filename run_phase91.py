import argparse
from src.vgarr_phase91 import run_benchmark

if __name__ == '__main__':
    p=argparse.ArgumentParser(description='Phase 9.1 independent risk-budget and diversity ablations')
    p.add_argument('--quick',action='store_true')
    p.add_argument('--scenarios',type=int,default=800)
    p.add_argument('--output',default='data/results/phase91')
    args=p.parse_args()
    df=run_benchmark(output=args.output,quick=args.quick,scenarios=args.scenarios)
    print(f'PHASE 9.1 COMPLETED: {len(df)} algorithm records')
    print('Outputs:',args.output)
