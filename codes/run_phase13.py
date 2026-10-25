"""Phase 13 runner with optional figure creation and existing-result packaging."""
import argparse
import sys
import zipfile
from pathlib import Path


def make_figures(result_dir, area, pair, rho, shift, samples):
    from plot_phase13 import main as plot_main
    old = sys.argv[:]
    try:
        sys.argv = ['plot_phase13.py', '--results', str(result_dir), '--area', area,
                    '--pair', str(pair), '--rho', str(rho), '--shift', shift,
                    '--samples', str(samples)]
        plot_main()
    finally:
        sys.argv = old


def pack(result_dir):
    folder = Path(result_dir)
    if not (folder / 'research_report.json').is_file():
        raise FileNotFoundError(f'No completed research_report.json in {folder}')
    dest = folder.parent / (folder.name + '_results.zip')
    with zipfile.ZipFile(dest, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(folder.rglob('*')):
            if path.is_file():
                z.write(path, path.relative_to(folder))
    print('Updated results ZIP:', dest)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--visualize-only', action='store_true')
    parser.add_argument('--auto-figures', action='store_true')
    parser.add_argument('--all-figures', action='store_true', help='Generate pair-0 figures for all completed real-network areas')
    parser.add_argument('--figure-area', default='pune')
    parser.add_argument('--figure-pair', type=int, default=0)
    parser.add_argument('--figure-rho', type=float, default=.65)
    parser.add_argument('--figure-shift', default='correlated_gamma')
    parser.add_argument('--figure-samples', type=int, default=10000)
    extra, original_args = parser.parse_known_args()
    output_parser = argparse.ArgumentParser(add_help=False)
    output_parser.add_argument('--output', default='data/results/phase13_full')
    parsed, _ = output_parser.parse_known_args(original_args)
    if not extra.visualize_only:
        from src.phase13_validation import main as benchmark_main
        previous = sys.argv[:]
        try:
            sys.argv = [previous[0]] + original_args
            benchmark_main()
        finally:
            sys.argv = previous
    if extra.auto_figures or extra.visualize_only or extra.all_figures:
        try:
            if extra.all_figures:
                import json
                from src.phase13_validation import FRESH_AREAS
                report = json.loads((Path(parsed.output) / 'research_report.json').read_text(encoding='utf-8'))
                for network in report['networks']:
                    if network['area'] in FRESH_AREAS:
                        make_figures(parsed.output, network['area'], 0,
                                     extra.figure_rho, extra.figure_shift, extra.figure_samples)
            else:
                make_figures(parsed.output, extra.figure_area, extra.figure_pair,
                             extra.figure_rho, extra.figure_shift, extra.figure_samples)
        except Exception as exc:
            if extra.visualize_only or extra.all_figures:
                raise
            print(f'Figures could not be generated: {exc}; benchmark results remain saved.')
        pack(parsed.output)


if __name__ == '__main__':
    main()
