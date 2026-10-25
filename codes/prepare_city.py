"""Color geographic maps of routes selected by Phase 19.

Install:
    python -m pip install networkx pandas matplotlib contextily pyproj shapely

Run (after run_phase_17.py finishes):
    python generate_route_maps.py --graphs-dir data/real_networks \
        --results data/results/phase17_19_full/all_method_results.csv \
        --output data/results/phase17_19_full/route_maps

Uses CartoDB Voyager street tiles (OpenStreetMap contributors), not a monochrome
network diagram. Internet is required for uncached map tiles.
"""
from __future__ import annotations
import argparse
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import pandas as pd
from pyproj import Transformer

GRAPH_FILES = {
    'pune': 'phase13_pune.graphml',
    'hyderabad': 'phase13_hyderabad.graphml',
    'chennai': 'phase13_chennai.graphml',
    'jaipur': 'phase13_jaipur.graphml',
    'kolkata': 'phase13_kolkata.graphml',
}
COLORS = {'dijkstra':'#2166f3', 'ru_cvar95':'#13a365', 'bs_gamma_robust':'#e84c45'}
NAMES = {'dijkstra':'Dijkstra · shortest expected time',
         'ru_cvar95':'CVaR95 · tail-risk optimized',
         'bs_gamma_robust':'Bertsimas–Sim · budgeted robust'}
TO_MERCATOR = Transformer.from_crs('EPSG:4326', 'EPSG:3857', always_xy=True)


def read_graph(path):
    graph = nx.read_graphml(path)
    coords = {}
    for node, attrs in graph.nodes(data=True):
        try:
            x = float(attrs.get('x', attrs.get('lon')))
            y = float(attrs.get('y', attrs.get('lat')))
            if not (math.isfinite(x) and math.isfinite(y) and -180 <= x <= 180 and -90 <= y <= 90):
                continue
            coords[str(node)] = TO_MERCATOR.transform(x, y)
        except (TypeError, ValueError):
            continue
    if not coords:
        raise ValueError(f'No valid lon/lat node coordinates in {path}. Inspect node attributes.')
    return coords


def route_coords(route, coords, city, od):
    nodes = str(route).split('->')
    missing = [n for n in nodes if n not in coords]
    if missing:
        raise ValueError(f'{city} OD {od}: {len(missing)} route nodes missing from GraphML, examples {missing[:3]}')
    return np.array([coords[n] for n in nodes], dtype=float)


def bounds_with_padding(all_xy, pad=0.15):
    xy = np.concatenate(all_xy, axis=0)
    x0, y0 = xy.min(axis=0)
    x1, y1 = xy.max(axis=0)
    dx, dy = max(x1-x0, 500), max(y1-y0, 500)
    xmid, ymid = (x0+x1)/2, (y0+y1)/2
    # Keep a usable landscape viewport while avoiding large empty margins.
    span_x, span_y = dx*(1+2*pad), dy*(1+2*pad)
    span_x = max(span_x, span_y*1.35)
    span_y = max(span_y, span_x/1.8)
    return (xmid-span_x/2, xmid+span_x/2, ymid-span_y/2, ymid+span_y/2)


def basemap(ax, ctx, bounds, zoom, strict):
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    try:
        ctx.add_basemap(
            ax,
            source=ctx.providers.Esri.WorldStreetMap,
            zoom=zoom,
            attribution_size=6,
        )
        return True
    except Exception as exc:
        msg = f'Unable to fetch colorful street tiles: {exc}'
        if strict:
            raise RuntimeError(msg) from exc
        print('WARNING:', msg, file=sys.stderr)
        ax.set_facecolor('#eaf1f4')
        ax.text(0.5, 0.05, 'Map tiles unavailable — geographic routes only',
                transform=ax.transAxes, ha='center', va='bottom', fontsize=9)
        return False


def plot_one(city, od, group, coords, output, ctx, zoom, dpi, strict):
    by_method = {str(row['method']): row for _, row in group.iterrows()}
    required = list(COLORS)
    if any(m not in by_method for m in required):
        print(f'SKIP {city} OD {od}: one or more selected methods missing')
        return False

    xy = {m: route_coords(by_method[m]['chosen_route'], coords, city, od) for m in required}
    bounds = bounds_with_padding(list(xy.values()))

    fig = plt.figure(figsize=(13.5, 9.5), facecolor='white')
    ax = fig.add_axes((0.035, 0.16, 0.93, 0.76))
    tile_ok = basemap(ax, ctx, bounds, zoom, strict)

    # Outline strokes keep colored routes legible over a busy city basemap.
    # Draw the nominal path first so optimized alternatives remain visible.
    for m in required:
        pts = xy[m]
        ax.plot(pts[:, 0], pts[:, 1], color='white', lw=7.2,
                alpha=0.95, solid_capstyle='round', zorder=7)
        ax.plot(pts[:, 0], pts[:, 1], color=COLORS[m], lw=4.4,
                alpha=0.93, solid_capstyle='round', zorder=8)

    origin, target = xy['dijkstra'][0], xy['dijkstra'][-1]
    ax.scatter(*origin, s=230, c='#15ad6c', edgecolors='white', linewidths=2.5, zorder=12)
    ax.scatter(*target, s=230, c='#d82936', edgecolors='white', linewidths=2.5, zorder=12)
    ax.annotate(' START', origin, xytext=(9, 10), textcoords='offset points',
                color='#065f46', weight='bold', fontsize=10, zorder=13)
    ax.annotate(' END', target, xytext=(9, 10), textcoords='offset points',
                color='#9f1239', weight='bold', fontsize=10, zorder=13)
    ax.set_xlim(bounds[0], bounds[1]); ax.set_ylim(bounds[2], bounds[3])
    ax.set_aspect('equal', adjustable='box')
    ax.set_axis_off()

    first = next(iter(by_method.values()))
    title = f"{city.title()}  |  OD {int(od)+1:02d}  |  Selected road routes"
    fig.suptitle(title, fontsize=19, weight='bold', x=0.04, ha='left', y=0.975)
    subtitle = (f"Uncertainty: {first['shift']}     "
                f"Candidate generator: {first['generator']}     "
                f"Budget: {int(first['budget'])}")
    fig.text(0.04, 0.925, subtitle, fontsize=10.5, color='#475569')

    legend = []
    for m in required:
        row = by_method[m]
        legend.append(Line2D([0], [0], color=COLORS[m], lw=4.4,
            label=f"{NAMES[m]}   |   Mean {float(row['test_mean']):.1f}s   |   CVaR95 {float(row['test_cvar95']):.1f}s"))
    fig.legend(handles=legend, loc='lower left', bbox_to_anchor=(0.04, 0.037),
               frameon=False, fontsize=10.3, ncol=1, labelspacing=0.65)
    fig.text(0.965, 0.026,
        'Map tiles © OpenStreetMap contributors © CARTO' if tile_ok else 'Geographic coordinates from Phase 13 GraphML',
        fontsize=8, color='#64748b', ha='right')
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, facecolor='white')
    plt.close(fig)
    return True


def plot_city_overview(city, rows, coords, output, ctx, zoom, dpi, strict):
    routes = []
    for _, row in rows.iterrows():
        routes.append((row, route_coords(row['chosen_route'], coords, city, row['od'])))
    if not routes:
        return False
    bounds = bounds_with_padding([xy for _, xy in routes], pad=0.2)
    fig = plt.figure(figsize=(14, 10), facecolor='white')
    ax = fig.add_axes((0.03, 0.12, 0.94, 0.77))
    tile_ok = basemap(ax, ctx, bounds, zoom, strict)
    for row, xy in routes:
        color = COLORS[str(row['method'])]
        ax.plot(xy[:, 0], xy[:, 1], lw=2.4, color='white', alpha=0.8, zorder=6)
        ax.plot(xy[:, 0], xy[:, 1], lw=1.7, color=color, alpha=0.83, zorder=7)
        if str(row['method']) == 'dijkstra':
            ax.scatter(*xy[0], s=30, color='#15ad6c', edgecolors='white', lw=0.5, zorder=9)
            ax.scatter(*xy[-1], s=30, color='#d82936', edgecolors='white', lw=0.5, zorder=9)
    ax.set_xlim(bounds[0], bounds[1]); ax.set_ylim(bounds[2], bounds[3]); ax.set_axis_off()
    fig.suptitle(f'{city.title()} | Evaluated route alternatives', x=0.04, y=0.96,
                 ha='left', fontsize=20, weight='bold')
    legend = [Line2D([0], [0], color=COLORS[m], lw=4, label=NAMES[m]) for m in COLORS]
    fig.legend(handles=legend, loc='lower left', bbox_to_anchor=(0.04, 0.035),
               ncol=3, frameon=False, fontsize=10)
    fig.text(0.965, 0.025,
             'Map tiles © OpenStreetMap contributors © CARTO' if tile_ok else 'Geographic coordinates from Phase 13 GraphML',
             fontsize=8, color='#64748b', ha='right')
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, facecolor='white')
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description='Render colorful geographical route maps from Phase 19 results')
    parser.add_argument('--graphs-dir', default='data/real_networks')
    parser.add_argument('--results', default='data/results/phase17_19_full/all_method_results.csv')
    parser.add_argument('--output', default='data/results/phase17_19_full/route_maps')
    parser.add_argument('--shift', default='correlated_gamma')
    parser.add_argument('--generator', default='risk_informed')
    parser.add_argument('--budget', type=int, default=10)
    parser.add_argument('--zoom', default='auto', help='Tile zoom level, e.g. 12, 13, or auto')
    parser.add_argument('--dpi', type=int, default=180)
    parser.add_argument('--max-od-per-city', type=int, default=0, help='0 means all OD pairs')
    parser.add_argument('--strict-tiles', action='store_true', help='Fail instead of producing a no-tiles fallback')
    parser.add_argument('--no-overviews', action='store_true')
    args = parser.parse_args()

    try:
        import contextily as ctx
    except ImportError as exc:
        raise SystemExit('Missing contextily. Run: python -m pip install contextily pyproj shapely') from exc

    result_path = Path(args.results)
    if not result_path.exists():
        raise SystemExit(f'Missing Phase 19 results: {result_path}. Run run_phase_17.py first.')
    df = pd.read_csv(result_path, dtype={'source': str, 'target': str, 'chosen_route': str})
    required = {'city','od','shift','generator','budget','method','chosen_route','test_mean','test_cvar95'}
    if not required.issubset(df.columns):
        raise SystemExit(f'Results CSV missing columns: {sorted(required-set(df.columns))}')

    selected = df.loc[
        (df['shift'].astype(str) == args.shift) &
        (df['generator'].astype(str) == args.generator) &
        (df['budget'] == args.budget) &
        (df['method'].isin(COLORS))
    ].copy()
    if selected.empty:
        available = df[['shift','generator','budget']].drop_duplicates().head(15)
        raise SystemExit('No matching results. Available configurations:\n'+available.to_string(index=False))

    zoom = 'auto' if args.zoom == 'auto' else int(args.zoom)
    output = Path(args.output)
    made = 0
    for city, city_rows in selected.groupby('city', sort=True):
        if city not in GRAPH_FILES:
            print(f'SKIP unknown city {city}')
            continue
        graph_path = Path(args.graphs_dir) / GRAPH_FILES[city]
        if not graph_path.exists():
            raise SystemExit(f'Missing network: {graph_path}')
        print(f'Loading geographic coordinates: {graph_path}', flush=True)
        coords = read_graph(graph_path)
        ods = sorted(city_rows['od'].unique())
        if args.max_od_per_city > 0:
            ods = ods[:args.max_od_per_city]
        overview_rows = []
        for od in ods:
            subset = city_rows.loc[city_rows['od'] == od]
            if subset['method'].duplicated().any():
                raise ValueError(f'Duplicate method rows for {city} OD {od}. Filter your results CSV.')
            dest = output / city / f'od_{int(od)+1:02d}_routes.png'
            if plot_one(city, od, subset, coords, dest, ctx, zoom, args.dpi, args.strict_tiles):
                print(f'  Saved {dest}', flush=True)
                made += 1
                overview_rows.append(subset)
        if not args.no_overviews and overview_rows:
            subset = pd.concat(overview_rows, ignore_index=True)
            dest = output / city / f'{city}_overview.png'
            plot_city_overview(city, subset, coords, dest, ctx, zoom, args.dpi, args.strict_tiles)
            print(f'  Saved {dest}', flush=True)
    print(f'Completed: {made} individual maps. Output: {output.resolve()}')


if __name__ == '__main__':
    main()
