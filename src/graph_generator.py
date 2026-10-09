import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from config import DATA_DIR, GRAPH_DIR, SEED, UNCERTAINTY_RATIO, DATASETS


def create_graph(kind: str, n: int, seed: int = SEED) -> nx.DiGraph:
    if n < 4:
        raise ValueError('n must be at least 4')
    if kind == 'grid':
        side = math.isqrt(n)
        if side * side != n:
            raise ValueError('Grid node count must be a perfect square')
        base = nx.grid_2d_graph(side, side)
        pos = {node: (float(node[1]), float(-node[0])) for node in base}
        base = nx.convert_node_labels_to_integers(base, ordering='sorted', label_attribute='grid_position')
        positions = {i: pos[base.nodes[i]['grid_position']] for i in base.nodes}
    elif kind == 'erdos_renyi':
        # p ~ 4/n gives an average degree near 4, with a connected seed topology.
        p = min(0.95, 4.0 / (n - 1))
        rng = np.random.default_rng(seed)
        for _ in range(200):
            base = nx.gnp_random_graph(n, p, seed=int(rng.integers(0, 2**31 - 1)))
            if nx.is_connected(base):
                break
        else:
            raise RuntimeError('Could not generate a connected random graph')
        positions = nx.spring_layout(base, seed=seed)
    elif kind == 'scale_free':
        base = nx.barabasi_albert_graph(n, m=2, seed=seed)
        positions = nx.spring_layout(base, seed=seed)
    else:
        raise ValueError(f'Unknown topology: {kind}')

    graph = nx.DiGraph()
    graph.add_nodes_from((int(node), {'x': float(positions[node][0]), 'y': float(positions[node][1])}) for node in base.nodes)
    rng = np.random.default_rng(seed + n)
    # Bidirectional links guarantee strong connectivity when the base graph is connected.
    for u, v in sorted(base.edges()):
        base_cost = float(rng.uniform(5.0, 20.0))
        for a, b in ((u, v), (v, u)):
            graph.add_edge(int(a), int(b), base_cost=round(base_cost, 6),
                           mean_cost=round(base_cost, 6),
                           std_cost=round(base_cost * UNCERTAINTY_RATIO, 6),
                           distribution='normal')
    if not nx.is_strongly_connected(graph):
        raise AssertionError('Generated graph must be strongly connected')
    return graph


def validate_graph(graph: nx.DiGraph) -> None:
    if not isinstance(graph, nx.DiGraph) or graph.number_of_nodes() < 2:
        raise ValueError('Expected a nonempty directed graph')
    if not nx.is_strongly_connected(graph):
        raise ValueError('Graph must be strongly connected')
    for u, v, attrs in graph.edges(data=True):
        if attrs['base_cost'] <= 0 or attrs['mean_cost'] <= 0 or attrs['std_cost'] < 0:
            raise ValueError(f'Invalid edge parameters for {u}->{v}')
        if attrs['distribution'] != 'normal':
            raise ValueError('Unexpected distribution label')


def export_dataset(graph: nx.DiGraph, kind: str, n: int, output_dir: Path = DATA_DIR) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f'{kind}_{n}'
    nodes = pd.DataFrame([{'node_id': node, 'x': attrs['x'], 'y': attrs['y']}
                          for node, attrs in sorted(graph.nodes(data=True))])
    edges = pd.DataFrame([{'edge_id': i, 'source': u, 'target': v, **attrs}
                          for i, (u, v, attrs) in enumerate(sorted(graph.edges(data=True)), start=1)])
    nodes.to_csv(output_dir / f'{stem}_nodes.csv', index=False)
    edges.to_csv(output_dir / f'{stem}_edges.csv', index=False)
    source, target = 0, n - 1
    baseline_path = nx.shortest_path(graph, source, target, weight='mean_cost')
    baseline_cost = nx.shortest_path_length(graph, source, target, weight='mean_cost')
    metadata = {'topology': kind, 'nodes': n, 'directed_edges': graph.number_of_edges(),
                'seed': SEED, 'uncertainty_ratio': UNCERTAINTY_RATIO,
                'source': source, 'target': target, 'baseline_path': baseline_path,
                'baseline_expected_cost': float(baseline_cost)}
    (output_dir / f'{stem}_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return metadata


def visualize_graph(graph: nx.DiGraph, kind: str, n: int, output_dir: Path = GRAPH_DIR) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pos = {node: (attrs['x'], attrs['y']) for node, attrs in graph.nodes(data=True)}
    path = nx.shortest_path(graph, 0, n - 1, weight='mean_cost')
    fig, ax = plt.subplots(figsize=(9, 7))
    # Draw undirected representation to avoid overlapping bidirectional arrows.
    nx.draw_networkx_edges(graph.to_undirected(), pos, ax=ax, width=0.6, alpha=0.35)
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=max(12, 2300 / n), node_color='#7db7e8')
    nx.draw_networkx_edges(graph, pos, edgelist=list(zip(path[:-1], path[1:])),
                           edge_color='#d62728', width=2.5, arrows=True, ax=ax)
    nx.draw_networkx_nodes(graph, pos, nodelist=[0, n - 1], node_size=max(65, 3500 / n),
                           node_color=['#2ca02c', '#ff7f0e'], ax=ax)
    if n <= 30:
        nx.draw_networkx_labels(graph, pos, ax=ax, font_size=7)
    ax.set_title(f'{kind.replace("_", " ").title()} | {n} nodes | baseline shortest path (red)')
    ax.axis('off')
    fig.tight_layout()
    path_out = output_dir / f'{kind}_{n}.png'
    fig.savefig(path_out, dpi=160, bbox_inches='tight')
    plt.close(fig)
    return path_out


def run_all() -> None:
    for kind, n in DATASETS:
        graph = create_graph(kind, n)
        validate_graph(graph)
        info = export_dataset(graph, kind, n)
        image_path = visualize_graph(graph, kind, n)
        print(f'{kind:12} n={n:3} edges={info["directed_edges"]:4} '
              f'path_cost={info["baseline_expected_cost"]:.2f} image={image_path.name}')
    print('PHASE 2 DATASETS GENERATED SUCCESSFULLY')


if __name__ == '__main__':
    run_all()
