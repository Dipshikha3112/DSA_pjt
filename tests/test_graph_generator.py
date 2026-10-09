import json

import networkx as nx
import pandas as pd
import pytest

from src.graph_generator import create_graph, export_dataset, validate_graph, visualize_graph


@pytest.mark.parametrize('kind,n', [('grid',25), ('erdos_renyi',30), ('scale_free',30)])
def test_graphs(kind,n):
    graph = create_graph(kind,n)
    assert graph.number_of_nodes() == n
    assert nx.is_strongly_connected(graph)
    validate_graph(graph)
    assert all(attrs['mean_cost'] > 0 for _, _, attrs in graph.edges(data=True))


def test_reproducibility():
    a = create_graph('scale_free', 30)
    b = create_graph('scale_free', 30)
    assert list(a.edges(data=True)) == list(b.edges(data=True))


def test_grid_edge_count():
    assert create_graph('grid', 25).number_of_edges() == 80


def test_export_and_plot(tmp_path):
    graph = create_graph('grid',25)
    metadata = export_dataset(graph,'grid',25,tmp_path)
    nodes = pd.read_csv(tmp_path/'grid_25_nodes.csv')
    edges = pd.read_csv(tmp_path/'grid_25_edges.csv')
    assert len(nodes) == 25 and len(edges) == 80
    assert metadata['baseline_expected_cost'] > 0
    assert json.loads((tmp_path/'grid_25_metadata.json').read_text())['source'] == 0
    assert visualize_graph(graph,'grid',25,tmp_path).exists()


def test_invalid_grid_size():
    with pytest.raises(ValueError):
        create_graph('grid',30)
