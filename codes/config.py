from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data' / 'synthetic'
GRAPH_DIR = ROOT / 'outputs' / 'graphs'
SEED = 42
UNCERTAINTY_RATIO = 0.20
DATASETS = [
    ('grid', 25), ('grid', 100), ('grid', 400),
    ('erdos_renyi', 30), ('erdos_renyi', 100), ('erdos_renyi', 250),
    ('scale_free', 30), ('scale_free', 100), ('scale_free', 250),
]
