"""Compare public methods to original source using synthetic data only.

Usage: python -B tools/check_method_reference.py --original-root ORIGINAL_SOURCE
Only selected Python function bodies are loaded. No original data, GIS files,
sample identities, weights, predictions or result files are accessed.
"""
import __future__
import argparse
import ast
import copy
import functools
import json
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import shapely
import torch
import yaml
from rasterio.transform import from_origin
from scipy.spatial import cKDTree
from sklearn.model_selection import KFold
from torch.nn import functional as F
from torch_geometric.utils import softmax
from relation_kagat.graph import geological_relations
from relation_kagat.model import RelationKAGAT
from relation_kagat.preprocess import clr, idw, transform_features
from relation_kagat.sampling import three_folds, _negative_centers


def source_tree(path):
    return ast.parse(path.read_text(encoding="utf-8-sig"))


def execute(nodes, path, namespace):
    code = compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec",
                   flags=__future__.annotations.compiler_flag)
    exec(code, namespace)


def function(path, name, namespace, class_name=None):
    body = source_tree(path).body
    if class_name:
        body = next(node.body for node in body if isinstance(node, ast.ClassDef) and node.name == class_name)
    node = next(node for node in body if isinstance(node, ast.FunctionDef) and node.name == name)
    execute([node], path, namespace)
    return namespace[name]


def compare(actual, expected, tolerance=2e-6):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=tolerance, equal_nan=True)
    delta = np.abs(np.asarray(actual, float) - np.asarray(expected, float))
    return float(np.nanmax(delta)) if np.isfinite(delta).any() else 0.


def check(root):
    rng = np.random.default_rng(901)
    result = {}
    namespace = dict(np=np, pd=pd, WorkflowError=ValueError, cKDTree=cKDTree)
    workflow = root / "data_process/workflow.py"
    original_clr = function(workflow, "transform_geochemistry", namespace)
    assays = pd.DataFrame([[1., 0., 4.], [-1e38, 4., np.nan], [2., 1., 1.], [np.inf, 3., 9.]])
    expected = original_clr(assays, list(assays), dict(processing=dict(
        geochemistry_invalid_values=[-1e38], geochemistry_transform="clr",
        geochemistry_zero_replacement="half_min_positive")))
    result["CLR_max_abs_error"] = compare(clr(assays, [-1e38], "half_min_positive"), expected)

    original_idw = function(workflow, "idw_grid", namespace)
    xy = rng.uniform(-1000, 7000, (45, 2))
    values = rng.normal(size=len(xy))
    transform = from_origin(0, 8000, 1000, 1000)
    row, col = np.indices((8, 8))
    query = np.c_[(col.ravel() + .5) * 1000, 8000 - (row.ravel() + .5) * 1000]
    expected = original_idw(xy, values, np.ones((8, 8), bool), transform,
                            dict(max_neighbors=12, power=2., search_radius_m=2000., min_neighbors=3,
                                 fallback_search_radius_m=5000., fallback_min_neighbors=3,
                                 fallback_allow_nearest=False), -9999.)
    actual = idw(xy, values, query).reshape(8, 8).astype(np.float32)
    actual[~np.isfinite(actual)] = -9999.
    result["IDW_max_abs_error"] = compare(actual, expected)

    function(root / "model/data_features.py", "encode_categories", namespace)
    original_preprocess = function(root / "model/training_pipeline.py", "prepare_features", namespace)
    names = ["A_clr", "B_clr", "aeromagnetic_F3", "distance_m"]
    raw = rng.normal(size=(24, 4)).astype(np.float32)
    raw[[1, 5], 0] = np.nan
    raw[3, 1], raw[4, 2] = np.nan, np.nan
    raw[16:] *= 500
    frame = pd.DataFrame(raw, columns=names)
    frame["age"] = [0, 1] * 8 + [9] * 8
    features = dict(continuous=names, categorical=["age"], geochemistry=names[:2], aeromagnetic=names[2])
    expected, _ = original_preprocess(frame, raw, names, ["age"], np.arange(16))
    actual, _, _ = transform_features(frame, features, np.arange(16))
    result["fold_preprocessing_max_abs_error"] = compare(actual, expected)

    center_source = root / "model/center_sampling.py"
    original_expand = function(center_source, "expand_three_by_three", namespace)
    function(center_source, "_center_components", namespace)
    original_negative = function(center_source, "select_negative_centers", namespace)
    row, col = np.indices((40, 40))
    cells = pd.DataFrame(dict(row=row.ravel(), col=col.ravel(),
                              x=(col.ravel() + .5) * 1000, y=(row.ravel() + .5) * 1000))
    positive = np.array([r * 40 + c for r in [5, 15, 25, 35] for c in [5, 15, 25, 35]])
    core = np.zeros(len(cells), int)
    core[positive] = 1
    cells["known_positive"] = core
    expected, expected_test, _ = original_negative(cells, core, dict(
        negative_sampling_strategy="spatial_exclusion_random", negative_deposit_buffer_m=3000.,
        minimum_negative_center_spacing_cells=3, negative_seed=137, train_centers=12, test_centers=4))
    actual = _negative_centers(cells, positive, dict(seed=137, exclusion_buffer_m=3000., minimum_negative_spacing_cells=3))
    np.testing.assert_array_equal(np.sort(actual[:12]), expected)
    np.testing.assert_array_equal(np.sort(actual[12:]), expected_test)
    result["negative_center_partitions_equal"] = True

    # Execute the original inline fold-assignment block, not a rewritten formula.
    path = root / "model/tune_spatial_random_models.py"
    main = next(node for node in source_tree(path).body if isinstance(node, ast.FunctionDef) and node.name == "main")
    def target_name(node):
        return node.targets[0].id if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) else None
    start = next(i for i, node in enumerate(main.body) if target_name(node) == "groups")
    stop = next(i for i, node in enumerate(main.body) if target_name(node) == "negative_folds")
    outer = dict(positive_train_centers=positive[:12], negative_train_centers=expected[:12])
    namespace.update(frame=cells, outer=outer, KFold=KFold)
    execute(main.body[start:stop + 1], path, namespace)
    folds = three_folds(cells, outer["positive_train_centers"], outer["negative_train_centers"], 20261010)
    for number, (nt, nv) in enumerate(namespace["negative_folds"]):
        pv = np.sort(namespace["assignments"][number])
        pt = np.setdiff1d(outer["positive_train_centers"], pv)
        train = np.union1d(original_expand(cells, pt), original_expand(cells, outer["negative_train_centers"][nt]))
        valid = np.union1d(original_expand(cells, pv), original_expand(cells, outer["negative_train_centers"][nv]))
        np.testing.assert_array_equal(folds[number]["train"], train)
        np.testing.assert_array_equal(folds[number]["validation"], valid)
    result["three_fold_expanded_indices_equal"] = True

    xy = rng.uniform(-3000, 3000, (80, 2))
    bodies = np.array([shapely.box(-3000, -3000, -1, 3000), shapely.box(1, -3000, 3000, 3000)], object)
    faults = np.array([shapely.LineString([(-4000, 0), (4000, 0)]),
                       shapely.LineString([(0, -4000), (300, 0), (-300, 4000)])], object)
    pairs = cKDTree(xy).query_pairs(1500, output_type="ndarray")
    geology_namespace = dict(np=np, shapely=shapely, functools=functools,
                             geometry=lambda: (xy, bodies, faults))
    path = root / "scripts/geological_edge_rules.py"
    function(path, "node_cache", geology_namespace)
    original_relations = function(path, "relations", geology_namespace)
    expected = original_relations(pairs, "min")
    actual = geological_relations(xy, pairs, dict(faults=faults, igneous_bodies=bodies), {})
    result["geological_relations_max_abs_error"] = compare(actual, expected)
    result["synthetic_undirected_edges"] = len(pairs)

    settings = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/default.yaml").read_text())["model"]
    settings["dropout"] = 0.
    torch.manual_seed(39)
    model = RelationKAGAT(4, [3], settings, [0, 1]).eval()
    x = torch.cat((torch.randn(80, 4), torch.ones(80, 1), torch.tensor([[0., 1., 0.]]).repeat(80, 1)), 1)
    edges = torch.tensor(np.concatenate((pairs, pairs[:, ::-1])).T)
    attrs = torch.tensor(np.tile(np.c_[np.linalg.norm(xy[pairs[:, 0]] - xy[pairs[:, 1]], axis=1) / 1500, actual], (2, 1)), dtype=torch.float32)
    reference = copy.deepcopy(model)
    reference.graph_operator = "gat"
    # Section 2.3 uses one relation for all four attributes. Compare the original
    # joint encoder, attention and message functions without independent channels.
    path = root / "model/relation_ka_gat.py"
    namespace = dict(torch=torch, F=F, softmax=softmax)
    encode = function(path, "_encode_relation", namespace, "RelationKAGATV2")
    attention = function(path, "_aggregation_weights", namespace, "RelationKAGATV2")
    message = function(path, "_encode_message", namespace, "RelationKAGATV2")
    with torch.no_grad():
        continuous = x[:, :4].clone()
        continuous[:, [0, 1]] *= x[:, -2:-1]
        node_input = torch.cat((torch.tanh(continuous / 3), reference.embeddings[0](x[:, 4].long()), x[:, -3:]), -1)
        h = reference.node_second(torch.tanh(reference.node_first(node_input)))
        source, target = edges
        relation = encode(reference, h[source], h[target], attrs)
        alpha = attention(reference, h, source, target, relation)
        gate = reference.edge_selector(relation).sigmoid()
        messages = message(reference, h[source], relation).reshape(-1, reference.heads, reference.hidden // reference.heads)
        weighted = (messages * alpha[:, :, None] * gate[:, :, None]).reshape(-1, reference.hidden)
        aggregate = torch.zeros_like(h).index_add_(0, target, weighted)
        gamma = reference.graph_gate.sigmoid().clamp(max=reference.graph_gate_max)
        updated = reference.output_norm(h + gamma * reference.update(torch.cat((h, aggregate), -1)))
        expected = reference.prediction_head(torch.tanh(updated)).squeeze(-1)
        actual = model(x, edges, attrs)
    result["model_logits_max_abs_error"] = compare(actual.numpy(), expected.numpy())
    result["passed"] = True
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-root", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        print(json.dumps(check(args.original_root.resolve()), indent=2))
