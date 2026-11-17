"""Synthetic numerical, geometry and leakage regressions for the public method."""
import copy
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
import shapely
import torch
import yaml
from rasterio.transform import from_origin
from relation_kagat.graph import geological_relations, read_geology
from relation_kagat.model import RelationKAGAT, predict_logits
from relation_kagat.preprocess import clr, idw, interpolate_assays, transform_features, read_nodes
from relation_kagat.sampling import center_components, expand_centers, fixed_samples
from relation_kagat.synthetic import make_demo
from relation_kagat import pipeline

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


class MethodTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="geoka_method_test_")
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def config(self):
        return yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))

    def test_clr_partial_missing_sentinels_and_explicit_replacement(self):
        values = np.array([[1., 0., 4.], [-1e38, 4., np.nan], [np.nan, 1., 1.]])
        actual = clr(values, invalid_values=[-1e38], zero_replacement="half_min_positive")
        logs = np.log([1, .5, 4])
        np.testing.assert_allclose(actual[0], logs - logs.mean())
        assert np.isnan(actual[1, 0]) and np.isnan(actual[1, 2]) and actual[1, 1] == 0
        np.testing.assert_allclose(actual[2, 1:], 0)
        np.testing.assert_array_equal(values[0], [1, 0, 4])  # caller input is unchanged
        with self.assertRaises(ValueError):
            clr([[1, 0], [2, 4]])
        with self.assertRaises(ValueError):
            clr([[0, 2], [np.nan, 4]], zero_replacement="half_min_positive")

    def test_idw_primary_support_fallback_and_no_nearest_extrapolation(self):
        points = [[0, 0], [2, 0], [4, 0]]
        result = idw(points, [2, 4, 6], [[0, 0], [1, 0], [50, 0]],
                     max_distance=1.5, fallback_distance=5)
        assert result[0] == 2 and np.isnan(result[-1])
        self.assertAlmostEqual(result[1], (2 + 4 + 6 / 9) / (2 + 1 / 9))
        unsupported = idw(points[:2], [2, 4], [[0, 0]], max_distance=1.5, fallback_distance=5)
        assert np.isnan(unsupported[0])  # exact location still needs three supporting samples

    def test_assays_are_transformed_before_interpolation(self):
        points = [[0, 0], [1000, 0], [2000, 0]]
        assays = [[1, 2], [4, 1], [2, 8]]
        settings = self.config()["preprocessing"]
        expected = clr(assays)
        actual = interpolate_assays(points, assays, [[1000, 0]], settings)
        np.testing.assert_allclose(actual[0], expected[1])

    def test_robust_scaling_fallback_and_original_missing_indicator_layout(self):
        frame = pd.DataFrame(dict(A_clr=[0., 0., 0., 0., 100., np.nan, 9999.],
                                  B_clr=[np.nan] * 6 + [12.],
                                  aeromagnetic_F3=[5., 5., 5., 5., 5., np.nan, 100.],
                                  age=[0, 1, 0, 1, 1, 1, 9]))
        features = dict(continuous=["A_clr", "B_clr", "aeromagnetic_F3"], categorical=["age"],
                        geochemistry=["A_clr", "B_clr"], aeromagnetic="aeromagnetic_F3")
        x, cards, state = transform_features(frame, features, [0, 1, 2, 3, 4, 5])
        self.assertAlmostEqual(state["scale"][0], np.std([0, 0, 0, 0, 100]))
        assert state["center"] == [0, 0, 5] and state["scale"][1:] == [1, 1]
        assert state["all_missing_training_features"] == ["B_clr"]
        assert cards == [3] and x[0, 3] == 1 and x[-1, 3] == 0  # raw category zero is known
        np.testing.assert_array_equal(x[0, -3:], [.5, 0, 0])
        np.testing.assert_array_equal(x[5, -3:], [1, 0, 1])
        original = copy.deepcopy(state)
        changed = frame.copy()
        changed.loc[6, ["A_clr", "B_clr", "aeromagnetic_F3", "age"]] = [1e8, -1e8, 3e8, 123]
        _, _, replay_state = transform_features(changed, features, [0, 1, 2, 3, 4, 5])
        assert replay_state == original

    def test_along_cross_touch_and_angle_rules(self):
        xy = np.array([[0, 200], [1000, 200], [0, -200], [0, 0],
                       [0, 1200], [1000, 1200], [2500, 200], [1000, 1000]], float)
        pairs = np.array([[0, 1], [2, 0], [3, 0], [4, 5], [0, 6], [3, 7]])
        geology = dict(faults=[shapely.LineString([(-10000, 0), (10000, 0)])], igneous_bodies=[])
        result = geological_relations(xy, pairs, geology, {})
        self.assertAlmostEqual(result[0, 1], np.exp(-.2), places=6)
        assert result[1, 2] == 1 and result[1, 1] == 0
        assert result[2, 2] == 0  # endpoint touch
        assert result[3, 1] == 0 and result[4, 1] == 0  # buffer and arc-distance limits
        self.assertAlmostEqual(result[5, 1], .5 * np.exp(-.5), places=6)
        reverse = geological_relations(xy, pairs[:, ::-1], geology, {})
        np.testing.assert_allclose(result, reverse, atol=1e-7)

    def test_same_body_requires_strict_interiors_and_whole_segment(self):
        body = shapely.Polygon([(0, 0), (3000, 0), (3000, 3000), (2000, 3000),
                                (2000, 1000), (1000, 1000), (1000, 3000), (0, 3000)])
        xy = np.array([[500, 2500], [2500, 2500], [500, 100], [2500, 100], [0, 500], [500, 500]])
        pairs = [[0, 1], [2, 3], [4, 5]]
        result = geological_relations(xy, pairs, dict(faults=[], igneous_bodies=[body]), {})
        np.testing.assert_array_equal(result[:, 0], [0, 1, 0])
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            geological_relations(xy, pairs, dict(faults=[], igneous_bodies=[body, body]), {})

    def test_geology_crs_inferred_flags_and_line_parts(self):
        template = self.root / "grid.tif"
        with rasterio.open(template, "w", driver="GTiff", width=2, height=2, count=1,
                           dtype="uint8", crs="EPSG:3857", transform=from_origin(0, 2000, 1000, 1000)) as dst:
            dst.write(np.ones((2, 2), np.uint8), 1)
        document = dict(crs="EPSG:3857", faults=[
            dict(id="mapped", inferred=False, wkt="MULTILINESTRING ((0 0, 1000 0), (0 0, 0 1000))"),
            dict(id="inferred", inferred=True, wkt="LINESTRING (0 0, 1000 1000)")], igneous_bodies=[])
        path = self.root / "geology.local.yaml"
        def write():
            path.write_text(yaml.safe_dump(document), encoding="utf-8")
        write()
        assert len(read_geology(path, template)["faults"]) == 2
        document["faults"][0]["inferred"] = "false"
        write()
        with self.assertRaisesRegex(ValueError, "boolean"):
            read_geology(path, template)
        document["faults"][0]["inferred"] = False
        document["crs"] = "EPSG:4326"
        write()
        with self.assertRaisesRegex(ValueError, "CRS"):
            read_geology(path, template)

    def test_model_each_geological_channel_affects_predictions(self):
        torch.manual_seed(16)
        model = RelationKAGAT(2, [3], self.config()["model"], [0]).eval()
        for layer in [*model.geo_bias, *model.geo_selector, *model.geo_messages]:
            torch.nn.init.normal_(layer.weight, std=.2)
        x = torch.cat((torch.randn(6, 2), torch.ones(6, 1), torch.tensor([[0., 1., 0.]]).repeat(6, 1)), 1)
        edges = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]])
        attrs = torch.zeros(5, 4)
        attrs[:, 0] = .5
        with torch.no_grad():
            zero = model(x, edges, attrs)
            model.geo_enabled = False
            torch.testing.assert_close(zero, model(x, edges, torch.ones_like(attrs) * attrs.new_tensor([.5, 1, 1, 1])))
            model.geo_enabled = True
            for channel in range(1, 4):
                changed = attrs.clone()
                changed[:, channel] = 1
                assert (model(x, edges, changed) - zero).abs().max() > 1e-5
                full = model(x, edges, changed)
                batched = predict_logits(model, x, edges, changed, np.arange(6), 2)
                torch.testing.assert_close(full, batched, atol=2e-6, rtol=1e-5)
        model.train()
        model(x, edges, torch.ones_like(attrs)).sum().backward()
        assert all(encoder.spline_weight.grad.abs().sum() > 0 for encoder in model.geo_encoders)

    def test_center_groups_patch_spacing_and_each_fold_preprocessing(self):
        path = make_demo(self.root / "demo", ROOT / "configs/default.yaml", 1)
        cfg = pipeline.load_config(path)
        pipeline.prepare(cfg)
        data, meta = pipeline.load_prepared(cfg)
        work = Path(cfg["paths"]["work"])
        samples = json.loads((work / "samples.json").read_text())
        frame = pd.read_csv(cfg["paths"]["nodes"])
        row, col = frame.row.to_numpy(), frame.col.to_numpy()
        negative = np.asarray(samples["negative_centers"])
        cells = np.c_[row[negative], col[negative]]
        spacing = np.max(np.abs(cells[:, None] - cells[None, :]), axis=2).astype(float)
        np.fill_diagonal(spacing, np.inf)
        assert spacing.min() >= 3
        assert len(expand_centers(frame, negative)) == 9 * len(negative)
        saved = json.loads((work / "preprocessing.json").read_text())
        for fold in meta["folds"]:
            name = f"fold_{fold['number']}"
            expected, cards, state = transform_features(frame, cfg["features"], data[f"{name}_train"])
            np.testing.assert_array_equal(expected, data[f"{name}_x"])
            assert cards == fold["cardinalities"] and state == saved[name]
        # Transitive overlap chains are one component, before any fold assignment.
        tiny = pd.DataFrame(dict(row=[1, 3, 5, 20], col=[1, 1, 1, 20]))
        groups = center_components(tiny, [0, 1, 2, 3])
        assert [group.tolist() for group in groups] == [[0, 1, 2], [3]]

    def test_changed_geology_is_rejected_after_preparation(self):
        path = make_demo(self.root / "demo", ROOT / "configs/default.yaml", 1)
        cfg = pipeline.load_config(path)
        pipeline.prepare(cfg)
        geometry = Path(cfg["paths"]["geology"])
        geometry.write_text(geometry.read_text() + "\n# changed\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Private input changed"):
            pipeline.load_prepared(cfg)

    def test_sampling_rejects_non_1km_grid(self):
        path = make_demo(self.root / "demo", ROOT / "configs/default.yaml", 1)
        cfg = pipeline.load_config(path)
        wrong = self.root / "wrong_resolution.tif"
        with rasterio.open(cfg["paths"]["template"]) as source:
            profile = source.profile.copy()
            values = source.read(1)
        profile["transform"] = from_origin(0, 40000, 500, 500)
        with rasterio.open(wrong, "w", **profile) as destination:
            destination.write(values, 1)
        with self.assertRaisesRegex(ValueError, "1 km"):
            read_nodes(cfg["paths"]["nodes"], wrong, cfg["features"])

    def test_cv_reuse_checks_settings_and_artifact_integrity(self):
        path = make_demo(self.root / "demo", ROOT / "configs/default.yaml", 1)
        cfg = pipeline.load_config(path)
        pipeline.prepare(cfg)
        result = pipeline.cross_validate(cfg)
        with patch.object(pipeline, "_fit", side_effect=AssertionError("Cached CV must not retrain")):
            assert pipeline.cross_validate(cfg) == result
        changed = copy.deepcopy(cfg)
        changed["training"]["learning_rate"] *= 2
        with self.assertRaisesRegex(ValueError, "configuration mismatch"):
            pipeline.cross_validate(changed)
        history = Path(cfg["paths"]["work"]) / "cross_validation/fold_0_history.json"
        history.write_text("[]", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            pipeline.cross_validate(cfg)


if __name__ == "__main__":
    unittest.main()
