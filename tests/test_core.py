import json
from pathlib import Path
import numpy as np
import pandas as pd
import unittest
import tempfile
import importlib.util
from unittest.mock import patch
import rasterio
import torch
import yaml
import shapely
from scipy.interpolate import BSpline
from rasterio.transform import from_origin
from relation_kagat.kan import KANLinear
from relation_kagat.model import RelationKAGAT, predict_logits
from relation_kagat.preprocess import clr, idw, transform_features, aligned_raster_table, read_nodes
from relation_kagat.graph import build_graph
from relation_kagat.sampling import fixed_samples
from relation_kagat.quality import audit_values, export_percentile
from relation_kagat.synthetic import make_demo
from relation_kagat import pipeline

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]


def config():
    return yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="relation_kagat_test_")
        self.tmp_path = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def test_splines_match_scipy_and_have_gradients(self):
        layer = KANLinear(2, 3, grid_size=3, spline_order=3)
        x = torch.tensor([[-.8, -.4], [.2, .7]], requires_grad=True)
        basis = layer.b_splines(x).detach().numpy()
        for feature in range(2):
            expected = BSpline.design_matrix(x.detach().numpy()[:, feature], layer.grid[feature].numpy(), 3).toarray()
            np.testing.assert_allclose(basis[:, feature], expected, atol=2e-7)
        layer(x).square().sum().backward()
        assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


    def test_model_full_graph_equals_target_batches_and_backward(self):
        torch.manual_seed(7)
        model = RelationKAGAT(2, [3], config()["model"])
        assert model.dropout == .3
        x = torch.cat((torch.randn(9, 2), torch.ones(9, 1), torch.ones(9, 3)), 1)
        # Node eight is isolated and still receives its own residual update.
        pairs = [(i, j) for i in range(8) for j in range(8) if abs(i - j) == 1]
        edges = torch.tensor(pairs).T
        attrs = torch.rand(len(pairs), 4)
        model.train()
        model(x, edges, attrs).sum().backward()
        assert model.relation_first.spline_weight.grad.abs().sum() > 0
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        model.eval()
        with torch.no_grad():
            full = model(x, edges, attrs)
        chunked = predict_logits(model, x, edges, attrs, np.arange(9), 3)
        torch.testing.assert_close(full, chunked, atol=2e-6, rtol=1e-5)


    def test_preprocessing_no_validation_leakage(self):
        frame = pd.DataFrame(dict(a=[1., 3., 1000.], b=[2., np.nan, 2000.], age=[1, 2, 9]))
        features = dict(continuous=["a", "b"], categorical=["age"], clr=[], log1p=[],
                        geochemistry=["a"], aeromagnetic="b")
        x, cardinalities, state = transform_features(frame, features, [0, 1])
        assert state["center"][0] == 2 and state["center"][1] == 2 and state["scale"][0] == 1
        assert cardinalities == [3] and x[2, 2] == 0 and x[1, -1] == 1
        replay, _, _ = transform_features(frame, features, state=state)
        np.testing.assert_array_equal(x, replay)
        np.testing.assert_allclose(clr([[1., 2., 4.]]).sum(1), 0, atol=1e-12)
        with self.assertRaises(ValueError):
            clr([[0., 1.]])
        assert np.isnan(clr([[np.nan, 1.]])[0, 0])
        assert clr([[np.nan, 1.]])[0, 1] == 0

    def test_label_leakage_is_rejected(self):
        frame = pd.DataFrame(dict(known_positive=[0, 1]))
        with self.assertRaisesRegex(ValueError, "cannot be configured"):
            transform_features(frame, dict(continuous=["known_positive"], categorical=[]), [0, 1])

    def test_public_tree_rejects_data_and_private_paths(self):
        spec = importlib.util.spec_from_file_location("public_check", ROOT / "tools/check_public_tree.py")
        checker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(checker)
        (self.tmp_path / "samples.csv").write_text("private content", encoding="utf-8")
        (_, failures) = checker.audit(self.tmp_path)
        assert any("Unexpected/non-source" in reason for reason in failures)
        # Assemble a fake path so this test itself does not embed a private path.
        (self.tmp_path / "README.md").write_text("Z" + ":" + "/private/input", encoding="utf-8")
        (_, failures) = checker.audit(self.tmp_path)
        assert any("absolute Windows path" in reason for reason in failures)

    def test_publication_inventory_rejects_changed_sources(self):
        spec = importlib.util.spec_from_file_location("public_check", ROOT / "tools/check_public_tree.py")
        checker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(checker)
        readme = self.tmp_path / "README.md"
        readme.write_text("Artificial source", encoding="utf-8")
        checker.write_manifest(self.tmp_path, ["README.md"])
        files, failures = checker.audit(self.tmp_path)
        assert not failures and not checker.verify_manifest(self.tmp_path, files)
        readme.write_text("Changed source", encoding="utf-8")
        assert any("SHA256 mismatch" in reason for reason in checker.verify_manifest(self.tmp_path, files))

    def test_empty_faults_and_isolated_graph(self):
        frame = pd.DataFrame(dict(x=[0., 100.], y=[0., 0.],
                                  stratigraphic_age=[1, 2], intrusive_body_id=[0, 0]))
        edges, attrs = build_graph(frame, dict(candidate_radius_m=1.), dict(faults=[], igneous_bodies=[]))
        assert edges.shape == (2, 0) and attrs.shape == (0, 4)


    def test_idw_exact_points_support_and_duplicate_mean(self):
        values = idw([[0, 0], [2, 0]], [2, 4], [[0, 0], [1, 0], [50, 0]], neighbors=2,
                     max_distance=3, min_neighbors=2, fallback_distance=3, fallback_min_neighbors=2)
        np.testing.assert_allclose(values[:2], [2, 3])
        assert np.isnan(values[2])
        repeated = idw([[0, 0], [0, 0], [1, 0]], [2, 4, 9], [[0, 0]], max_distance=3, fallback_distance=3)
        assert repeated[0] == 3


    def test_graph_semantics_distance_and_fault_touch(self):
        frame = pd.DataFrame(dict(x=[0., 1., 2.], y=[0., 0., 0.],
                                  stratigraphic_age=[0, 0, 1], intrusive_body_id=[1, 1, 0]))
        geology = dict(faults=[shapely.from_wkt("LINESTRING (1 0, 1 1)")], igneous_bodies=[])
        edges, attrs = build_graph(frame, dict(candidate_radius_m=1.1), geology)
        assert edges.shape == (2, 4)
        np.testing.assert_array_equal(attrs[:2], attrs[2:])
        assert not attrs[:, 1].any() and not attrs[:, 3].any()  # touching is not crossing
        np.testing.assert_allclose(attrs[:, 0], 1 / 1.1)


    def test_fixed_samples_reuse_and_binding(self):
        tmp_path = self.tmp_path
        path = make_demo(tmp_path / "demo", ROOT / "configs/default.yaml", 1)
        cfg = pipeline.load_config(path)
        frame = read_nodes(cfg["paths"]["nodes"], cfg["paths"]["template"], cfg["features"])
        manifest = tmp_path / "samples.json"
        first = fixed_samples(frame, cfg["sampling"], manifest)
        second = fixed_samples(frame, cfg["sampling"], manifest)
        assert first == second
        assert len(first["positive_centers"]) == len(first["negative_centers"])
        assert len(first["folds"]) == 3
        development, test = map(set, [first["splits"]["development"], first["splits"]["test"]])
        assert not development & test
        validations = []
        for fold in first["folds"]:
            train, valid = set(fold["train"]), set(fold["validation"])
            assert not train & valid and train | valid == development and not (train | valid) & test
            validations.extend(fold["validation"])
        assert len(validations) == len(set(validations)) and set(validations) == development
        frame.loc[0, "geochem_1_clr"] += 1
        with self.assertRaisesRegex(ValueError, "Frozen"):
            fixed_samples(frame, cfg["sampling"], manifest)


    def test_actual_tiff_readback_and_distribution_rejection(self):
        tmp_path = self.tmp_path
        template = tmp_path / "template.tif"
        with rasterio.open(template, "w", driver="GTiff", width=10, height=10, count=1,
                           dtype="uint8", crs="EPSG:3857", transform=from_origin(0, 10000, 1000, 1000), nodata=0) as dst:
            dst.write(np.ones((10, 10), np.uint8), 1)
        row, col = np.indices((10, 10))
        out = export_percentile(template, row.ravel(), col.ravel(), np.linspace(.01, .99, 100), tmp_path / "pass")
        actual = json.loads((out.parent / "raster_distribution.json").read_text())
        assert actual["passed"] and actual["valid_projected_area_m2"] == 100000000
        for name in ["min", "P05", "P25", "median", "P75", "P90", "P95", "max"]:
            assert name in actual
        with rasterio.open(out) as src:
            reread = audit_values(src.read(1, masked=True).compressed(), "percentile")
        for name in reread:
            assert actual[name] == reread[name]
        assert actual["fraction_ge_0.5"] == .5 and actual["fraction_ge_0.8"] == .2 and actual["fraction_ge_0.9"] == .1
        for values in [np.ones(100), np.full(100, .5), np.linspace(.8, 1, 100), np.full(100, np.nan)]:
            assert not audit_values(values)["passed"]
        with self.assertRaisesRegex(ValueError, "Raw score gate"):
            export_percentile(template, row.ravel(), col.ravel(), np.ones(100), tmp_path / "fail")
        assert not list((tmp_path / "fail").glob("*.tif"))
        extracted = aligned_raster_table(template, {"score": out})
        assert len(extracted) == 100


    def check_training_and_release_status(self, saturated):
        tmp_path = self.tmp_path
        path = make_demo(tmp_path / "demo", ROOT / "configs/default.yaml", 2)
        cfg = pipeline.load_config(path)
        pipeline.prepare(cfg)
        with patch.object(pipeline, "predict_logits", wraps=pipeline.predict_logits) as observed:
            pipeline.train(cfg)
        work = Path(cfg["paths"]["work"])
        data, meta = pipeline.load_prepared(cfg)
        test_ids = set(data["test_indices"])
        calls = [set(call.args[4]) for call in observed.call_args_list]
        assert sum(bool(ids & test_ids) for ids in calls) == 1 and calls[-1] == test_ids
        summary = json.loads((work / "cross_validation/summary.json").read_text())
        assert len(summary["folds"]) == 3 and not summary["outer_test_used"]
        assert json.loads((work / "training_history.json").read_text())[-1]["epoch"] == summary["refit_epochs"]
        with np.load(work / "cross_validation/oof.npz") as oof:
            assert set(oof["node_indices"]) == set(data["development_indices"])
            assert not set(oof["node_indices"]) & test_ids
        assert json.loads((work / "run_status.json").read_text())["status"] == "trained"
        # Test the publication state machine with controlled scores, independently
        # of the stochastic skill of a two-epoch synthetic model.
        def controlled_logits(model, x, edges, attrs, targets, batch_size):
            if saturated:
                return torch.full((len(targets),), 20.)
            scores = torch.linspace(.01, .99, len(targets))
            return torch.logit(scores)
        mock = patch.object(pipeline, "predict_logits", controlled_logits)
        mock.start()
        self.addCleanup(mock.stop)
        if saturated:
            with self.assertRaisesRegex(ValueError, "Raw score gate"):
                pipeline.predict(cfg)
            assert json.loads((work / "run_status.json").read_text())["status"] == "failed"
            assert not list((work / "prediction").glob("*.tif"))
        else:
            pipeline.predict(cfg)
            assert json.loads((work / "run_status.json").read_text())["status"] == "complete"
            assert json.loads((work / "prediction/raster_distribution.json").read_text())["passed"]

    def test_successful_release_status(self):
        self.check_training_and_release_status(False)

    def test_failed_release_status(self):
        self.check_training_and_release_status(True)


if __name__ == "__main__":
    unittest.main()
