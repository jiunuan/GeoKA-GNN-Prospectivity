"""Prepare, three-fold development CV, final refit, held-out test and map audit."""
import hashlib
import json
import os
import random
from pathlib import Path
import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from .graph import build_graph, read_geology, EDGE_NAMES
from .model import RelationKAGAT, target_subgraph, predict_logits
from .preprocess import read_nodes, transform_features
from .quality import export_percentile, write_json
from .sampling import fixed_samples

SCHEMA = "geoka_three_fold_geology_v2"
INPUT_KEYS = ["nodes", "template", "geology"]


def load_config(path):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in [*INPUT_KEYS, "work"]:
        cfg["paths"][key] = str((path.parent / cfg["paths"][key]).resolve())
    work = Path(cfg["paths"]["work"])
    for key in INPUT_KEYS:
        source = Path(cfg["paths"][key])
        if source == work or work in source.parents:
            raise ValueError("Keep input data outside the run output directory")
    return cfg


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def status(work, stage, state, **details):
    write_json(work / "run_status.json", dict(stage=stage, status=state, **details))


def prepare(cfg):
    work = Path(cfg["paths"]["work"])
    work.mkdir(parents=True, exist_ok=True)
    if (work / "prepared.npz").exists():
        raise FileExistsError("Prepared run exists; use train/predict or a fresh work directory")
    status(work, "prepare", "running")
    try:
        frame = read_nodes(cfg["paths"]["nodes"], cfg["paths"]["template"], cfg["features"])
        samples = fixed_samples(frame, cfg["sampling"], work / "samples.json")
        geology = read_geology(cfg["paths"]["geology"], cfg["paths"]["template"])
        edges, attrs = build_graph(frame, cfg["graph"], geology)
        labels = np.full(len(frame), -1, np.float32)
        labels[samples["positive_nodes"]] = 1
        labels[samples["negative_nodes"]] = 0
        arrays = dict(edge_index=edges, edge_attr=attrs, labels=labels,
                      row=frame.row.to_numpy(), col=frame.col.to_numpy(),
                      development_indices=np.asarray(samples["splits"]["development"], np.int64),
                      test_indices=np.asarray(samples["splits"]["test"], np.int64))
        preprocessing, folds = {}, []
        # Every fold has its own feature matrix, statistics and category vocabulary.
        for fold in samples["folds"]:
            name = f"fold_{fold['number']}"
            x, cardinalities, state = transform_features(frame, cfg["features"], fold["train"])
            arrays[f"{name}_x"] = x
            arrays[f"{name}_train"] = np.asarray(fold["train"], np.int64)
            arrays[f"{name}_validation"] = np.asarray(fold["validation"], np.int64)
            preprocessing[name] = state
            folds.append(dict(number=fold["number"], cardinalities=cardinalities))
        arrays["x"], cardinalities, preprocessing["final"] = transform_features(
            frame, cfg["features"], samples["splits"]["development"])
        np.savez_compressed(work / "prepared.npz", **arrays)
        write_json(work / "preprocessing.json", preprocessing)
        write_json(work / "prepared_metadata.json", dict(
            schema=SCHEMA, continuous_count=len(cfg["features"]["continuous"]),
            cardinalities=cardinalities, folds=folds, missing_count=3,
            geochemistry_indices=[cfg["features"]["continuous"].index(name)
                                  for name in cfg["features"]["geochemistry"]],
            edge_names=EDGE_NAMES, features=cfg["features"], graph=cfg["graph"], sampling=cfg["sampling"],
            inputs_sha256={key: sha256(cfg["paths"][key]) for key in INPUT_KEYS},
            artifacts_sha256={name: sha256(work / name) for name in
                              ["prepared.npz", "samples.json", "preprocessing.json"]}))
        status(work, "prepare", "prepared", nodes=len(frame), directed_edges=edges.shape[1], folds=3)
    except Exception as error:
        status(work, "prepare", "failed", error=str(error))
        raise


def load_prepared(cfg):
    work = Path(cfg["paths"]["work"])
    metadata = json.loads((work / "prepared_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("schema") != SCHEMA:
        raise ValueError("Incompatible prepared schema; use a fresh work directory")
    for section in ["features", "graph", "sampling"]:
        if cfg[section] != metadata[section]:
            raise ValueError(f"Prepared {section} configuration changed")
    for key, digest in metadata["inputs_sha256"].items():
        if sha256(cfg["paths"][key]) != digest:
            raise ValueError(f"Private input changed since preparation: {key}")
    for name, digest in metadata["artifacts_sha256"].items():
        if sha256(work / name) != digest:
            raise ValueError(f"Prepared artifact changed: {name}")
    with np.load(work / "prepared.npz", allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    return data, metadata


def metrics(labels, logits):
    scores = torch.sigmoid(torch.as_tensor(logits)).numpy()
    predicted = scores >= .5
    return dict(AUC=float(roc_auc_score(labels, scores)), accuracy=float(accuracy_score(labels, predicted)),
                precision=float(precision_score(labels, predicted, zero_division=0)),
                recall=float(recall_score(labels, predicted, zero_division=0)),
                F1=float(f1_score(labels, predicted, zero_division=0)), threshold=.5,
                threshold_semantics="raw_relative_score_not_map_percentile_or_probability")


def tensors(data, device):
    return (torch.as_tensor(data["x"], device=device),
            torch.as_tensor(data["edge_index"], device=device),
            torch.as_tensor(data["edge_attr"], device=device))


def _training_signature(options):
    return {key: value for key, value in options.items() if key not in {"device", "inference_batch_size"}}


def _seed(options):
    if any(int(options[k]) < 1 for k in ["epochs", "patience", "batch_size", "inference_batch_size"]):
        raise ValueError("Training counts must be positive")
    if not 0 <= options["label_smoothing"] < 1:
        raise ValueError("label_smoothing must be in [0, 1)")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(int(options["seed"]))
    np.random.seed(int(options["seed"]))
    torch.manual_seed(int(options["seed"]))
    torch.use_deterministic_algorithms(True)


def _model(meta, settings, device):
    return RelationKAGAT(meta["continuous_count"], meta["cardinalities"], settings,
                        meta["geochemistry_indices"]).to(device)


def _fit(data, meta, cfg, train_indices, validation_indices, epochs, label):
    options = cfg["training"]
    _seed(options)
    device = torch.device(options["device"])
    x, edges, attrs = tensors(data, device)
    labels = torch.as_tensor(data["labels"], device=device)
    model = _model(meta, cfg["model"], device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=options["learning_rate"], weight_decay=options["weight_decay"])
    best_auc, best_state, stale, history, best_epoch = -1., None, 0, [], 0
    rng = np.random.default_rng(int(options["seed"]))
    for epoch in range(1, int(epochs) + 1):
        model.train()
        order = rng.permutation(train_indices)
        total_loss = 0.
        optimizer.zero_grad()
        for start in range(0, len(order), options["batch_size"]):
            ids = order[start:start + options["batch_size"]]
            xx, ee, aa, selected = target_subgraph(x, edges, attrs, ids)
            logits = model(xx, ee, aa)[selected]
            truth = labels[torch.as_tensor(ids, device=device)]
            truth = truth * (1 - options["label_smoothing"]) + .5 * options["label_smoothing"]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, truth, reduction="sum") / len(order)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            total_loss += float(loss.detach())
        l1, smooth = model.regularization()
        penalty = options["spline_l1"] * l1 + options["spline_smoothness"] * smooth
        penalty.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), options["clip_grad_norm"], error_if_nonfinite=True)
        optimizer.step()
        row = dict(epoch=epoch, loss=total_loss, regularization=float(penalty.detach()))
        improved = validation_indices is None
        if validation_indices is not None:
            logits = predict_logits(model, x, edges, attrs, validation_indices, options["inference_batch_size"])
            validation = metrics(data["labels"][validation_indices], logits)
            row["validation_AUC"] = validation["AUC"]
            improved = validation["AUC"] > best_auc
            if improved:
                best_auc = validation["AUC"]
        history.append(row)
        if improved:
            best_epoch, stale = epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"{label} epoch={epoch} loss={total_loss:.5f}" +
                  (f" validation_AUC={row['validation_AUC']:.4f}" if validation_indices is not None else ""), flush=True)
        if validation_indices is not None and stale >= options["patience"]:
            break
    model.load_state_dict(best_state)
    return model, history, best_epoch


def _binding(cfg, work):
    return dict(model_settings=cfg["model"], training_settings=_training_signature(cfg["training"]),
                prepared_sha256=sha256(work / "prepared.npz"),
                metadata_sha256=sha256(work / "prepared_metadata.json"))


def _read_cv(cfg, work):
    folder = work / "cross_validation"
    result = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    if result["binding"] != _binding(cfg, work):
        raise ValueError("Cross-validation/input/configuration mismatch; use a fresh work directory")
    for name, digest in result["artifacts_sha256"].items():
        if sha256(folder / name) != digest:
            raise ValueError(f"Cross-validation artifact changed: {name}")
    return result


def cross_validate(cfg):
    work = Path(cfg["paths"]["work"])
    folder = work / "cross_validation"
    if (folder / "summary.json").exists():
        load_prepared(cfg)
        return _read_cv(cfg, work)
    status(work, "cross_validate", "running")
    try:
        data, meta = load_prepared(cfg)
        folder.mkdir(exist_ok=False)
        oof = np.full(len(data["labels"]), np.nan)
        results, artifacts = [], []
        for fold in meta["folds"]:
            name = f"fold_{fold['number']}"
            fold_data = dict(data, x=data[f"{name}_x"])
            fold_meta = dict(meta, cardinalities=fold["cardinalities"])
            train_ids, valid_ids = data[f"{name}_train"], data[f"{name}_validation"]
            model, history, epoch = _fit(fold_data, fold_meta, cfg, train_ids, valid_ids,
                                         cfg["training"]["epochs"], name)
            x, edges, attrs = tensors(fold_data, torch.device(cfg["training"]["device"]))
            logits = predict_logits(model, x, edges, attrs, valid_ids, cfg["training"]["inference_batch_size"])
            if np.isfinite(oof[valid_ids]).any():
                raise ValueError("Duplicate OOF validation nodes")
            oof[valid_ids] = logits.numpy()
            results.append(dict(number=fold["number"], best_epoch=epoch, train_nodes=len(train_ids),
                                validation_nodes=len(valid_ids), metrics=metrics(data["labels"][valid_ids], logits)))
            torch.save(dict(state_dict={key: value.detach().cpu() for key, value in model.state_dict().items()},
                            fold=fold["number"], cardinalities=fold["cardinalities"],
                            binding=_binding(cfg, work)), folder / f"{name}.pt")
            write_json(folder / f"{name}_history.json", history)
            artifacts.extend([f"{name}.pt", f"{name}_history.json"])
        dev = data["development_indices"]
        if not np.isfinite(oof[dev]).all() or np.isfinite(oof[data["test_indices"]]).any():
            raise ValueError("Invalid OOF coverage or outer-test leakage")
        np.savez_compressed(folder / "oof.npz", node_indices=dev, labels=data["labels"][dev],
                            raw_scores=torch.sigmoid(torch.as_tensor(oof[dev])).numpy())
        artifacts.append("oof.npz")
        auc = [fold["metrics"]["AUC"] for fold in results]
        result = dict(binding=_binding(cfg, work), folds=results, mean_fold_AUC=float(np.mean(auc)),
                      std_fold_AUC=float(np.std(auc)), pooled_OOF_metrics=metrics(data["labels"][dev], oof[dev]),
                      refit_epochs=max(1, int(np.rint(np.median([fold["best_epoch"] for fold in results])))),
                      refit_rule="rounded_median_best_epoch_across_three_development_folds",
                      outer_test_used=False,
                      artifacts_sha256={name: sha256(folder / name) for name in artifacts})
        write_json(folder / "summary.json", result)
        status(work, "cross_validate", "cross_validated", folds=3, refit_epochs=result["refit_epochs"])
        return result
    except Exception as error:
        status(work, "cross_validate", "failed", error=str(error))
        raise


def train(cfg):
    work = Path(cfg["paths"]["work"])
    if (work / "model.pt").exists():
        raise FileExistsError("Model already exists; refusing to overwrite completed training")
    try:
        cv = cross_validate(cfg)
        status(work, "train", "running")
        data, meta = load_prepared(cfg)
        # Refit on all development nodes for a CV-chosen fixed number of epochs.
        # Neither early stopping nor epoch selection consults the outer test.
        model, history, epoch = _fit(data, meta, cfg, data["development_indices"], None,
                                     cv["refit_epochs"], "final")
        x, edges, attrs = tensors(data, torch.device(cfg["training"]["device"]))
        logits = predict_logits(model, x, edges, attrs, data["test_indices"], cfg["training"]["inference_batch_size"])
        result = metrics(data["labels"][data["test_indices"]], logits)
        torch.save(dict(state_dict={key: value.detach().cpu() for key, value in model.state_dict().items()},
                        binding=_binding(cfg, work), best_epoch=epoch,
                        cv_summary_sha256=sha256(work / "cross_validation/summary.json")), work / "model.pt")
        write_json(work / "training_history.json", history)
        write_json(work / "test_metrics.json", result)
        status(work, "train", "trained", refit_epochs=epoch, outer_test_evaluated_once=True,
               note="Raster release gate has not yet run")
    except Exception as error:
        status(work, "train", "failed", error=str(error))
        raise


def predict(cfg):
    work = Path(cfg["paths"]["work"])
    if any((work / "prediction").glob("*.tif")):
        raise FileExistsError("Existing prediction raster; inspect its status instead of overwriting")
    status(work, "predict", "running")
    try:
        data, meta = load_prepared(cfg)
        _read_cv(cfg, work)
        checkpoint = torch.load(work / "model.pt", map_location="cpu", weights_only=True)
        if (checkpoint["binding"] != _binding(cfg, work) or checkpoint["cv_summary_sha256"] !=
                sha256(work / "cross_validation/summary.json")):
            raise ValueError("Checkpoint/input/configuration mismatch")
        device = torch.device(cfg["training"]["device"])
        model = _model(meta, cfg["model"], device)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        x, edges, attrs = tensors(data, device)
        logits = predict_logits(model, x, edges, attrs, np.arange(len(x)), cfg["training"]["inference_batch_size"])
        raw = torch.sigmoid(logits).numpy()
        output = export_percentile(cfg["paths"]["template"], data["row"], data["col"], raw, work / "prediction")
        status(work, "predict", "complete", raster=output.name, semantics="relative_area_percentile_not_probability")
    except Exception as error:
        status(work, "predict", "failed", error=str(error))
        raise
