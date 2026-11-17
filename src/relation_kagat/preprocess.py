"""Research preprocessing: assay CLR, staged IDW and training-fold median/IQR."""
import warnings
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from scipy.spatial import cKDTree


def clr(values, invalid_values=(), zero_replacement=None):
    """Original row CLR: use observed components; preserve missing components.

    Nonpositive assays require the explicit ``half_min_positive`` policy.
    Replacement minima are computed per assay element, before interpolation.
    """
    values = np.array(values, dtype=float, copy=True)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("CLR requires at least two columns")
    values[~np.isfinite(values)] = np.nan
    for sentinel in invalid_values:
        values[values == float(sentinel)] = np.nan
    if zero_replacement not in {None, "half_min_positive"}:
        raise ValueError("Unknown nonpositive assay policy")
    if np.any(values <= 0):
        if zero_replacement != "half_min_positive":
            raise ValueError("Nonpositive assays require explicit half_min_positive replacement")
        for index in range(values.shape[1]):
            positive = values[:, index][values[:, index] > 0]
            if not len(positive):
                raise ValueError(f"Assay element {index} has no positive observations")
            values[values[:, index] <= 0, index] = positive.min() / 2
    logs = np.log(values)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return logs - np.nanmean(logs, axis=1, keepdims=True)


def idw(point_xy, values, target_xy, neighbors=12, power=2.0,
        max_distance=2000.0, min_neighbors=3, fallback_distance=5000.0,
        fallback_min_neighbors=3):
    """Original two-stage IDW: 2 km then 5 km, >=3 points, no nearest fill.

    Duplicate positions retain the original algorithm's exact-point averaging.
    A coincident observation does not bypass the minimum-support requirement.
    """
    point_xy, target_xy, values = map(lambda x: np.asarray(x, dtype=float),
                                     (point_xy, target_xy, values))
    if point_xy.shape != (len(values), 2) or target_xy.ndim != 2 or target_xy.shape[1] != 2:
        raise ValueError("IDW expects (N, 2) coordinates and one-dimensional values")
    if values.ndim != 1 or not np.isfinite(target_xy).all():
        raise ValueError("Invalid IDW inputs")
    if (not isinstance(neighbors, int) or neighbors < 1 or not np.isfinite(power) or power <= 0
            or not np.isfinite([max_distance, fallback_distance]).all()
            or not 0 < max_distance <= fallback_distance
            or not 1 <= min_neighbors <= neighbors or not 1 <= fallback_min_neighbors <= neighbors):
        raise ValueError("Invalid IDW settings")
    valid = np.isfinite(values) & np.isfinite(point_xy).all(1)
    if not valid.any():
        return np.full(len(target_xy), np.nan)
    points, values = point_xy[valid], values[valid]
    tree = cKDTree(points)
    result = np.full(len(target_xy), np.nan)
    for row, target in enumerate(target_xy):
        for radius, minimum in [(max_distance, min_neighbors),
                                (fallback_distance, fallback_min_neighbors)]:
            distance, index = tree.query(target, k=neighbors, distance_upper_bound=radius)
            distance, index = np.atleast_1d(distance), np.atleast_1d(index)
            keep = np.isfinite(distance) & (index < len(values))
            if keep.sum() < minimum:
                continue
            distance, local = distance[keep], values[index[keep]]
            exact = distance == 0
            if exact.any():
                result[row] = local[exact].mean()
            else:
                weights = distance ** -power
                result[row] = np.sum(weights * local) / weights.sum()
            break
    return result


def interpolate_assays(point_xy, assays, target_xy, settings):
    """CLR at assay locations, then interpolate each element independently."""
    transformed = clr(assays, settings.get("invalid_values", [-1e38]),
                      settings.get("zero_replacement", "half_min_positive"))
    options = settings["idw"]
    return np.column_stack([idw(
        point_xy, column, target_xy, neighbors=options["max_neighbors"], power=options["power"],
        max_distance=options["search_radius_m"], min_neighbors=options["min_neighbors"],
        fallback_distance=options["fallback_search_radius_m"],
        fallback_min_neighbors=options["fallback_min_neighbors"])
        for column in transformed.T])


def require_metric_grid(src):
    if src.count != 1 or src.crs is None or not src.crs.is_projected:
        raise ValueError("Use a single-band projected template")
    if src.crs.linear_units != "metre":
        raise ValueError("The grid CRS must use metres")
    if abs(src.transform.a * src.transform.e - src.transform.b * src.transform.d) <= 0:
        raise ValueError("Degenerate raster transform")


def aligned_raster_table(template, rasters):
    """Extract named raster bands without resampling; masks and zeros are preserved.

    `rasters` maps output column names to private raster paths. Include geological
    IDs and known_positive, or join those separately before writing the node CSV.
    """
    with rasterio.open(template) as ref:
        require_metric_grid(ref)
        row, col = np.nonzero(ref.read_masks(1) > 0)
        x, y = rasterio.transform.xy(ref.transform, row, col)
        frame = pd.DataFrame(dict(node_id=[f"r{r}_c{c}" for r, c in zip(row, col)],
                                  row=row, col=col, x=x, y=y))
        for name, path in rasters.items():
            if name in frame:
                raise ValueError(f"Reserved raster column: {name}")
            with rasterio.open(path) as src:
                if src.count != 1 or src.crs != ref.crs or src.shape != ref.shape or src.transform != ref.transform:
                    raise ValueError(f"Raster is not aligned: {name}")
                frame[name] = src.read(1, masked=True).astype(float).filled(np.nan)[row, col]
    return frame


def read_nodes(path, template, features):
    frame = pd.read_csv(path, dtype={"node_id": str})
    required = {"node_id", "row", "col", "known_positive"}
    required.update(features["continuous"] + features["categorical"])
    if required - set(frame):
        raise ValueError(f"Missing columns: {sorted(required - set(frame))}")
    if frame.node_id.isna().any() or frame.node_id.duplicated().any():
        raise ValueError("node_id must be nonempty and unique")
    for name in ["row", "col"]:
        if not np.isfinite(frame[name]).all() or np.any(frame[name] != np.floor(frame[name])):
            raise ValueError(f"{name} must contain finite integer indices")
        frame[name] = frame[name].astype(np.int64)
    frame = frame.sort_values(["row", "col"]).reset_index(drop=True)
    if frame.duplicated(["row", "col"]).any() or not frame.known_positive.isin([0, 1]).all():
        raise ValueError("Duplicate cells or invalid known_positive flags")
    with rasterio.open(template) as src:
        require_metric_grid(src)
        if (not np.allclose([abs(src.transform.a), abs(src.transform.e)], 1000., rtol=0, atol=1e-6)
                or not np.allclose([src.transform.b, src.transform.d], 0., rtol=0, atol=1e-9)):
            raise ValueError("Center-based 3x3 sampling requires an unrotated 1 km grid")
        row, col = np.nonzero(src.read_masks(1) > 0)
        if not np.array_equal(frame[["row", "col"]].to_numpy(), np.column_stack((row, col))):
            raise ValueError("Node table must cover every valid template cell exactly once")
        xx, yy = rasterio.transform.xy(src.transform, row, col)
        for name, values in [("x", xx), ("y", yy)]:
            if name in frame and not np.allclose(frame[name], values, atol=0.01, rtol=0):
                raise ValueError("Coordinates do not match grid cell centers")
            frame[name] = values
    return frame


def transform_features(frame, features, fit_indices=None, state=None):
    names, cat_names = features["continuous"], features["categorical"]
    if not names or len(set(names + cat_names)) != len(names + cat_names):
        raise ValueError("Feature names must be nonempty and unique")
    if set(names + cat_names) & {"known_positive", "node_id", "row", "col", "x", "y"}:
        raise ValueError("Labels, identifiers and coordinates cannot be configured as predictors")
    values = frame[names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    values[~np.isfinite(values)] = np.nan
    clr_names, log_names = features.get("clr", []), features.get("log1p", [])
    if set(clr_names) & set(log_names) or (set(clr_names) | set(log_names)) - set(names):
        raise ValueError("CLR/log1p must be disjoint subsets of continuous features")
    if clr_names:
        columns = [names.index(name) for name in clr_names]
        values[:, columns] = clr(values[:, columns])
    for name in log_names:
        index = names.index(name)
        if np.any(values[:, index] < 0):
            raise ValueError("log1p features must be nonnegative")
        values[:, index] = np.log1p(values[:, index])
    # The research loader materializes its transformed feature table as float32.
    values = values.astype(np.float32).astype(float)
    chemistry = features.get("geochemistry", [name for name in names if name.endswith("_clr")])
    magnetic = features.get("aeromagnetic", "aeromagnetic_F3")
    if not chemistry or set(chemistry) - set(names) or magnetic not in names:
        raise ValueError("Configure geochemistry columns and one aeromagnetic column within continuous features")
    chemistry_indices = [names.index(name) for name in chemistry]
    if state is None:
        if fit_indices is None or len(fit_indices) == 0:
            raise ValueError("Training indices required to fit preprocessing")
        train = values[fit_indices]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            center = np.nan_to_num(np.nanmedian(train, axis=0), nan=0.)
            q25, q75 = np.nanpercentile(train, [25, 75], axis=0)
            scale = q75 - q25
            fallback = np.nanstd(train, axis=0)
        scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, fallback)
        scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.)
        state = dict(method="train_fold_median_IQR", features=deepcopy(features),
                     center=center.tolist(), scale=scale.tolist(), categories={},
                     all_missing_training_features=[names[i] for i in np.flatnonzero(~np.isfinite(train).any(0))])
        for name in cat_names:
            raw = frame[name].to_numpy(dtype=float)
            known = raw[fit_indices]
            known = known[np.isfinite(known)]
            state["categories"][name] = sorted(np.unique(known).tolist())
    if state["features"] != features:
        raise ValueError("Feature configuration changed after preprocessing was fitted")
    if state.get("method") != "train_fold_median_IQR":
        raise ValueError("Incompatible preprocessing schema; prepare a fresh run")
    normalized = np.nan_to_num((values - state["center"]) / state["scale"], nan=0., posinf=0., neginf=0.)
    categories, cardinalities = [], []
    for name in cat_names:
        raw = frame[name].to_numpy(dtype=float)
        if np.isinf(raw).any() or np.any(raw[np.isfinite(raw)] != np.floor(raw[np.isfinite(raw)])):
            raise ValueError("Categorical IDs must be integers or missing")
        mapping = {value: index + 1 for index, value in enumerate(state["categories"][name])}
        categories.append(np.array([mapping.get(value, 0) for value in raw])[:, None])
        cardinalities.append(len(mapping) + 1)
    missing = np.column_stack((np.isnan(values[:, chemistry_indices]).mean(1),
                               np.isfinite(values[:, chemistry_indices]).all(1),
                               np.isnan(values[:, names.index(magnetic)])))
    encoded = np.concatenate((normalized, *categories, missing), axis=1).astype(np.float32)
    if not np.isfinite(encoded).all():
        raise ValueError("Nonfinite encoded features")
    return encoded, cardinalities, state
