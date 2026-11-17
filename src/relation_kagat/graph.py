"""Radius graph with strict same-igneous-body, along-fault and cross-fault rules."""
from pathlib import Path
import numpy as np
import rasterio
import shapely
import yaml
from scipy.spatial import cKDTree

EDGE_NAMES = ["normalized_distance", "same_igneous_body", "along_fault", "cross_fault"]


def read_geology(path, template):
    """Read a private, CRS-declared YAML; discard explicitly inferred faults."""
    document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    with rasterio.open(template) as src:
        if rasterio.crs.CRS.from_user_input(document["crs"]) != src.crs:
            raise ValueError("Geology CRS must equal the metric template CRS; reproject explicitly")
    result = dict(faults=[], igneous_bodies=[])
    for name, types in [("faults", {"LineString", "MultiLineString"}),
                        ("igneous_bodies", {"Polygon", "MultiPolygon"})]:
        records = document[name]
        if not isinstance(records, list):
            raise ValueError(f"{name} must be a list (possibly empty)")
        ids = set()
        for record in records:
            identifier = record["id"]
            if not isinstance(identifier, str) or not identifier or identifier in ids:
                raise ValueError(f"Nonempty unique geometry id required: {name}")
            ids.add(identifier)
            if name == "faults":
                if type(record.get("inferred")) is not bool:
                    raise ValueError("Every fault requires an explicit boolean inferred flag")
                if record["inferred"]:
                    continue
            geometry = shapely.from_wkt(record["wkt"])
            if (geometry.geom_type not in types or geometry.is_empty or not geometry.is_valid
                    or not np.isfinite(shapely.get_coordinates(geometry)).all()):
                raise ValueError(f"Invalid {name} geometry: {identifier}")
            result[name].extend(shapely.get_parts(geometry).tolist())
    return result


def geological_relations(xy, pairs, geology, settings):
    """Compute original Same/Along/Cross rules, without creating new edges.

    Same requires strict endpoint interiors and whole-segment polygon coverage.
    Along uses one mapped fault part, tangent alignment and along-line distance.
    Cross excludes touching and collinear overlap.
    """
    xy, pairs = np.asarray(xy, float), np.asarray(pairs, np.int64).reshape(-1, 2)
    output = np.zeros((len(pairs), 3), np.float32)
    if not len(pairs):
        return output
    near = float(settings.get("fault_buffer_m", 1000.))
    tangent = float(settings.get("fault_tangent_half_window_m", 500.))
    arc_limit = float(settings.get("fault_along_max_m", 2000.))
    decay = float(settings.get("fault_decay_m", 2000.))
    alignment_min = float(settings.get("fault_min_cosine_squared", .5))
    if (not np.isfinite([near, tangent, arc_limit, decay, alignment_min]).all()
            or min(near, tangent, arc_limit, decay) <= 0 or not 0 <= alignment_min <= 1):
        raise ValueError("Invalid geological relation settings")
    points = shapely.points(xy)
    u, v = pairs.T
    bodies = geology["igneous_bodies"]
    if len(bodies):
        hits = shapely.STRtree(bodies).query(points, predicate="within")
        counts = np.bincount(hits[0], minlength=len(xy))
        if (counts > 1).any():
            raise ValueError("Ambiguous strict igneous-body assignment: overlapping polygon interiors")
        body = np.full(len(xy), -1, np.int64)
        body[hits[0]] = hits[1]
        ids = np.flatnonzero((body[u] >= 0) & (body[u] == body[v]))
        if len(ids):
            output[ids, 0] = shapely.covers(np.asarray(bodies, object)[body[u[ids]]],
                                          shapely.linestrings(xy[pairs[ids]]))
    dx, dy = (xy[v] - xy[u]).T
    norm = dx * dx + dy * dy
    point_tree = shapely.STRtree(points)
    faults = geology["faults"]
    for fault in faults:
        ids = point_tree.query(fault, predicate="dwithin", distance=near)
        distance = np.full(len(xy), np.inf)
        arc = np.zeros(len(xy))
        tx, ty = np.zeros(len(xy)), np.zeros(len(xy))
        distance[ids] = shapely.distance(points[ids], fault)
        arc[ids] = shapely.line_locate_point(fault, points[ids])
        lo = shapely.line_interpolate_point(fault, np.maximum(0, arc[ids] - tangent))
        hi = shapely.line_interpolate_point(fault, np.minimum(fault.length, arc[ids] + tangent))
        tx[ids], ty[ids] = shapely.get_x(hi) - shapely.get_x(lo), shapely.get_y(hi) - shapely.get_y(lo)
        candidates = np.flatnonzero((distance[u] <= near) & (distance[v] <= near))
        i, j = u[candidates], v[candidates]
        ci = (dx[candidates] * tx[i] + dy[candidates] * ty[i]) ** 2 / np.maximum(norm[candidates] * (tx[i] ** 2 + ty[i] ** 2), 1e-20)
        cj = (dx[candidates] * tx[j] + dy[candidates] * ty[j]) ** 2 / np.maximum(norm[candidates] * (tx[j] ** 2 + ty[j] ** 2), 1e-20)
        keep = (ci >= alignment_min) & (cj >= alignment_min) & (np.abs(arc[i] - arc[j]) <= arc_limit)
        selected = candidates[keep]
        if len(selected):
            weight = np.minimum(ci[keep], cj[keep]) * np.exp(-(distance[u[selected]] + distance[v[selected]]) / decay)
            weight[shapely.crosses(shapely.linestrings(xy[pairs[selected]]), fault)] = 0
            output[selected, 1] = np.maximum(output[selected, 1], weight.astype(np.float32))
    if len(faults):
        tree = shapely.STRtree(faults)
        for start in range(0, len(pairs), 50000):
            hits = tree.query(shapely.linestrings(xy[pairs[start:start + 50000]]), predicate="crosses")
            output[start + np.unique(hits[0]), 2] = 1
    return output


def build_graph(frame, settings, geology):
    radius = float(settings["candidate_radius_m"])
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Candidate radius must be positive")
    xy = frame[["x", "y"]].to_numpy(dtype=float)
    if not np.isfinite(xy).all() or len(np.unique(xy, axis=0)) != len(xy):
        raise ValueError("Graph coordinates must be finite and unique")
    pairs = cKDTree(xy).query_pairs(radius, output_type="ndarray")
    if len(pairs):
        pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
    distance = np.linalg.norm(xy[pairs[:, 0]] - xy[pairs[:, 1]], axis=1) / radius
    relations = geological_relations(xy, pairs, geology, settings)
    attributes = np.column_stack((distance, relations)).astype(np.float32)
    edges = np.concatenate((pairs, pairs[:, ::-1])).T.astype(np.int64)
    return edges, np.tile(attributes, (2, 1))
