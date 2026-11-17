"""Freeze center-based sampling, an outer holdout and three development folds."""
import hashlib
import json
import numpy as np
from scipy.spatial import cKDTree
from sklearn.model_selection import KFold


def expand_centers(frame, centers):
    """Expand centers once to their 3x3 cells; retain valid grid cells only."""
    cells = frame[["row", "col"]].to_numpy(np.int64)
    lookup = {tuple(cell): i for i, cell in enumerate(cells)}
    expanded = {lookup[(int(cells[i, 0]) + dr, int(cells[i, 1]) + dc)]
                for i in centers for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                if (int(cells[i, 0]) + dr, int(cells[i, 1]) + dc) in lookup}
    return np.asarray(sorted(expanded), np.int64)


def center_components(frame, centers):
    """Keep positive centers whose expanded patches overlap in one component."""
    centers = np.asarray(centers, np.int64)
    cells = frame.iloc[centers][["row", "col"]].to_numpy(np.int64)
    parent = np.arange(len(centers))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return int(i)
    for left in range(len(centers)):
        for right in range(left + 1, len(centers)):
            if np.max(np.abs(cells[left] - cells[right])) <= 2:
                a, b = find(left), find(right)
                if a != b:
                    parent[b] = a
    groups = {}
    for index, node in enumerate(centers):
        groups.setdefault(find(index), []).append(int(node))
    return [np.asarray(group, np.int64) for group in groups.values()]


def _outer_positive_split(frame, positive, count, seed):
    groups = center_components(frame, positive)
    rng = np.random.default_rng(seed)
    groups = [groups[i] for i in rng.permutation(len(groups))]
    choices = {0: []}
    for index, group in enumerate(groups):
        for total, selected in list(choices.items())[::-1]:
            candidate = total + len(group)
            if candidate <= count and candidate not in choices:
                choices[candidate] = [*selected, index]
    if count not in choices:
        raise ValueError("Requested outer test count would split overlapping positive patches")
    selected = set(choices[count])
    test = np.sort(np.concatenate([g for i, g in enumerate(groups) if i in selected]))
    development = np.sort(np.concatenate([g for i, g in enumerate(groups) if i not in selected]))
    return development, test


def _negative_centers(frame, positive, settings):
    xy = frame[["x", "y"]].to_numpy(float)
    cells = frame[["row", "col"]].to_numpy(np.int64)
    buffer = float(settings["exclusion_buffer_m"])
    spacing = int(settings["minimum_negative_spacing_cells"])
    if not np.isfinite(buffer) or buffer < 3000 or spacing < 3:
        raise ValueError("Require >=3 km positive-center exclusion and >=3-cell negative spacing")
    distance = cKDTree(xy[positive]).query(xy)[0]
    lookup = {tuple(cell): i for i, cell in enumerate(cells)}
    positive_nodes = set(expand_centers(frame, positive))
    candidates = []
    for node in np.flatnonzero((frame.known_positive.to_numpy() == 0) & (distance >= buffer)):
        row, col = cells[node]
        patch = [lookup.get((row + dr, col + dc)) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]
        if None not in patch and not positive_nodes.intersection(patch):
            candidates.append(int(node))
    chosen = []
    rng = np.random.default_rng(int(settings["seed"]))
    for node in rng.permutation(candidates):
        if chosen and np.any(np.max(np.abs(cells[chosen] - cells[node]), axis=1) < spacing):
            continue
        chosen.append(int(node))
        if len(chosen) == len(positive):
            break
    if len(chosen) != len(positive):
        raise ValueError(f"Insufficient nonoverlapping background patches: {len(chosen)}/{len(positive)}; no resampling")
    return np.asarray(chosen, np.int64)


def three_folds(frame, positive, negative, seed):
    """Original grouped-positive / shuffled-negative three-fold algorithm."""
    groups = center_components(frame, positive)
    if len(groups) < 3 or len(negative) < 3:
        raise ValueError("Three folds require >=3 independent positive components and background centers")
    rng = np.random.default_rng(seed)
    assignments, sizes = [[], [], []], [0, 0, 0]
    for group in sorted([groups[i] for i in rng.permutation(len(groups))], key=len, reverse=True):
        fold = int(np.argmin(sizes))
        assignments[fold].extend(group.tolist())
        sizes[fold] += len(group)
    result = []
    for number, (nt, nv) in enumerate(KFold(3, shuffle=True, random_state=seed).split(negative)):
        pv = np.sort(assignments[number])
        pt = np.setdiff1d(positive, pv)
        train = np.union1d(expand_centers(frame, pt), expand_centers(frame, negative[nt]))
        validation = np.union1d(expand_centers(frame, pv), expand_centers(frame, negative[nv]))
        if np.intersect1d(train, validation).size:
            raise ValueError("Expanded training/validation patches overlap")
        result.append(dict(number=number, train=train.tolist(), validation=validation.tolist(),
                           positive_train_centers=pt.tolist(), positive_validation_centers=pv.tolist(),
                           negative_train_centers=negative[nt].tolist(),
                           negative_validation_centers=negative[nv].tolist()))
    return result


def fixed_samples(frame, settings, manifest_path):
    serialized = frame.to_csv(index=False, float_format="%.17g", lineterminator="\n")
    specification = dict(schema="center_three_fold_v2", settings=settings,
                         nodes_sha256=hashlib.sha256(serialized.encode()).hexdigest())
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if saved["specification"] != specification:
            raise ValueError("Frozen samples do not match inputs/settings; use a new work directory")
        return saved
    positive = np.flatnonzero(frame.known_positive.to_numpy() == 1)
    fraction = float(settings["test_fraction"])
    if not 0 < fraction < 1 or len(positive) < 4:
        raise ValueError("Need >=4 positive centers and test_fraction in (0, 1)")
    test_count = settings.get("test_positive_centers")
    test_count = max(1, int(round(len(positive) * fraction))) if test_count is None else int(test_count)
    if not 1 <= test_count <= len(positive) - 3:
        raise ValueError("Outer holdout must leave >=3 development positive centers")
    pdev, ptest = _outer_positive_split(frame, positive, test_count, int(settings["split_seed"]))
    negative = _negative_centers(frame, positive, settings)
    # Sort within the frozen outer partitions, as the original sampler does.
    ndev, ntest = np.sort(negative[:len(pdev)]), np.sort(negative[len(pdev):])
    positive_nodes, negative_nodes = expand_centers(frame, positive), expand_centers(frame, negative)
    if np.intersect1d(positive_nodes, negative_nodes).size:
        raise ValueError("Expanded positive/background labels conflict")
    development = np.union1d(expand_centers(frame, pdev), expand_centers(frame, ndev))
    test = np.union1d(expand_centers(frame, ptest), expand_centers(frame, ntest))
    if np.intersect1d(development, test).size:
        raise ValueError("Outer development/test patches overlap")
    folds = three_folds(frame, pdev, ndev, int(settings["cv_seed"]))
    if any(np.intersect1d(f["validation"], test).size or np.intersect1d(f["train"], test).size for f in folds):
        raise ValueError("Outer test leaked into development folds")
    validations = [i for fold in folds for i in fold["validation"]]
    if len(validations) != len(set(validations)) or set(validations) != set(development):
        raise ValueError("Each development node must be validated exactly once")
    result = dict(specification=specification, positive_centers=positive.tolist(),
                  negative_centers=negative.tolist(), positive_nodes=positive_nodes.tolist(),
                  negative_nodes=negative_nodes.tolist(),
                  positive_development_centers=pdev.tolist(), positive_test_centers=ptest.tolist(),
                  negative_development_centers=ndev.tolist(), negative_test_centers=ntest.tolist(),
                  splits=dict(development=development.tolist(), test=test.tolist()), folds=folds,
                  semantics="center_level_1_to_1_case_control;3x3_expansion;background_not_confirmed_barren")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result
