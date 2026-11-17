"""Mandatory raw-score and actual-on-disk raster distribution gates."""
import csv
import json
from pathlib import Path
import numpy as np
import rasterio
from scipy.stats import rankdata
from .preprocess import require_metric_grid

QUANTILES = {"min": 0, "P05": .05, "P25": .25, "median": .5,
             "P75": .75, "P90": .9, "P95": .95, "max": 1}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def distribution(values):
    values = np.asarray(values, dtype=float).ravel()
    finite = values[np.isfinite(values)]
    result = dict(valid_cells=len(values), nonfinite_cells=int(len(values) - len(finite)))
    if not len(finite):
        result.update({name: None for name in QUANTILES})
        result.update({f"fraction_ge_{cut}": None for cut in [.5, .8, .9]})
        result.update(std=None, unique_fraction=None, fraction_near_max=None)
        return result
    result.update({name: float(np.quantile(finite, quantile)) for name, quantile in QUANTILES.items()})
    result.update({f"fraction_ge_{cut}": float(np.mean(finite >= cut)) for cut in [.5, .8, .9]})
    result.update(std=float(finite.std()), unique_fraction=len(np.unique(finite)) / len(finite),
                  fraction_near_max=float(np.mean(np.abs(finite - finite.max()) <= 1e-6)))
    return result


def audit_values(values, kind="raw"):
    row = distribution(values)
    failures = []
    if not row["valid_cells"] or row["nonfinite_cells"]:
        failures.append("empty_or_nonfinite")
    if row["min"] is not None:
        if row["min"] < 0 or row["max"] > 1:
            failures.append("outside_unit_interval")
        if row["std"] < .01 or row["P95"] - row["P05"] < .05:
            failures.append("near_constant")
        if row["unique_fraction"] < .01 or row["fraction_near_max"] > .05:
            failures.append("score_saturation_or_ties")
        if any(row[f"fraction_ge_{cut}"] > upper for cut, upper in [(.5, .9), (.8, .5), (.9, .25)]):
            failures.append("excessive_high_score_area")
        if kind == "percentile":
            for cut, lower, upper in [(.5, .45, .55), (.8, .15, .25), (.9, .05, .15)]:
                if not lower <= row[f"fraction_ge_{cut}"] <= upper:
                    failures.append(f"unexpected_percentile_area_{cut}")
            if not .04 <= row["P05"] <= .08 or not .45 <= row["median"] <= .55 or not .90 <= row["P95"] <= .98:
                failures.append("unexpected_percentile_quantiles")
    row.update(kind=kind, passed=not failures, failures=";".join(failures))
    return row


def write_audit(folder, name, row):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / f"{name}.json", row)
    with (folder / f"{name}.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def export_percentile(template, rows, cols, raw_scores, folder):
    """Refuse invalid raw outputs before ranking; reopen and read the actual TIFF.

    Fractions represent area in the uniformly spaced projected grid. No
    geographic reprojection or interpolation is applied to the published map.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    raw = audit_values(raw_scores)
    write_audit(folder, "raw_score_distribution", raw)
    if not raw["passed"]:
        raise ValueError(f"Raw score gate failed: {raw['failures']}; investigate/refit before export")
    final = folder / "relative_prospectivity_percentile.tif"
    candidate = folder / "candidate_relative_prospectivity_percentile.tif"
    if final.exists() or candidate.exists():
        raise FileExistsError("Prediction raster already exists; use a fresh run directory")
    # Midranks preserve ties rather than manufacturing artificial variation.
    percentile = (rankdata(raw_scores, method="average") - .5) / len(raw_scores)
    rows, cols = np.asarray(rows), np.asarray(cols)
    with rasterio.open(template) as src:
        require_metric_grid(src)
        expected_rows, expected_cols = np.nonzero(src.read_masks(1) > 0)
        if not np.array_equal(np.column_stack((rows, cols)), np.column_stack((expected_rows, expected_cols))):
            raise ValueError("Prediction must cover the full valid template mask in row-major order")
        if len(raw_scores) != len(rows):
            raise ValueError("Prediction length mismatch")
        profile = dict(driver="GTiff", width=src.width, height=src.height, count=1,
                       dtype="float32", crs=src.crs, transform=src.transform,
                       nodata=-9999.0, compress="deflate")
        raster = np.full(src.shape, -9999.0, np.float32)
        raster[rows, cols] = percentile.astype(np.float32)
    with rasterio.open(candidate, "w", **profile) as dst:
        dst.write(raster, 1)
        dst.set_band_description(1, "Relative prospectivity area percentile; NOT occurrence probability")
        dst.update_tags(score_semantics="relative_area_percentile_not_probability",
                        calibration="none", area_basis="equal_projected_grid_cell_area")
    # Mandatory actual disk read, never substitute the array used for writing.
    with rasterio.open(candidate) as src:
        values = src.read(1, masked=True).compressed()
        row = audit_values(values, "percentile")
        cell_area = abs(src.transform.a * src.transform.e - src.transform.b * src.transform.d)
        row.update(raster=final.name if row["passed"] else candidate.name,
                   valid_projected_area_m2=float(cell_area * len(values)))
    write_audit(folder, "raster_distribution", row)
    if not row["passed"]:
        raise ValueError(f"Raster gate failed: {row['failures']}; candidate is not a release")
    candidate.rename(final)
    return final
