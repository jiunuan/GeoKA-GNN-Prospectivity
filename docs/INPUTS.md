# Private input contract

[简体中文](INPUTS.zh-CN.md)

Data owners supply all files locally. Use an appropriate, low-distortion metric
projected CRS shared by the grid and geological geometries. The research 3x3
sampling protocol requires an **unrotated 1 km by 1 km** grid.

## Nodes and template

| Input | Contract |
|---|---|
| `nodes.csv` | One row per valid cell; unique string `node_id`; integer `row`, `col`; binary `known_positive`; configured continuous and categorical features |
| `grid.tif` | Single-band metric projected template and valid-cell mask, covered exactly once by the node table |
| `geology.local.yaml` | Explicit CRS plus mapped fault parts and actual igneous-body polygon parts, as below |

`known_positive=1` denotes an independent occurrence center, and 0 an unlabelled
background candidate. Do not pre-expand occurrence labels into neighboring cells.
The program groups centers before splitting and expanding into 3x3 patches.
Supply all known centers, including eventual validation/test centers, before
background exclusion. Account for known occurrences outside the grid when
preparing inputs; they are not available to the program's center buffer.

Optional `x`/`y` coordinates must match template centers within 0.01 m; otherwise
coordinates are derived. Labels, coordinates and identifiers cannot be predictors.
Stratigraphic age can be a categorical node predictor. It does not define the
same-body edge. `intrusive_body_id` is not required: actual polygons define Same.

Category IDs are finite integers or missing. Raw category zero is legitimate when
observed in training. Missing/unseen categories map to embedding index zero.
`features.geochemistry` lists chemistry columns; `features.aeromagnetic` identifies
one magnetic column. Both refer to configured continuous features and determine
the three research availability indicators and chemistry-availability gate.

## Geological geometries

The following is an artificial example. Each record has a unique string `id`.
Reproject private geometry explicitly to the template CRS before providing it.

```yaml
crs: EPSG:32646
faults:
  - id: mapped_fault_1
    inferred: false
    wkt: LINESTRING (500000 4700000, 510000 4700000)
  - id: inferred_fault_2
    inferred: true
    wkt: LINESTRING (500000 4700000, 500000 4710000)
igneous_bodies:
  - id: body_part_1
    wkt: POLYGON ((500000 4700000, 510000 4700000, 510000 4710000, 500000 4710000, 500000 4700000))
```

- An explicit boolean `inferred` is required for each fault. Inferred faults
  contribute to neither Along nor Cross.
- LineString/MultiLineString and Polygon/MultiPolygon are supported. Multipart
  geometries are split; disconnected fault/body parts remain separate.
- Bodies represent individually mapped igneous/intrusive bodies, not broad
  lithology classes. Invalid geometries fail. A node strictly inside more than
  one body also fails instead of silently selecting an owner.
- Boundary points are not strictly inside. Even when both endpoints are inside
  one concave body, a segment leaving that polygon does not qualify as Same.
- If no such geological features are mapped, provide `[]` for the corresponding
  list. An empty list must not stand in for missing input preparation.

## Assay and raster preprocessing

```python
from relation_kagat.preprocess import interpolate_assays, aligned_raster_table

# assay_xy: metric assay positions; assays: samples-by-elements raw matrix.
# CLR at sample positions, followed by per-element staged 2 km / 5 km IDW:
# interpolated = interpolate_assays(assay_xy, assays, grid_xy, cfg["preprocessing"])

# Existing rasters must share CRS, shape and affine transform. No resampling.
table = aligned_raster_table("data/private/grid.tif", {
    "geochem_1_clr": "data/private/geochem_1_clr.tif",
    "known_positive": "data/private/positive_centers.tif",
    "stratigraphic_age": "data/private/age_ids.tif",
})
# Add all other configured features before saving.
table.to_csv("data/private/nodes.csv", index=False)
```

CLR converts configured invalid sentinels and infinities to missing. Explicit
`half_min_positive` replaces nonpositive values with half the corresponding
element's smallest positive assay. The row mean log uses observed components;
missing components remain NaN. IDW minimum support counts finite observations.
Duplicate coordinates follow the original algorithm: average selected zero-distance
observations on exact hits. Document any aggregation performed beforehand.

`prepare` starts at the aligned table and does not rerun IDW. Keep `features.clr: []`
for already transformed inputs. Configured `log1p` runs before fold normalization.
Each fold learns median/IQR and category vocabulary independently. Missing
standardized entries become zero. Output layout is continuous features, category
indices, chemistry missing fraction, chemistry complete-availability flag, and
magnetic missing flag. Entirely missing training columns use center 0/scale 1 and
are recorded in preprocessing metadata.

Proprietary source-field extraction, GIS mapping and rasterization are local
data-owner tasks. This repository supplies the numerical methods, alignment
checks and a direct interface to locally prepared geological geometry.
