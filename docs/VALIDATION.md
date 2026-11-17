# Method validation / 方法验证

Version 0.2 was checked on 2026-11-17 with Python 3.12 and PyTorch 2.7.1 on CPU.
All fixtures are generated artificially at runtime. No original research data,
GIS layers, sample identities, trained weights or research results are required.

0.2版验证只使用人工数据及原项目Python源码，不读取真实研究数据或坐标。

## Reproduce the checks

```bash
python -B -m unittest discover -s tests -v
python -B tools/check_public_tree.py
```

For a separate checkout of the original source, the optional reference check is:

```bash
python -B tools/check_method_reference.py --original-root ORIGINAL_SOURCE_DIRECTORY
```

The reference tool selects original Python function bodies through AST, supplies
synthetic data, and compares them to the public implementation. It does not run
the original entry points or load original NPZ, GIS or checkpoint files.

## Original-source numerical comparisons

| Method | Reference source | Synthetic comparison |
|---|---|---|
| Assay CLR, invalid sentinel and nonpositive replacement | `data_process/workflow.py:transform_geochemistry` | Maximum absolute difference 0 |
| Two-stage IDW, including unsupported targets | `data_process/workflow.py:idw_grid` | Maximum float32 difference 0 |
| Training-fold median/IQR, vocabulary and three missing indicators | `model/training_pipeline.py:prepare_features` and `model/data_features.py:encode_categories` | Maximum absolute difference 0 |
| Fixed background center sampling and outer partitions | `model/center_sampling.py:select_negative_centers` | Identical center partitions |
| Grouped positive centers and shuffled negative three-fold split | Inline fold block in `model/tune_spatial_random_models.py:main` | Identical expanded training/validation indices in all three folds |
| Same/Along/Cross geometry | `scripts/geological_edge_rules.py:relations`, alignment `min` | Maximum absolute difference 0 across 492 artificial undirected edges |
| Independent geological-channel forward pass | `outputs/geological_relations/code/last_model.py:FinalRoundKAN.forward` | Maximum absolute logit difference 0 with identical parameters and artificial inputs |

The model comparison enables nonzero geological-channel weights. The original
eight-column reference edge layout is adapted explicitly to the public four-column
interface, retaining distance in the backbone and independent Same/Along/Cross
channels. This checks method arithmetic; it is not a historical checkpoint replay.

上述对照覆盖预处理、抽样、三折索引、几何关系和模型计算。模型对照主动启用了非零地质通道，
没有用初始化时的零通道掩盖差异。公开默认特征数量、dropout等可调设置不声称与每个历史实验相同。

## Regression coverage

- B-splines against SciPy and finite gradients.
- Full-graph versus exact one-hop target inference, including isolated nodes.
- Per-fold fitting; held-out extreme values and unseen categories do not change
  training preprocessing; degenerate IQR and all-missing training columns.
- Partial-observation CLR, explicit replacement, finite-support IDW and duplicate
  exact-point averaging; CLR is applied before interpolation.
- Endpoint touching versus true crossing, along-fault angle/buffer/arc limits,
  inferred-fault exclusion, strict polygon interior and whole-segment coverage.
- Every development node validated once; expanded patches do not cross folds;
  the outer test is predicted once after final refit and never during CV.
- Independent geological channels affect predictions and receive gradients.
- Changed input/geometry/configuration and changed CV artifacts are rejected.
- Actual GeoTIFF readback with required quantiles and area fractions; raw-score
  failure stops map release and writes `failed`.
- Source-only publication scan and SHA-256 inventory integrity.

Release-state tests use controlled raw scores after real synthetic CV/refit to
isolate acceptance-gate behavior. The synthetic CLI demo trains real models but
does not certify a geological prospectivity map. No GPU reproducibility or
real-data scientific performance is asserted by these software checks.

## Environment

The checked environment used NumPy 2.2.6, pandas 2.3.0, SciPy 1.15.3,
scikit-learn 1.8.0, torch-geometric 2.7.0, Rasterio 1.4.4, Shapely 2.1.1
and PyYAML 6.0.2. Installation checks reuse these installed dependencies.

A local GIS installation may inject incompatible `PROJ_LIB`/`PROJ_DATA` variables.
Use Rasterio's matching projection database in the active test process; do not
copy machine-specific paths into public configuration or change system settings.
