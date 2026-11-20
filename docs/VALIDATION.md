# Method validation / 方法验证

The paper-aligned method was checked on 2026-11-18 with Python 3.12 and PyTorch 2.7.1 on CPU.
All fixtures are generated artificially at runtime. No original research data,
GIS layers, sample identities, trained weights or research results are required.

验证只使用人工数据及原项目Python源码，不读取真实研究数据或坐标。

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
| Joint four-attribute relation encoding, attention and message arithmetic | `model/relation_ka_gat.py:RelationKAGATV2` joint functions, assembled into the Section 2.3 aggregation | Maximum absolute logit difference 0 with identical parameters and artificial inputs |

All four edge attributes enter one relation encoder. The source comparison uses
the original joint relation, attention and message functions with a four-column
interface. A separate regression evaluates the paper's neighbor sum directly,
including soft gates, multiple incoming edges, isolated nodes and the residual
LayerNorm update. There are eight KAN modules and no independent geological
residual channels. These are method checks, not historical checkpoint replays.

上述对照覆盖预处理、抽样、三折索引、几何关系及论文统一关系编码和聚合计算。
公开默认特征数量和可调训练配置由使用者按数据设置，不声称人工演示复现正文或附录中的历史指标、参数量及计时。

## Classification threshold

Section 4.1 is implemented by maximizing Youden J over pooled development OOF
scores after the three folds' checkpoints have been selected. J is
sensitivity + specificity - 1. Among tied maxima, the highest finite cutoff is selected.
The prediction rule is score >= cutoff. The chosen threshold and objective are
saved in `cross_validation/threshold_selection.json` and the CV summary, with the
cutoff also stored in the final checkpoint and test metrics.
The `cross-validate` and `train` commands print the selected cutoff, including
when the cross-validation results are reused.

The outer test is excluded from OOF coverage and threshold selection. CV reuse
recomputes the threshold from frozen OOF scores and rejects a changed selection.
The fixed cutoff is then used to evaluate the final model on the outer test.
Fold classification metrics use the pooled development threshold and remain
development summaries. Raw-score thresholds do not apply to area-percentile maps.

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
- Every geological attribute affects the joint relation and receives gradients.
- The eight-module joint architecture matches the Section 2.3 equations.
- Youden selection matches exhaustive confusion counts, handles tied and constant
  scores, rejects invalid inputs and persists unchanged from CV to the final test.
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
