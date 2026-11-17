# 私有输入约定

[English](INPUTS.md)

所有文件由数据所有者在本地提供，不随源码发布。距离和几何运算使用与网格一致的米制投影坐标系。
建议选择研究区适用、畸变较小的投影。算法保留原项目1 km网格上3×3扩展的定义，因此节点表
必须覆盖一个**无旋转、1 km×1 km**网格的全部有效像元。

## 节点表和模板

|文件|要求|
|---|---|
|`nodes.csv`|每个有效网格一行；唯一字符串`node_id`；整数`row`、`col`；二值`known_positive`；配置指定的连续与类别特征|
|`grid.tif`|单波段、米制投影、有效像元掩膜；节点表必须覆盖全部有效像元且不得重复|
|`geology.local.yaml`|显式CRS、断层部件及实际火成岩体面部件，格式见下文|

`known_positive=1`表示独立矿点中心，0表示未标记背景候选；请勿预先将邻格也标为正中心。
程序统一执行中心分组、划分及3×3扩展。必须提供所有已知中心，含后续验证与测试中心；
背景排除按这些中心计算，网格外矿点需在准备输入时单独检查其缓冲影响。

可选`x`、`y`必须与像元中心相差不超过0.01 m；否则由模板计算。坐标、行列号、ID及标签
禁止作为预测特征。同地层年代可以作为节点类别特征，但不作为本版的同岩体边关系。
`intrusive_body_id`不是必需列，同岩体关系由真实多边形计算。

类别ID为有限整数或缺失。训练中实际出现的0也是合法类别；缺失及训练中未见的类别映射到
嵌入索引0。`features.geochemistry`列出化探连续列，`features.aeromagnetic`指定航磁连续列，
两者均须属于`features.continuous`，用于原项目的三个缺测指标和化探可用性门控。

## 地质几何文件

以下只是人工示例；实际几何留在本地。每条记录有唯一字符串`id`，坐标须已与模板CRS一致：

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

- 每条断层必须显式提供布尔`inferred`；推测断层不参与Along或Cross。
- 支持LineString/MultiLineString和Polygon/MultiPolygon；多部件拆开计算，不能跨不连通断层
  计算沿线关系，也不能将不连通岩体当作一个连续面。
- 岩体面应表示实际制图的独立火成／侵入岩体部件，而非粗粒度岩性类别。无效几何直接报错；
  一个节点严格落入多个岩体面时也报错，不自动选择归属。
- 岩体边界点不满足严格内部条件；即使两端同属一个凹面，连线离开该面也不产生Same关系。
- 确实没有制图断层或火成岩体时，相应列表写为`[]`；不可用空列表代替尚未提供的数据。

## 化探与栅格预处理

```python
from relation_kagat.preprocess import interpolate_assays, aligned_raster_table

# assay_xy为米制采样点坐标，assays为“采样点×元素”原始化探矩阵。
# 按配置recipe先在采样点做CLR，再逐元素做2 km / 5 km分级IDW。
# interpolated = interpolate_assays(assay_xy, assays, grid_xy, cfg["preprocessing"])

# 对已有栅格只提取，不静默重采样。CRS、形状及仿射变换必须完全对齐。
table = aligned_raster_table("data/private/grid.tif", {
    "geochem_1_clr": "data/private/geochem_1_clr.tif",
    "known_positive": "data/private/positive_centers.tif",
    "stratigraphic_age": "data/private/age_ids.tif",
})
# 补齐其他配置特征后保存。
table.to_csv("data/private/nodes.csv", index=False)
```

原始CLR将配置的无效哨兵和无穷值视作缺测；非正值只按明确选择的`half_min_positive`处理，
每个元素使用其最小正值的一半。部分组分缺失时只对已观测组分求对数均值，缺失项保持NaN。
IDW最少点数按有效观测条数计算；重复位置沿用原算法，精确命中时对选中零距离观测取均值。
如需先聚合重复采样，应由数据所有者明确记录。

`prepare`从已对齐节点表开始，不会再次执行IDW。已做CLR的特征应设`features.clr: []`，
避免重复变换。可配置的`log1p`在分折标准化前执行。每折独立拟合中位数／IQR及类别词表，
标准化缺失值置0，输出顺序为连续列、类别嵌入索引、化探缺失比例、化探完整可用标志、航磁缺失标志。
训练折完全缺失的连续列使用中心0、尺度1，并记录在预处理元数据中。

专有数据库字段提取、源GIS制图和栅格化由数据所有者在本地完成；本仓库提供上述数值方法、
对齐检查及直接读取本地几何的接口。
