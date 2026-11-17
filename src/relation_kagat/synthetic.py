"""Generate a wholly artificial fixture at runtime; no research data is bundled."""
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
import yaml
from rasterio.transform import from_origin
from scipy.spatial import cKDTree
from .preprocess import clr


def make_demo(folder, config_template, epochs=3):
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    private = folder / "private"
    private.mkdir()
    rng = np.random.default_rng(12345)
    side = 40
    row, col = np.indices((side, side))
    row, col = row.ravel(), col.ravel()
    x, y = (col + .5) * 1000, (side - row - .5) * 1000
    positive = np.zeros(side * side, np.int64)
    for br in range(4):
        for bc in range(4):
            rr, cc = br * 10 + rng.integers(4, 7), bc * 10 + rng.integers(4, 7)
            positive[rr * side + cc] = 1
    distance = cKDTree(np.column_stack((x[positive == 1], y[positive == 1]))).query(np.column_stack((x, y)))[0]
    signal = np.exp(-distance / 1200)
    chemistry = clr(np.exp(np.column_stack((signal + rng.normal(0, .1, len(x)),
                                            .7 * signal + rng.normal(0, .1, len(x)),
                                            rng.normal(0, 1, len(x))))))
    frame = pd.DataFrame(dict(node_id=[f"synthetic_{i}" for i in range(len(x))], row=row, col=col,
                              known_positive=positive, stratigraphic_age=col // 10,
                              geochem_1_clr=chemistry[:, 0], geochem_2_clr=chemistry[:, 1],
                              geochem_3_clr=chemistry[:, 2],
                              aeromagnetic_F3=signal + rng.normal(0, .2, len(x)),
                              distance_to_intrusion_m=np.abs(x - 20000)))
    frame.to_csv(private / "nodes.csv", index=False)
    with rasterio.open(private / "grid.tif", "w", driver="GTiff", height=side, width=side,
                       count=1, dtype="uint8", crs="EPSG:3857", nodata=0,
                       transform=from_origin(0, side * 1000, 1000, 1000)) as dst:
        dst.write(np.ones((side, side), np.uint8), 1)
        dst.update_tags(data_origin="wholly_synthetic_software_test_not_research_data")
    geology = dict(crs="EPSG:3857", faults=[
        dict(id="synthetic_fault", inferred=False, wkt="LINESTRING (20000 0, 20000 40000)"),
        dict(id="synthetic_inferred", inferred=True, wkt="LINESTRING (0 0, 40000 40000)")],
        igneous_bodies=[
            dict(id="synthetic_body_A", wkt="POLYGON ((0 0, 20000 0, 20000 40000, 0 40000, 0 0))"),
            dict(id="synthetic_body_B", wkt="POLYGON ((20000 0, 40000 0, 40000 40000, 20000 40000, 20000 0))")])
    (private / "geology.local.yaml").write_text(yaml.safe_dump(geology), encoding="utf-8")
    cfg = yaml.safe_load(Path(config_template).read_text(encoding="utf-8"))
    cfg["paths"] = dict(nodes="private/nodes.csv", template="private/grid.tif",
                         geology="private/geology.local.yaml", work="run")
    cfg["training"].update(epochs=int(epochs), device="cpu")
    path = folder / "demo.local.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path
