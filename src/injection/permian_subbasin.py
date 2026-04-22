"""
Permian sub-basin lookup for injection wells (console reporting only).

Uses ``permian_basin_boundaries/PermianBasin_boundary.shp`` (WGS84 polygons with
a ``Name`` field). Implemented with **pyshp** + **Shapely** only — it does **not**
import geopandas, so it stays compatible with older numpy/pandas stacks. Project
``requirements.txt`` pins geopandas separately for notebooks and mapping code.
Does not modify any DataFrames written to disk.

**Which CSV to use (GIST injection layout under ``--target-dir``, e.g. ``src/data``):**

- **Recommended:** ``disposal_well_b3_format.csv`` (TexNet) or
  ``rrc_disposal_well_b3_format.csv`` (RRC). These have B3 columns
  ``InjectionWellId``, ``SurfaceHoleLatitude``, ``SurfaceHoleLongitude`` — the
  defaults for this script.
- **Alternative:** raw ``disposal_well.csv`` / ``rrc_disposal_well.csv`` use
  TexNet-style headers; pass ``--preset texnet`` or ``--preset rrc_raw``.
- **After GIST processing:** ``gist_well_shallow.csv`` / ``gist_well_deep.csv``
  use ``ID`` instead of ``InjectionWellId``; pass ``--preset gist``.

Standalone examples (from ``src/injection``):

    python permian_subbasin.py ../data/disposal_well_b3_format.csv
    python permian_subbasin.py ../data/disposal_well.csv --preset texnet
"""

from __future__ import annotations

import argparse
import logging
import sys
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import shapefile
from shapely.geometry import Point, Polygon

logger = logging.getLogger(__name__)

# When a point could fall in overlapping polygons (unlikely), prefer this order.
_POLYGON_ORDER: Tuple[str, ...] = (
    "Delaware Basin",
    "Midland Basin",
    "Central Basin Platform",
)

_OUTSIDE_LABEL = "Outside"
_INVALID_LABEL = "Invalid coordinates"


def _default_shapefile_path() -> Path:
    return Path(__file__).resolve().parent / "permian_basin_boundaries" / "PermianBasin_boundary.shp"


def _pyshp_shape_to_polygon(shp: shapefile.Shape) -> Polygon:
    """Build a valid Shapely polygon from a pyshp shape (polygon / polygon Z)."""
    pts = shp.points
    if not pts:
        raise ValueError("Empty shape geometry")
    parts = list(shp.parts) + [len(pts)]
    rings: List[List[Tuple[float, float]]] = []
    for i in range(len(parts) - 1):
        chunk = pts[parts[i] : parts[i + 1]]
        rings.append([(float(p[0]), float(p[1])) for p in chunk])
    exterior = rings[0]
    holes = rings[1:] if len(rings) > 1 else []
    poly = Polygon(exterior, holes)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


@lru_cache(maxsize=4)
def _ordered_name_polygons(shapefile_path: Optional[str] = None) -> Tuple[Tuple[str, Polygon], ...]:
    """
    Load basin polygons once and return (Name, Shapely Polygon) tuples in priority order.
    """
    path = Path(shapefile_path) if shapefile_path else _default_shapefile_path()
    if not path.is_file():
        raise FileNotFoundError(f"Permian basin shapefile not found: {path}")

    reader = shapefile.Reader(str(path))
    field_names = [f[0] for f in reader.fields[1:]]
    if "Name" not in field_names:
        raise ValueError(f"Shapefile {path} has no 'Name' field; got {field_names}")
    name_idx = field_names.index("Name")

    by_name: dict[str, Polygon] = {}
    for i in range(len(reader)):
        shp = reader.shape(i)
        rec = reader.record(i)
        name = str(rec[name_idx]).strip()
        by_name[name] = _pyshp_shape_to_polygon(shp)

    ordered: List[Tuple[str, Polygon]] = []
    for name in _POLYGON_ORDER:
        if name in by_name:
            ordered.append((name, by_name[name]))
    # Any extra polygons not in _POLYGON_ORDER (after priority list)
    for name, poly in by_name.items():
        if name not in _POLYGON_ORDER:
            ordered.append((name, poly))
    return tuple(ordered)


def permian_basin_labels(
    lat: pd.Series,
    lon: pd.Series,
    *,
    shapefile_path: Optional[str] = None,
) -> pd.Series:
    """
    Return a per-row sub-basin label from the Permian boundary layer.

    Parameters
    ----------
    lat, lon : pd.Series
        Surface latitude / longitude (decimal degrees), aligned indices.
    shapefile_path : optional
        Override path to the .shp file (defaults next to this module).

    Returns
    -------
    pd.Series
        Same index as inputs. Values are shapefile ``Name`` strings,
        ``_OUTSIDE_LABEL``, or ``_INVALID_LABEL``.
    """
    if not lat.index.equals(lon.index):
        raise ValueError("lat and lon must share the same index")

    name_polys = _ordered_name_polygons(shapefile_path)
    n = len(lat)
    out = pd.Series([_OUTSIDE_LABEL] * n, index=lat.index, dtype=object)

    invalid = lat.isna() | lon.isna() | ((lat == 0.0) & (lon == 0.0))
    out.loc[invalid] = _INVALID_LABEL

    def classify(lo: float, la: float) -> str:
        pt = Point(lo, la)
        for name, poly in name_polys:
            if poly.covers(pt):
                return name
        return _OUTSIDE_LABEL

    for idx in lat.loc[~invalid].index:
        out.loc[idx] = classify(float(lon.loc[idx]), float(lat.loc[idx]))

    return out


def print_permian_basins_for_wells(
    injection_well_ids: pd.Series,
    lat: pd.Series,
    lon: pd.Series,
    *,
    shapefile_path: Optional[str] = None,
    log_summary: bool = True,
    summary_stream=None,
) -> None:
    """
    Print each well's InjectionWellId and assigned Permian sub-basin to stdout.

    Does not modify any dataframe. Optionally logs aggregate counts at INFO.
    If ``summary_stream`` is set (e.g. ``sys.stdout``), writes the summary there
    so it appears after per-well lines even when logging uses stderr.
    """
    labels = permian_basin_labels(lat, lon, shapefile_path=shapefile_path)
    if not injection_well_ids.index.equals(lat.index):
        raise ValueError("injection_well_ids must align with lat/lon index")

    #for wid, basin in zip(injection_well_ids.astype(str), labels):
    #    print(f"{wid}\t{basin}")

    if log_summary:
        vc = labels.value_counts()
        summary = ", ".join(f"{k}={v}" for k, v in vc.items())
        msg = f"Permian sub-basin summary ({len(labels)} wells): {summary}"
        if summary_stream is not None:
            print(msg, file=summary_stream)
        else:
            logger.info("%s", msg)


def resolve_well_columns(
    df: pd.DataFrame,
    *,
    preset: str = "auto",
    id_column: Optional[str] = None,
    lat_column: Optional[str] = None,
    lon_column: Optional[str] = None,
) -> Tuple[str, str, str]:
    """
    Pick well id / lat / lon column names for a well CSV.

    Presets match GIST injection filenames under ``--target-dir`` (e.g. ``src/data``):

    - **b3** — ``disposal_well_b3_format.csv``, ``rrc_disposal_well_b3_format.csv``
    - **texnet** — raw ``disposal_well.csv`` (TexNet API export)
    - **gist** — ``gist_well_shallow.csv`` / ``gist_well_deep.csv`` (uses ``ID``)

    **auto** tries b3, then texnet, then gist column sets.
    """
    custom = [id_column, lat_column, lon_column]
    if any(custom):
        if not all(custom):
            raise ValueError(
                "If any of --id-column / --lat-column / --lon-column is set, all three are required."
            )
        id_c, lat_c, lon_c = id_column, lat_column, lon_column
        missing = [c for c in (id_c, lat_c, lon_c) if c not in df.columns]
        if missing:
            raise ValueError(f"Missing column(s) in CSV: {missing}. Got: {list(df.columns)}")
        return id_c, lat_c, lon_c

    profiles: dict[str, Tuple[str, str, str]] = {
        "b3": ("InjectionWellId", "SurfaceHoleLatitude", "SurfaceHoleLongitude"),
        "texnet": ("Uicnumber", "SurfaceLatitude", "SurfaceLongitude"),
        "gist": ("ID", "SurfaceHoleLatitude", "SurfaceHoleLongitude"),
    }

    if preset == "auto":
        for label, trip in (
            ("b3", profiles["b3"]),
            ("texnet", profiles["texnet"]),
            ("gist", profiles["gist"]),
        ):
            if all(c in df.columns for c in trip):
                logger.info("Using column profile '%s': %s", label, trip)
                return trip
        raise ValueError(
            "Could not auto-detect id/lat/lon columns. "
            "Use --preset b3 | texnet | gist or pass --id-column/--lat-column/--lon-column. "
            f"Columns in file: {list(df.columns)}"
        )

    if preset not in profiles:
        raise ValueError(f"Unknown preset: {preset}")
    trip = profiles[preset]
    missing = [c for c in trip if c not in df.columns]
    if missing:
        raise ValueError(
            f"preset '{preset}' expects columns {trip}; missing {missing}. "
            f"Columns in file: {list(df.columns)}"
        )
    return trip


def main(argv: Optional[List[str]] = None) -> int:
    """CLI: classify wells in a CSV and print id + Permian sub-basin to stdout."""
    parser = argparse.ArgumentParser(
        description=(
            "Print each well's sub-basin (Delaware / Midland / Central Basin Platform / Outside) "
            "from surface coordinates. Does not modify the CSV."
        )
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Well table CSV (see module docstring for recommended files).",
    )
    parser.add_argument(
        "--preset",
        choices=("auto", "b3", "texnet", "gist"),
        default="auto",
        help="Column layout: b3=B3 injection file; texnet=raw disposal_well; gist=gist_well_*.csv; auto=detect.",
    )
    parser.add_argument("--id-column", help="Override well id column name.")
    parser.add_argument("--lat-column", help="Override latitude column name.")
    parser.add_argument("--lon-column", help="Override longitude column name.")
    parser.add_argument(
        "--shapefile",
        type=str,
        default=None,
        help="Optional path to PermianBasin_boundary.shp (default: packaged boundaries).",
    )
    args = parser.parse_args(argv)

    if not args.csv_path.is_file():
        print(f"Error: file not found: {args.csv_path}", file=sys.stderr)
        return 1

    if not logging.root.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(levelname)s: %(message)s",
            stream=sys.stdout,
            force=True,
        )

    df = pd.read_csv(args.csv_path, low_memory=False)
    try:
        id_c, lat_c, lon_c = resolve_well_columns(
            df,
            preset=args.preset,
            id_column=args.id_column,
            lat_column=args.lat_column,
            lon_column=args.lon_column,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    well_ids = df[id_c].astype(str).reset_index(drop=True)
    lat = pd.to_numeric(df[lat_c], errors="coerce").reset_index(drop=True)
    lon = pd.to_numeric(df[lon_c], errors="coerce").reset_index(drop=True)

    print_permian_basins_for_wells(
        well_ids,
        lat,
        lon,
        shapefile_path=args.shapefile,
        log_summary=True,
        summary_stream=sys.stdout,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
