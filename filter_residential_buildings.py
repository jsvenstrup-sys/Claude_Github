#!/usr/bin/env python3
"""
Filter Illinois building footprints to residential buildings within the CMAP study area.

Only buildings that spatially overlap a CMAP LUI20 residential land-use polygon are kept.
"Residential" = domiciles where people sleep (apartments, condos, single-family, duplexes).

CMAP LUI20 residential codes used (field: LANDUSE):
  1111 – Single-Family Detached
  1112 – Single-Family Attached (townhomes, no shared entryway)
  1113 – Single-Family Attached to Non-Residential Use
  1121 – Two-Family (Duplex) Detached
  1122 – Two-Family (Duplex) Attached
  1130 – Multi-Family (apartments, condos, two-flats, SROs)

Coverage: northeastern Illinois 7-county CMAP metro only (where CMAP data exists).

Sources:
  IL DNR Building Footprints:
    https://geoservices3.dnr.illinois.gov/arcgis/rest/services/statewide_building_footprints/MapServer/0
  CMAP LUI20 FeatureServer:
    https://services5.arcgis.com/LcMXE3TFhi1BSaCY/arcgis/rest/services/LUI20_geodatabase_v1_CMAP/FeatureServer

Usage:
  python filter_residential_buildings.py [--output OUTPUT_DIR] [--chunk-size N]

  On Windows, if 'python' is not on PATH use the full installer path:
  "C:/Users/HP z440/AppData/Local/Programs/Python/Python312/python.exe" filter_residential_buildings.py [--output OUTPUT_DIR] [--chunk-size N]

  Install dependencies first (same full path):
  "C:/Users/HP z440/AppData/Local/Programs/Python/Python312/python.exe" -m pip install geopandas pandas requests fiona pyproj shapely

Output:
  <output_dir>/il_residential_buildings_cmap.shp  (and companion files)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Service endpoints
# ---------------------------------------------------------------------------
BLDG_BASE = (
    "https://geoservices3.dnr.illinois.gov/arcgis/rest/services"
    "/statewide_building_footprints/MapServer/0"
)
CMAP_SERVER = (
    "https://services5.arcgis.com/LcMXE3TFhi1BSaCY/arcgis/rest/services"
    "/LUI20_geodatabase_v1_CMAP/FeatureServer"
)

# ---------------------------------------------------------------------------
# CMAP residential land-use codes (domiciles where people sleep)
# ---------------------------------------------------------------------------
RESIDENTIAL_CODES = (1111, 1112, 1113, 1121, 1122, 1130)

# Output CRS – WGS84 for broad compatibility
OUT_CRS = "EPSG:4326"
# Working CRS – Illinois State Plane East (NAD83, feet) for accurate overlay
WORK_CRS = "EPSG:3435"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get(url: str, params: dict, retries: int = 4, timeout: int = 180) -> dict:
    """GET with exponential-backoff retry and basic error handling."""
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict) and "error" in data:
                raise RuntimeError(f"ArcGIS error: {data['error']}")
            return data
        except Exception as exc:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f"    [retry {attempt + 1}/{retries - 1}] {exc}  – waiting {wait}s …")
            time.sleep(wait)


def layer_meta(base_url: str) -> dict:
    """Fetch layer metadata JSON."""
    return _get(base_url, {"f": "json"}, timeout=30)


def find_cmap_layer(server_url: str, target_name_fragment: str = "") -> str:
    """Return the base URL of the first polygon layer in a FeatureServer."""
    meta = _get(server_url, {"f": "json"}, timeout=30)
    layers = meta.get("layers", [])
    if not layers:
        raise RuntimeError(
            f"No layers found at {server_url}. Response: {meta}"
        )
    # Prefer a layer whose name matches the fragment (case-insensitive)
    for layer in layers:
        if target_name_fragment.lower() in layer.get("name", "").lower():
            return f"{server_url}/{layer['id']}"
    # Fall back to first layer
    return f"{server_url}/{layers[0]['id']}"


def detect_landuse_field(meta: dict) -> str:
    """Return the LANDUSE field name from layer metadata, with fallbacks."""
    fields = {f["name"].upper(): f["name"] for f in meta.get("fields", [])}
    for candidate in ("LANDUSE", "LUI_CODE", "LU_CODE", "LAND_USE", "LUCODE"):
        if candidate in fields:
            return fields[candidate]
    raise RuntimeError(
        f"Cannot detect land-use field. Available fields: {list(fields.values())}"
    )


def fetch_all_features(
    base_url: str,
    where: str,
    out_fields: str = "*",
    out_sr: int = 4326,
    geometry_filter: dict | None = None,
    page_size: int = 2000,
) -> gpd.GeoDataFrame:
    """
    Download all matching features via OID-based pagination.

    OID pagination is more reliable than offset pagination for large tables.
    """
    meta = layer_meta(base_url)
    oid_field = meta.get("objectIdField", "OBJECTID")
    server_max = meta.get("maxRecordCount", 1000)
    page_size = min(page_size, server_max)

    # Build base params
    base_params: dict = {
        "where": where,
        "outFields": out_fields,
        "outSR": out_sr,
        "returnGeometry": "true",
        "f": "geojson",
    }
    if geometry_filter:
        base_params.update(geometry_filter)

    # ── Get all matching OIDs first ──────────────────────────────────────────
    oid_params = {**base_params, "returnIdsOnly": "true", "f": "json"}
    oid_params.pop("outFields", None)
    oid_params.pop("returnGeometry", None)
    oid_data = _get(base_url + "/query", oid_params, timeout=120)
    all_oids = oid_data.get("objectIds") or []
    total = len(all_oids)
    print(f"    OIDs matched: {total:,}")
    if total == 0:
        return gpd.GeoDataFrame(
            columns=["geometry"], geometry="geometry", crs=f"EPSG:{out_sr}"
        )

    # ── Fetch in OID-range batches ───────────────────────────────────────────
    all_oids_sorted = sorted(all_oids)
    gdfs: list[gpd.GeoDataFrame] = []
    fetched = 0

    for start_idx in range(0, total, page_size):
        batch_oids = all_oids_sorted[start_idx : start_idx + page_size]
        oid_min, oid_max = batch_oids[0], batch_oids[-1]
        fetched += len(batch_oids)
        print(
            f"    Fetching {fetched:,}/{total:,} "
            f"(OIDs {oid_min}–{oid_max}) …",
            end="\r",
            flush=True,
        )
        batch_params = {
            **base_params,
            "where": f"{oid_field} >= {oid_min} AND {oid_field} <= {oid_max}",
            "returnGeometry": "true",
        }
        # Remove any previous geometry filter from where-only requests
        data = _get(base_url + "/query", batch_params, timeout=300)
        features = data.get("features", [])
        if features:
            chunk = gpd.GeoDataFrame.from_features(features, crs=f"EPSG:{out_sr}")
            gdfs.append(chunk)

    print()  # newline after \r progress
    if not gdfs:
        return gpd.GeoDataFrame(
            columns=["geometry"], geometry="geometry", crs=f"EPSG:{out_sr}"
        )
    return pd.concat(gdfs, ignore_index=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("output"),
        help="Directory for output shapefile (default: ./output/)",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=2000,
        help="Features per page when querying services (default: 2000)",
    )
    p.add_argument(
        "--cmap-layer",
        type=str,
        default="",
        help=(
            "Name fragment to match a specific CMAP FeatureServer layer "
            "(default: first layer)"
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir: Path = args.output
    out_shp: Path = out_dir / "il_residential_buildings_cmap.shp"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Discover CMAP layer ──────────────────────────────────────────
    print("Step 1: Discovering CMAP FeatureServer layer …")
    cmap_base = find_cmap_layer(CMAP_SERVER, args.cmap_layer)
    print(f"  Using layer: {cmap_base}")
    cmap_meta = layer_meta(cmap_base)
    landuse_field = detect_landuse_field(cmap_meta)
    print(f"  Land-use field: {landuse_field}")

    # ── Step 2: Fetch CMAP residential polygons ──────────────────────────────
    print("\nStep 2: Fetching CMAP residential land-use polygons …")
    codes_sql = ", ".join(str(c) for c in RESIDENTIAL_CODES)
    cmap_where = f"{landuse_field} IN ({codes_sql})"
    print(f"  Filter: {cmap_where}")

    cmap_gdf = fetch_all_features(
        cmap_base,
        where=cmap_where,
        out_fields=f"{landuse_field},LANDUSE2,Shape_Area",
        out_sr=4326,
        page_size=args.chunk_size,
    )
    if cmap_gdf.empty:
        print("ERROR: No CMAP residential polygons retrieved. Check connectivity.")
        sys.exit(1)
    print(f"  Loaded {len(cmap_gdf):,} CMAP residential polygons.")

    # Reproject to working CRS for accurate spatial operations
    cmap_proj = cmap_gdf.to_crs(WORK_CRS)

    # ── Step 3: Build bbox envelope for building pre-filter ──────────────────
    print("\nStep 3: Building bounding-box filter from CMAP extent …")
    xmin, ymin, xmax, ymax = cmap_gdf.total_bounds  # WGS84
    print(f"  Extent (WGS84): xmin={xmin:.4f} ymin={ymin:.4f} xmax={xmax:.4f} ymax={ymax:.4f}")

    geom_envelope = json.dumps({
        "xmin": xmin,
        "ymin": ymin,
        "xmax": xmax,
        "ymax": ymax,
        "spatialReference": {"wkid": 4326},
    })
    geom_filter = {
        "geometry": geom_envelope,
        "geometryType": "esriGeometryEnvelope",
        "spatialRel": "esriSpatialRelIntersects",
        "inSR": "4326",
    }

    # ── Step 4: Fetch building footprints within CMAP bbox ───────────────────
    print("\nStep 4: Fetching IL DNR building footprints within CMAP extent …")
    print("  (Querying ~7 counties – this will take several minutes)")

    bldg_meta = layer_meta(BLDG_BASE)
    oid_field = bldg_meta.get("objectIdField", "OBJECTID")

    # Count buildings in bbox first
    count_data = _get(
        BLDG_BASE + "/query",
        {
            "where": "1=1",
            **geom_filter,
            "returnCountOnly": "true",
            "f": "json",
        },
        timeout=120,
    )
    total_bldgs = count_data.get("count", 0)
    print(f"  Buildings in CMAP bounding box: {total_bldgs:,}")

    if total_bldgs == 0:
        print("ERROR: No building footprints found in CMAP extent. Check connectivity.")
        sys.exit(1)

    bldg_gdf = fetch_all_features(
        BLDG_BASE,
        where="1=1",
        out_fields=f"{oid_field},COUNTY",
        out_sr=4326,
        geometry_filter=geom_filter,
        page_size=min(args.chunk_size, bldg_meta.get("maxRecordCount", 1000)),
    )
    print(f"  Loaded {len(bldg_gdf):,} buildings in CMAP bounding box.")

    # ── Step 5: Precise spatial join ─────────────────────────────────────────
    print("\nStep 5: Spatial join – keeping buildings within CMAP residential zones …")
    print("  Reprojecting buildings to working CRS …")
    bldg_proj = bldg_gdf.to_crs(WORK_CRS)

    print("  Running sjoin (intersects) …")
    joined = gpd.sjoin(
        bldg_proj,
        cmap_proj[[landuse_field, "geometry"]],
        how="inner",
        predicate="intersects",
    )

    # Drop duplicate buildings (a building may touch >1 CMAP polygon)
    dup_key = oid_field if oid_field in joined.columns else None
    if dup_key:
        joined = joined.drop_duplicates(subset=[dup_key])
    joined = joined.drop(columns=["index_right"], errors="ignore")

    print(f"  Residential buildings (after dedup): {len(joined):,}")

    # ── Step 6: Export shapefile ─────────────────────────────────────────────
    print(f"\nStep 6: Writing shapefile → {out_shp} …")
    result = joined.to_crs(OUT_CRS)

    # Shapefile field names are limited to 10 chars; rename if needed
    rename = {}
    for col in result.columns:
        if col != "geometry" and len(col) > 10:
            rename[col] = col[:10]
    if rename:
        print(f"  Renaming long field names: {rename}")
        result = result.rename(columns=rename)

    result.to_file(out_shp, driver="ESRI Shapefile")
    print("  Done.")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Output shapefile : {out_shp.resolve()}")
    print(f"  Feature count    : {len(result):,}")
    print(f"  CRS              : {OUT_CRS}")
    if landuse_field in result.columns:
        code_counts = result[landuse_field].value_counts().sort_index()
        labels = {
            1111: "Single-Family Detached",
            1112: "Single-Family Attached",
            1113: "SF Attached to Non-Res",
            1121: "Two-Family Detached",
            1122: "Two-Family Attached",
            1130: "Multi-Family (apt/condo)",
        }
        print("\n  Buildings by land-use code:")
        for code, count in code_counts.items():
            label = labels.get(int(code), "")
            print(f"    {code}  {label:<26}  {count:>10,}")
    print("=" * 60)


if __name__ == "__main__":
    main()
