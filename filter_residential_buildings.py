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
  Microsoft US Building Footprints (Illinois):
    https://minedbuildings.z5.web.core.windows.net/legacy/usbuildings-v2/Illinois.geojson.zip
  CMAP LUI20 FeatureServer:
    https://services5.arcgis.com/LcMXE3TFhi1BSaCY/arcgis/rest/services/LUI20_geodatabase_v1_CMAP/FeatureServer

Usage:
  python filter_residential_buildings.py [--output OUTPUT_DIR] [--chunk-size N] [--cache-dir DIR]

  On Windows, if 'python' is not on PATH use the full installer path:
  "C:/Users/HP z440/AppData/Local/Programs/Python/Python312/python.exe" filter_residential_buildings.py [--output OUTPUT_DIR]

  Install dependencies first (same full path):
  "C:/Users/HP z440/AppData/Local/Programs/Python/Python312/python.exe" -m pip install geopandas pandas requests fiona pyproj shapely

Output:
  <output_dir>/il_residential_buildings_cmap.shp  (and companion files)
"""

import argparse
import sys
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Service endpoints
# ---------------------------------------------------------------------------
# Microsoft US Building Footprints – Illinois state file (hosted on Azure Blob Storage)
MSFT_IL_URL = (
    "https://minedbuildings.z5.web.core.windows.net/legacy/usbuildings-v2"
    "/Illinois.geojson.zip"
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

def _request(url: str, params: dict, retries: int = 4, timeout: int = 180) -> dict:
    """POST with exponential-backoff retry and basic error handling.

    Using POST avoids URL-length limits when objectIds lists are large.
    """
    for attempt in range(retries):
        try:
            r = requests.post(url, data=params, timeout=timeout)
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


def _get(url: str, params: dict, retries: int = 4, timeout: int = 180) -> dict:
    """GET wrapper kept for metadata/count calls that are safe as GET."""
    return _request(url, params, retries=retries, timeout=timeout)


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
            "objectIds": ",".join(str(o) for o in batch_oids),
            "where": "1=1",
        }
        data = _request(base_url + "/query", batch_params, timeout=300)
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


def fetch_msft_buildings(
    bbox: tuple[float, float, float, float],
    cache_dir: Path,
) -> gpd.GeoDataFrame:
    """
    Download the Microsoft USBuildingFootprints Illinois zip (once) and return
    buildings clipped to *bbox* = (xmin, ymin, xmax, ymax) in EPSG:4326.

    The ~500 MB zip is cached in *cache_dir* so subsequent runs skip the download.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "Illinois.geojson.zip"

    if zip_path.exists():
        print(f"  Using cached file: {zip_path}")
    else:
        print(f"  Downloading Microsoft IL building footprints → {zip_path}")
        print("  (This is a large file; it will be cached for future runs.)")
        with requests.get(MSFT_IL_URL, stream=True, timeout=600) as r:
            r.raise_for_status()
            total_bytes = int(r.headers.get("content-length", 0))
            downloaded = 0
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MB chunks
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_bytes:
                        pct = downloaded / total_bytes * 100
                        print(f"    {downloaded / 1e6:.0f} MB / {total_bytes / 1e6:.0f} MB  ({pct:.0f}%)", end="\r", flush=True)
        print(f"\n  Download complete: {zip_path.stat().st_size / 1e6:.0f} MB")

    print(f"  Reading buildings clipped to CMAP bbox …")
    gdf = gpd.read_file(f"zip://{zip_path}", bbox=bbox)
    gdf = gdf.set_crs("EPSG:4326", allow_override=True)
    return gdf


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
        help="Features per page when querying CMAP service (default: 2000)",
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("cache"),
        help="Directory to cache the Microsoft IL building footprints zip (default: ./cache/)",
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
        out_fields="*",
        out_sr=4326,
        page_size=args.chunk_size,
    )
    if cmap_gdf.empty:
        print("ERROR: No CMAP residential polygons retrieved. Check connectivity.")
        sys.exit(1)
    print(f"  Loaded {len(cmap_gdf):,} CMAP residential polygons.")

    # Reproject to working CRS for accurate spatial operations
    cmap_proj = cmap_gdf.to_crs(WORK_CRS)

    # ── Step 3: Build bbox from CMAP extent ──────────────────────────────────
    print("\nStep 3: Building bounding-box filter from CMAP extent …")
    xmin, ymin, xmax, ymax = cmap_gdf.total_bounds  # WGS84
    print(f"  Extent (WGS84): xmin={xmin:.4f} ymin={ymin:.4f} xmax={xmax:.4f} ymax={ymax:.4f}")

    # ── Step 4: Download Microsoft building footprints ────────────────────────
    print("\nStep 4: Loading Microsoft USBuildingFootprints for Illinois …")
    bldg_gdf = fetch_msft_buildings(
        bbox=(xmin, ymin, xmax, ymax),
        cache_dir=args.cache_dir,
    )
    if bldg_gdf.empty:
        print("ERROR: No building footprints found in CMAP extent.")
        sys.exit(1)
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
    joined = joined.drop(columns=["index_right"], errors="ignore")
    joined = joined[~joined.index.duplicated(keep="first")]

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
