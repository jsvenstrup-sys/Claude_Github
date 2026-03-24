#!/usr/bin/env python3
"""
Illinois Buildable Area Analysis
=================================
Builds a buildable-area polygon for 9 northeastern Illinois counties by:
  1. Combining county boundary polygons into one study-area polygon
  2. Erasing areas within each setback/exclusion buffer

Target Counties
---------------
Cook, Lake, McHenry, DuPage, Will, Kane, Kendall, DeKalb, Winnebago

Projection
----------
EPSG:3435 — Illinois State Plane East, NAD83, US Survey Feet
  • All buffer distances applied in feet (no unit conversion needed)
  • Final outputs saved in both EPSG:3435 and WGS84 (6 decimal places)
  • ~1-2 ft accuracy; never uses geographic degrees for distance math

Setback / Exclusion Layers
---------------------------
  1. 150 ft  Residential zones      (Claude-generated local shapefile)
  2.  25 ft  All other buildings    (Microsoft USBuildingFootprints)
  3.  50 ft  IDOT road jurisdiction (IDOT ArcGIS Open Data)
  4.  30 ft  Railroads              (IDOT ArcGIS Open Data)
  5.  50 ft  FEMA flood plain       (FEMA NFHL via REST + ISGS fallback)
  6.  25 ft  Wetlands               (USFWS National Wetlands Inventory)

Running the Script
------------------
Run on a Windows machine that has access to the local shapefiles.

    python illinois_buildable_area.py

Data that can be pre-downloaded manually (see DATA_DIR below):
  • Microsoft Building Footprints  → data/ms_buildings/Illinois.geojson
  • IDOT Jurisdiction              → data/idot_jurisdiction/idot_jurisdiction.geojson
  • Illinois Railroads             → data/railroads/il_railroads.geojson
  • ISGS Flood Zones (zip)         → data/floodplain/  (extract the zip here)
  • FWS NWI Wetlands (zip)         → data/wetlands/    (extract the zip here)

Requirements
------------
    pip install geopandas shapely fiona requests pandas pyproj numpy
"""

import io
import json
import os
import re
import sys
import time
import warnings
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import box
from shapely.ops import unary_union

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION  — edit paths / settings here
# =============================================================================

# Projected CRS: Illinois State Plane East, NAD83, US Survey Feet
# All 9 target counties are within this zone; distances directly in feet.
CRS_PROJ   = "EPSG:3435"
CRS_WGS84  = "EPSG:4326"

# Output coordinate precision for WGS84 files (6 dp ≈ 0.1 m at Illinois lat)
OUTPUT_DECIMAL_PLACES = 6

TARGET_COUNTIES = [
    "Cook", "Lake", "McHenry", "DuPage", "Will",
    "Kane", "Kendall", "DeKalb", "Winnebago",
]

# Buffer distances — US Survey Feet (EPSG:3435 native unit)
BUFFER_FT = {
    "residential"    : 150,
    "other_buildings": 25,
    "idot_roads"     : 50,
    "railroads"      : 30,
    "floodplain"     : 50,
    "wetlands"       : 25,
}

# Path to Claude-generated residential shapefile directory (Windows path)
RESIDENTIAL_SHP_DIR = Path(r"C:\Users\HP z440\Desktop\New Energy\Trico\output")

# Script-relative data / output directories (created automatically)
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

for _d in (DATA_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# =============================================================================
# LOGGING
# =============================================================================

def log(msg: str) -> None:
    print(f"[buildable] {msg}", flush=True)


# =============================================================================
# DOWNLOAD / IO HELPERS
# =============================================================================

def _get(url: str, timeout: int = 120, stream: bool = False) -> requests.Response:
    """requests.get with basic retry (3×, exponential back-off)."""
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=timeout, stream=stream)
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            if attempt == 2:
                raise
            wait = 2 ** attempt
            log(f"  Retry {attempt+1}/3 after {wait}s ({exc})")
            time.sleep(wait)


def download_zip(url: str, dest_dir: Path, desc: str = "") -> Path:
    """Download a zip from *url* and extract to *dest_dir*."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    log(f"  Downloading {desc or url} …")
    resp = _get(url, timeout=300)
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        zf.extractall(dest_dir)
    log(f"  Extracted to {dest_dir}")
    return dest_dir


def first_file(directory: Path, ext: str) -> Path | None:
    """Return the first file with *ext* found recursively, or None."""
    hits = list(directory.rglob(f"*{ext}"))
    return hits[0] if hits else None


def arcgis_paged_query(service_url: str, bbox_wgs84: tuple = None) -> list:
    """
    Download all features from an ArcGIS FeatureServer layer via paginated
    queries.  Returns a list of GeoJSON feature dicts.

    Parameters
    ----------
    service_url : str
        FeatureServer layer URL, e.g.
        https://services.arcgis.com/.../FeatureServer/0
    bbox_wgs84 : tuple (minx, miny, maxx, maxy) in EPSG:4326, optional
        Spatial filter applied to every page request.
    """
    query_url = f"{service_url.rstrip('/')}/query"
    all_features: list = []
    offset = 0
    page_size = 1000

    base_params: dict = {
        "where"             : "1=1",
        "outFields"         : "*",
        "f"                 : "geojson",
        "returnGeometry"    : "true",
        "outSR"             : "4326",
        "resultRecordCount" : page_size,
    }
    if bbox_wgs84:
        minx, miny, maxx, maxy = bbox_wgs84
        base_params.update({
            "geometry"     : f"{minx},{miny},{maxx},{maxy}",
            "geometryType" : "esriGeometryEnvelope",
            "inSR"         : "4326",
            "spatialRel"   : "esriSpatialRelIntersects",
        })

    while True:
        params = {**base_params, "resultOffset": offset}
        try:
            resp = _get(query_url, timeout=60)
            data = resp.json()
        except Exception as exc:
            log(f"  Paged query error at offset {offset}: {exc}")
            break

        features = data.get("features", [])
        all_features.extend(features)
        log(f"    … {len(all_features)} features fetched")
        if len(features) < page_size:
            break
        offset += page_size

    return all_features


def geojson_from_features(features: list) -> dict:
    return {"type": "FeatureCollection", "features": features}


# =============================================================================
# STEP 1 — COUNTY BOUNDARIES
# =============================================================================

def load_study_area() -> gpd.GeoDataFrame:
    """
    Download 2023 Census TIGER/Line Illinois counties, filter to the 9 target
    counties, dissolve to a single polygon, and reproject to CRS_PROJ.
    """
    county_dir = DATA_DIR / "counties"
    county_shp = first_file(county_dir, ".shp")

    if county_shp is None:
        # Census TIGER county files are national (all states in one zip).
        url = ("https://www2.census.gov/geo/tiger/TIGER2024/COUNTY/"
               "tl_2024_us_county.zip")
        download_zip(url, county_dir, "Census TIGER Illinois counties")
        county_shp = first_file(county_dir, ".shp")
        if county_shp is None:
            raise FileNotFoundError(
                "County shapefile not found after download.\n"
                f"Manually download from Census TIGER and extract to {county_dir}"
            )

    gdf = gpd.read_file(county_shp)
    # Filter to Illinois (STATEFP "17") first to avoid name collisions
    # (e.g. "Cook" or "Kane" exist in other states)
    if "STATEFP" in gdf.columns:
        gdf = gdf[gdf["STATEFP"] == "17"].copy()
    target_upper = {c.upper() for c in TARGET_COUNTIES}
    gdf = gdf[gdf["NAME"].str.upper().isin(target_upper)].copy()

    if gdf.empty:
        raise ValueError(
            "No target counties matched. Check the 'NAME' field in the "
            "TIGER county shapefile."
        )

    found = sorted(gdf["NAME"].tolist())
    log(f"  Found {len(found)} counties: {', '.join(found)}")
    gdf = gdf.to_crs(CRS_PROJ)
    dissolved = gdf.dissolve().reset_index(drop=True)
    area_acres = dissolved.geometry.area.sum() / 43_560
    log(f"  Study area: {area_acres:,.0f} acres")
    return dissolved


# =============================================================================
# STEP 2 — RESIDENTIAL SETBACK (150 ft)
# =============================================================================

def load_residential(study_area: gpd.GeoDataFrame) -> gpd.GeoDataFrame | None:
    """
    Load the Claude-generated residential shapefile from RESIDENTIAL_SHP_DIR,
    clip to study area, and buffer 150 ft.
    """
    if not RESIDENTIAL_SHP_DIR.exists():
        log(f"  SKIP — directory not found: {RESIDENTIAL_SHP_DIR}")
        log("  (Run on the Windows machine or update RESIDENTIAL_SHP_DIR)")
        return None

    shps = list(RESIDENTIAL_SHP_DIR.rglob("*.shp"))
    if not shps:
        log(f"  SKIP — no .shp files found in {RESIDENTIAL_SHP_DIR}")
        return None

    log(f"  Found {len(shps)} shapefile(s) in {RESIDENTIAL_SHP_DIR}")
    # Bounding box in WGS84 for fast bbox pre-filter on read
    study_bbox_wgs = tuple(study_area.to_crs(CRS_WGS84).total_bounds)
    gdfs = []
    for i, shp in enumerate(shps, 1):
        log(f"  [{i}/{len(shps)}] {shp.name} …")
        try:
            # bbox pre-filter avoids loading features outside study area
            gdf = gpd.read_file(shp, bbox=study_bbox_wgs).to_crs(CRS_PROJ)
            if gdf.empty:
                log(f"    no features in bbox — skip")
                continue
            log(f"    {len(gdf)} features in bbox, clipping …")
            clipped = gpd.clip(gdf, study_area)
            if not clipped.empty:
                log(f"    {len(clipped)} features after clip")
                gdfs.append(clipped)
            else:
                log(f"    0 features after clip — skip")
        except Exception as exc:
            log(f"    WARNING: {exc} — skip")

    if not gdfs:
        log("  SKIP — no residential features intersect the study area")
        return None

    combined = gpd.GeoDataFrame(
        pd.concat(gdfs, ignore_index=True), crs=CRS_PROJ
    )
    combined["geometry"] = combined.geometry.buffer(BUFFER_FT["residential"])
    log(f"  {len(combined)} features buffered {BUFFER_FT['residential']} ft")
    return combined


# =============================================================================
# STEP 3 — OTHER BUILDINGS SETBACK (25 ft)
# =============================================================================

def load_buildings(study_area: gpd.GeoDataFrame) -> gpd.GeoDataFrame | None:
    """
    Load Microsoft USBuildingFootprints for Illinois, clip to study area,
    and buffer 25 ft.

    Data source:
        https://github.com/Microsoft/USBuildingFootprints
    Manual download: place Illinois.geojson in data/ms_buildings/
    """
    bldg_dir  = DATA_DIR / "ms_buildings"
    bldg_file = first_file(bldg_dir, ".geojson") or first_file(bldg_dir, ".json")

    if bldg_file is None:
        bldg_dir.mkdir(parents=True, exist_ok=True)
        # Try the v1.1 public blob (may change; manual download preferred)
        urls = [
            "https://usbuildingdata.blob.core.windows.net/usbuildings-v1-1/Illinois.zip",
            "https://usbuildingdata.blob.core.windows.net/usbuildings-v2/Illinois.geojson.zip",
        ]
        for url in urls:
            try:
                download_zip(url, bldg_dir, "Microsoft Building Footprints — Illinois")
                bldg_file = first_file(bldg_dir, ".geojson") or first_file(bldg_dir, ".json")
                if bldg_file:
                    break
            except Exception as exc:
                log(f"  URL failed: {exc}")

    if bldg_file is None:
        log("  SKIP — Microsoft Building Footprints not available.")
        log("  Manual download steps:")
        log("    1. Visit https://github.com/Microsoft/USBuildingFootprints")
        log("    2. Download Illinois.zip")
        log(f"   3. Extract Illinois.geojson to {bldg_dir}")
        return None

    log(f"  Reading buildings (may be large) …")
    study_bounds = tuple(study_area.to_crs(CRS_WGS84).total_bounds)  # bbox for spatial filter
    gdf = gpd.read_file(str(bldg_file), bbox=study_bounds).to_crs(CRS_PROJ)
    gdf = gpd.clip(gdf, study_area)
    gdf["geometry"] = gdf.geometry.buffer(BUFFER_FT["other_buildings"])
    log(f"  {len(gdf)} buildings buffered {BUFFER_FT['other_buildings']} ft")
    return gdf


# =============================================================================
# STEP 4 — IDOT ROAD JURISDICTION SETBACK (50 ft)
# =============================================================================

def load_idot_roads(study_area: gpd.GeoDataFrame) -> gpd.GeoDataFrame | None:
    """
    Download IDOT Illinois Jurisdiction layer, clip to study area,
    and buffer 50 ft.

    Source: https://gis-idot.opendata.arcgis.com/datasets/IDOT::illinois-jurisdiction/about
    Manual download: save GeoJSON as data/idot_jurisdiction/idot_jurisdiction.geojson
    """
    idot_dir  = DATA_DIR / "idot_jurisdiction"
    idot_file = idot_dir / "idot_jurisdiction.geojson"

    if not idot_file.exists():
        idot_dir.mkdir(parents=True, exist_ok=True)
        study_bbox = tuple(study_area.to_crs(CRS_WGS84).total_bounds)

        # Attempt 1: ArcGIS Hub download API (slug-based)
        hub_urls = [
            "https://opendata.arcgis.com/api/v3/datasets/IDOT__illinois-jurisdiction_0/downloads/data?format=geojson&spatialRefId=4326",
            "https://opendata.arcgis.com/api/v3/datasets/IDOT__illinois-jurisdiction/downloads/data?format=geojson&spatialRefId=4326",
            "https://hub.arcgis.com/api/v3/datasets/IDOT__illinois-jurisdiction_0/downloads/data?format=geojson&spatialRefId=4326",
        ]
        for url in hub_urls:
            try:
                resp = _get(url, timeout=120)
                idot_file.write_bytes(resp.content)
                log("  Downloaded IDOT Jurisdiction via Hub API.")
                break
            except Exception:
                pass

        # Attempt 2: known FeatureServer REST endpoints (paginated)
        if not idot_file.exists():
            feature_servers = [
                "https://services1.arcgis.com/gMJSxUaUFn8cPaB8/arcgis/rest/services/Illinois_Jurisdiction/FeatureServer/0",
                "https://services1.arcgis.com/gMJSxUaUFn8cPaB8/arcgis/rest/services/IDOT_Illinois_Jurisdiction/FeatureServer/0",
            ]
            for fs_url in feature_servers:
                try:
                    log(f"  Trying FeatureServer: {fs_url}")
                    features = arcgis_paged_query(fs_url, bbox_wgs84=study_bbox)
                    if features:
                        idot_file.write_text(
                            json.dumps(geojson_from_features(features))
                        )
                        log(f"  Downloaded {len(features)} jurisdiction features.")
                        break
                except Exception as exc:
                    log(f"  FeatureServer failed: {exc}")

    if not idot_file.exists():
        log("  SKIP — IDOT Jurisdiction data not available.")
        log("  Manual download steps:")
        log("    1. Visit https://gis-idot.opendata.arcgis.com/datasets/IDOT::illinois-jurisdiction/about")
        log("    2. Click Download → GeoJSON")
        log(f"   3. Save as: {idot_file}")
        return None

    gdf = gpd.read_file(str(idot_file)).to_crs(CRS_PROJ)
    gdf = gpd.clip(gdf, study_area)
    gdf["geometry"] = gdf.geometry.buffer(BUFFER_FT["idot_roads"])
    log(f"  {len(gdf)} IDOT jurisdiction features buffered {BUFFER_FT['idot_roads']} ft")
    return gdf


# =============================================================================
# STEP 5 — RAILROAD SETBACK (30 ft)
# =============================================================================

def load_railroads(study_area: gpd.GeoDataFrame) -> gpd.GeoDataFrame | None:
    """
    Download IDOT Illinois Railroads layer, clip to study area,
    and buffer 30 ft.

    Source: https://gis-idot.opendata.arcgis.com/datasets/illinois-railroads/explore
    Manual download: save GeoJSON as data/railroads/il_railroads.geojson
    """
    rail_dir  = DATA_DIR / "railroads"
    rail_file = rail_dir / "il_railroads.geojson"

    if not rail_file.exists():
        rail_dir.mkdir(parents=True, exist_ok=True)
        study_bbox = tuple(study_area.to_crs(CRS_WGS84).total_bounds)

        hub_urls = [
            "https://opendata.arcgis.com/api/v3/datasets/illinois-railroads/downloads/data?format=geojson&spatialRefId=4326",
            "https://opendata.arcgis.com/api/v3/datasets/IDOT__illinois-railroads/downloads/data?format=geojson&spatialRefId=4326",
            "https://hub.arcgis.com/api/v3/datasets/illinois-railroads/downloads/data?format=geojson&spatialRefId=4326",
        ]
        for url in hub_urls:
            try:
                resp = _get(url, timeout=120)
                rail_file.write_bytes(resp.content)
                log("  Downloaded Illinois Railroads via Hub API.")
                break
            except Exception:
                pass

        if not rail_file.exists():
            feature_servers = [
                "https://services1.arcgis.com/gMJSxUaUFn8cPaB8/arcgis/rest/services/Illinois_Railroads/FeatureServer/0",
                "https://services1.arcgis.com/gMJSxUaUFn8cPaB8/arcgis/rest/services/IDOT_Illinois_Railroads/FeatureServer/0",
            ]
            for fs_url in feature_servers:
                try:
                    log(f"  Trying FeatureServer: {fs_url}")
                    features = arcgis_paged_query(fs_url, bbox_wgs84=study_bbox)
                    if features:
                        rail_file.write_text(
                            json.dumps(geojson_from_features(features))
                        )
                        log(f"  Downloaded {len(features)} railroad features.")
                        break
                except Exception as exc:
                    log(f"  FeatureServer failed: {exc}")

    if not rail_file.exists():
        log("  SKIP — Illinois Railroads data not available.")
        log("  Manual download steps:")
        log("    1. Visit https://gis-idot.opendata.arcgis.com/datasets/illinois-railroads/explore")
        log("    2. Click Download → GeoJSON")
        log(f"   3. Save as: {rail_file}")
        return None

    gdf = gpd.read_file(str(rail_file)).to_crs(CRS_PROJ)
    gdf = gpd.clip(gdf, study_area)
    gdf["geometry"] = gdf.geometry.buffer(BUFFER_FT["railroads"])
    log(f"  {len(gdf)} railroad features buffered {BUFFER_FT['railroads']} ft")
    return gdf


# =============================================================================
# STEP 6 — FLOOD PLAIN SETBACK (50 ft)
# =============================================================================

def load_floodplain(study_area: gpd.GeoDataFrame) -> gpd.GeoDataFrame | None:
    """
    Load FEMA 100-year flood zone polygons for the study area.

    Primary:  FEMA NFHL public FeatureServer REST API (auto-download)
    Fallback: ISGS Clearinghouse zip (manual download)
              https://clearinghouse.isgs.illinois.edu/data/hydrology/
              flood-zones-unincorporated-areas-one-hundred-and-five-hundred-year
              → extract to data/floodplain/

    FEMA flood zone codes included (100-yr and conservative 500-yr):
        A, AE, AH, AO, A99, VE, V, X (shaded / 500-yr)
    """
    flood_dir  = DATA_DIR / "floodplain"
    flood_file = flood_dir / "fema_flood_zones.geojson"

    if not flood_file.exists():
        flood_dir.mkdir(parents=True, exist_ok=True)
        study_bbox = tuple(study_area.to_crs(CRS_WGS84).total_bounds)

        # FEMA NFHL public FeatureServer — layer 28 = Special Flood Hazard Areas
        # https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer/28
        fema_fs = "https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer/28"
        log("  Querying FEMA NFHL FeatureServer (layer 28 — flood hazard areas) …")
        try:
            features = arcgis_paged_query(fema_fs, bbox_wgs84=study_bbox)
            if features:
                flood_file.write_text(
                    json.dumps(geojson_from_features(features))
                )
                log(f"  Downloaded {len(features)} FEMA flood zone features.")
        except Exception as exc:
            log(f"  FEMA NFHL query failed: {exc}")

    # Fallback: look for ISGS shapefiles placed manually
    if not flood_file.exists():
        shps = list(flood_dir.rglob("*.shp"))
        if shps:
            log(f"  Using {len(shps)} ISGS shapefile(s) from {flood_dir}")
        else:
            log("  SKIP — no flood plain data found.")
            log("  Manual download steps:")
            log("    Option A (FEMA NFHL direct):")
            log("      Visit https://msc.fema.gov/portal/home")
            log("      Download NFHL for Illinois (state) → extract to data/floodplain/")
            log("    Option B (ISGS Clearinghouse):")
            log("      Visit https://clearinghouse.isgs.illinois.edu/data/hydrology/")
            log("      flood-zones-unincorporated-areas-one-hundred-and-five-hundred-year")
            log("      Download zip → extract to data/floodplain/")
            return None
    else:
        shps = None

    # FEMA flood zone codes to include
    FLOOD_ZONES = {"A", "AE", "AH", "AO", "A99", "VE", "V"}

    study_bbox_geom = box(*study_area.to_crs(CRS_WGS84).total_bounds).buffer(0.05)

    if flood_file.exists():
        gdf = gpd.read_file(str(flood_file))
    else:
        gdfs = []
        for shp in shps:
            try:
                g = gpd.read_file(shp)
                g = g[g.intersects(study_bbox_geom)]
                if "FLD_ZONE" in g.columns:
                    g = g[g["FLD_ZONE"].isin(FLOOD_ZONES)]
                gdfs.append(g)
            except Exception as exc:
                log(f"  Warning reading {shp.name}: {exc}")
        if not gdfs:
            log("  SKIP — no usable flood zone features found.")
            return None
        gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)

    if "FLD_ZONE" in gdf.columns:
        gdf = gdf[gdf["FLD_ZONE"].isin(FLOOD_ZONES)].copy()

    gdf = gdf.to_crs(CRS_PROJ)
    gdf = gpd.clip(gdf, study_area)
    gdf["geometry"] = gdf.geometry.buffer(BUFFER_FT["floodplain"])
    log(f"  {len(gdf)} flood zone features buffered {BUFFER_FT['floodplain']} ft")
    return gdf


# =============================================================================
# STEP 7 — WETLANDS SETBACK (25 ft)
# =============================================================================

def load_wetlands(study_area: gpd.GeoDataFrame) -> gpd.GeoDataFrame | None:
    """
    Load USFWS National Wetlands Inventory for Illinois, clip to study area,
    and buffer 25 ft.

    Source: https://www.fws.gov/program/national-wetlands-inventory/download-state-wetlands-data
    Manual download: extract Illinois zip to data/wetlands/
    """
    wetland_dir = DATA_DIR / "wetlands"

    has_data = (
        any(wetland_dir.rglob("*.shp"))
        or any(wetland_dir.rglob("*.gdb"))
        or any(wetland_dir.rglob("*.gpkg"))
    )

    if not has_data:
        wetland_dir.mkdir(parents=True, exist_ok=True)
        # FWS NWI state downloads — URL pattern may change; try several
        nwi_urls = [
            "https://www.fws.gov/wetlands/data/State-Downloads/IL_shapefile_wetlands.zip",
            "https://www.fws.gov/wetlands/Data/State-Downloads/IL_shapefile_wetlands.zip",
            "https://www.fws.gov/wetlands/data/State-Downloads/IL_geodatabase_wetlands.zip",
            "https://www.fws.gov/wetlands/Data/State-Downloads/IL_geodatabase_wetlands.zip",
        ]
        for url in nwi_urls:
            try:
                download_zip(url, wetland_dir, "FWS NWI Illinois Wetlands")
                has_data = (
                    any(wetland_dir.rglob("*.shp"))
                    or any(wetland_dir.rglob("*.gdb"))
                )
                if has_data:
                    break
            except Exception as exc:
                log(f"  URL failed: {exc}")

    if not has_data:
        log("  SKIP — NWI wetland data not available.")
        log("  Manual download steps:")
        log("    1. Visit https://www.fws.gov/program/national-wetlands-inventory/download-state-wetlands-data")
        log("    2. Select Illinois (IL) → Download")
        log(f"   3. Extract zip to: {wetland_dir}")
        return None

    # Read shapefile(s) or GDB
    shps = list(wetland_dir.rglob("*.shp"))
    gdbs = list(wetland_dir.rglob("*.gdb"))
    gpkgs = list(wetland_dir.rglob("*.gpkg"))

    study_bounds_wgs = tuple(study_area.to_crs(CRS_WGS84).total_bounds)
    gdfs = []

    for shp in shps:
        try:
            g = gpd.read_file(str(shp), bbox=study_bounds_wgs).to_crs(CRS_PROJ)
            gdfs.append(g)
        except Exception as exc:
            log(f"  Warning reading {shp.name}: {exc}")

    for gpkg in gpkgs:
        try:
            g = gpd.read_file(str(gpkg), bbox=study_bounds_wgs).to_crs(CRS_PROJ)
            gdfs.append(g)
        except Exception as exc:
            log(f"  Warning reading {gpkg.name}: {exc}")

    if not gdfs and gdbs:
        import fiona
        for gdb in gdbs:
            try:
                layers = fiona.listlayers(str(gdb))
                for layer in layers:
                    if any(k in layer.lower() for k in ("wetland", "nwi", "riparian")):
                        g = gpd.read_file(
                            str(gdb), layer=layer, bbox=study_bounds_wgs
                        ).to_crs(CRS_PROJ)
                        gdfs.append(g)
            except Exception as exc:
                log(f"  Warning reading {gdb.name}: {exc}")

    if not gdfs:
        log("  SKIP — no usable wetland features found in data directory.")
        return None

    combined = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=CRS_PROJ)
    combined = gpd.clip(combined, study_area)
    combined["geometry"] = combined.geometry.buffer(BUFFER_FT["wetlands"])
    log(f"  {len(combined)} wetland features buffered {BUFFER_FT['wetlands']} ft")
    return combined


# =============================================================================
# COORDINATE ROUNDING
# =============================================================================

def _round_coords(geom, decimals: int = 6):
    """Round all coordinate values in a Shapely geometry."""
    wkt = geom.wkt
    rounded = re.sub(
        r"-?\d+\.\d+",
        lambda m: str(round(float(m.group()), decimals)),
        wkt,
    )
    from shapely import wkt as shp_wkt
    return shp_wkt.loads(rounded)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    log("=" * 70)
    log("Illinois Buildable Area Analysis")
    log("Projection: EPSG:3435 (IL State Plane East, NAD83, US Survey Feet)")
    log("=" * 70)

    # ── Step 1: Study area ──────────────────────────────────────────────────
    log("\n[1/7] Loading county boundaries …")
    study_area = load_study_area()
    study_geom = unary_union(study_area.geometry.tolist())

    # ── Steps 2–7: Exclusion layers ─────────────────────────────────────────
    layers: dict = {
        "Residential (150 ft)"     : ("residential",     load_residential),
        "Buildings (25 ft)"        : ("other_buildings",  load_buildings),
        "IDOT roads (50 ft)"       : ("idot_roads",       load_idot_roads),
        "Railroads (30 ft)"        : ("railroads",        load_railroads),
        "Flood plain (50 ft)"      : ("floodplain",       load_floodplain),
        "Wetlands (25 ft)"         : ("wetlands",         load_wetlands),
    }

    exclusion_geoms: list = []
    skipped: list = []

    for step_num, (label, (key, func)) in enumerate(layers.items(), start=2):
        log(f"\n[{step_num}/7] {label} …")
        try:
            gdf = func(study_area)
            if gdf is not None and not gdf.empty:
                exclusion_geoms.extend(gdf.geometry.tolist())
                log(f"  ✓ {label} — {len(gdf)} features added to exclusion mask")
            else:
                skipped.append(label)
        except Exception as exc:
            log(f"  ERROR loading {label}: {exc}")
            skipped.append(label)

    if not exclusion_geoms:
        log("\nERROR: No exclusion layers loaded — cannot generate output.")
        sys.exit(1)

    # ── Union exclusion mask ─────────────────────────────────────────────────
    log("\n[8/7] Building exclusion mask (union of all buffers) …")
    excl_geom  = unary_union(exclusion_geoms)
    excl_clipped = excl_geom.intersection(study_geom)

    log("[9/7] Subtracting exclusion mask from study area …")
    buildable_geom = study_geom.difference(excl_clipped)

    # ── Outputs ─────────────────────────────────────────────────────────────
    log("\n[10/7] Saving outputs …")

    # 1. Projected CRS shapefile (EPSG:3435, feet)
    proj_gdf = gpd.GeoDataFrame({"geometry": [buildable_geom]}, crs=CRS_PROJ)
    out_proj_shp = OUTPUT_DIR / "buildable_IL_counties_EPSG3435.shp"
    proj_gdf.to_file(str(out_proj_shp))
    log(f"  Shapefile  (EPSG:3435, ft): {out_proj_shp}")

    # 2. WGS84 GeoJSON with 6 decimal places
    wgs_gdf = proj_gdf.to_crs(CRS_WGS84).copy()
    wgs_gdf["geometry"] = wgs_gdf.geometry.apply(
        lambda g: _round_coords(g, OUTPUT_DECIMAL_PLACES)
    )
    out_wgs_geojson = OUTPUT_DIR / "buildable_IL_counties_WGS84.geojson"
    wgs_gdf.to_file(str(out_wgs_geojson), driver="GeoJSON")
    log(f"  GeoJSON    (WGS84, 6 dp):  {out_wgs_geojson}")

    # 3. WGS84 shapefile
    out_wgs_shp = OUTPUT_DIR / "buildable_IL_counties_WGS84.shp"
    wgs_gdf.to_file(str(out_wgs_shp))
    log(f"  Shapefile  (WGS84):        {out_wgs_shp}")

    # ── Summary ─────────────────────────────────────────────────────────────
    buildable_acres = proj_gdf.geometry.area.sum() / 43_560
    study_acres     = study_geom.area / 43_560
    pct_buildable   = 100 * buildable_acres / study_acres if study_acres else 0

    log(f"\n{'='*70}")
    log("COMPLETE")
    log(f"  Study area : {study_acres:>12,.0f} acres")
    log(f"  Buildable  : {buildable_acres:>12,.0f} acres  ({pct_buildable:.1f}%)")
    log(f"  Outputs    : {OUTPUT_DIR.resolve()}")

    if skipped:
        log(f"\n  WARNING — {len(skipped)} layer(s) were skipped:")
        for s in skipped:
            log(f"    • {s}")
        log("  Result is PARTIAL.  See instructions above to supply missing data.")
    else:
        log("  All 6 exclusion layers applied — result is COMPLETE.")

    log("=" * 70)


if __name__ == "__main__":
    main()
