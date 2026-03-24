#!/usr/bin/env python3
"""
Residential Setback — 150 ft exclusion zone
============================================
Reads the residential building footprint shapefile, applies a 150 ft buffer,
clips to the 9-county study area, and saves the result.

Projection: EPSG:3435 (Illinois State Plane East, NAD83, US Survey Feet)
  All buffer distances are in feet — no unit conversion needed.

Run:
    python residential_setback.py

Outputs (in ./output/):
    residential_setback_EPSG3435.shp    — projected, feet
    residential_setback_WGS84.geojson   — WGS84, 6 decimal places
"""

import warnings
import zipfile
import io
import time
from pathlib import Path

import geopandas as gpd
import requests
from shapely.ops import unary_union
from shapely.validation import make_valid

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG
# =============================================================================

CRS_PROJ  = "EPSG:3435"   # IL State Plane East, US Survey Feet
CRS_WGS84 = "EPSG:4326"

BUFFER_FT = 150

# Path to residential buildings shapefile (or folder containing one .shp)
RESIDENTIAL_SHP = Path(
    r"C:\Users\HP z440\Desktop\New Energy\Trico\output\il_residential_buildings_cmap.shp"
)

BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(DATA_DIR / "counties").mkdir(parents=True, exist_ok=True)


def log(msg):
    print(f"[setback] {msg}", flush=True)


# =============================================================================
# COUNTY BOUNDARIES
# =============================================================================

TARGET_COUNTIES = [
    "Cook", "Lake", "McHenry", "DuPage", "Will",
    "Kane", "Kendall", "DeKalb", "Winnebago",
]


def load_study_area() -> gpd.GeoDataFrame:
    county_dir = DATA_DIR / "counties"
    shps = list(county_dir.rglob("*.shp"))

    if not shps:
        log("Downloading Census TIGER county boundaries ...")
        url = ("https://www2.census.gov/geo/tiger/TIGER2024/COUNTY/"
               "tl_2024_us_county.zip")
        for attempt in range(3):
            try:
                resp = requests.get(url, timeout=300)
                resp.raise_for_status()
                with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                    zf.extractall(county_dir)
                break
            except Exception as exc:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
        shps = list(county_dir.rglob("*.shp"))

    gdf = gpd.read_file(shps[0])
    if "STATEFP" in gdf.columns:
        gdf = gdf[gdf["STATEFP"] == "17"]
    target_upper = {c.upper() for c in TARGET_COUNTIES}
    gdf = gdf[gdf["NAME"].str.upper().isin(target_upper)].copy()
    gdf = gdf.to_crs(CRS_PROJ)
    dissolved = gdf.dissolve().reset_index(drop=True)
    log(f"Study area: {dissolved.geometry.area.sum() / 43_560:,.0f} acres  "
        f"({len(gdf)} counties)")
    return dissolved


# =============================================================================
# RESIDENTIAL SETBACK
# =============================================================================

def run(study_area: gpd.GeoDataFrame) -> None:
    if not RESIDENTIAL_SHP.exists():
        raise FileNotFoundError(
            f"Residential shapefile not found:\n  {RESIDENTIAL_SHP}\n"
            "Update RESIDENTIAL_SHP at the top of this script."
        )

    # ── Read with bbox pre-filter ────────────────────────────────────────────
    bbox_wgs = tuple(study_area.to_crs(CRS_WGS84).total_bounds)
    log(f"Reading {RESIDENTIAL_SHP.name} with bbox filter ...")
    gdf = gpd.read_file(RESIDENTIAL_SHP, bbox=bbox_wgs)
    log(f"  {len(gdf):,} features in bounding box")

    if gdf.empty:
        log("ERROR: No features found in the study area bounding box.")
        log("Check that the shapefile overlaps northeastern Illinois.")
        return

    # ── Reproject ────────────────────────────────────────────────────────────
    log("Reprojecting to EPSG:3435 ...")
    gdf = gdf.to_crs(CRS_PROJ)

    # ── Fix invalid geometries ───────────────────────────────────────────────
    log("Checking/fixing geometry validity ...")
    invalid_mask = ~gdf.geometry.is_valid
    n_invalid = invalid_mask.sum()
    if n_invalid:
        log(f"  Fixing {n_invalid:,} invalid geometries with make_valid() ...")
        gdf.loc[invalid_mask, "geometry"] = (
            gdf.loc[invalid_mask, "geometry"].apply(make_valid)
        )
    else:
        log("  All geometries valid.")

    # Drop any nulls that survived make_valid
    gdf = gdf[~gdf.geometry.isna() & gdf.geometry.notna()].copy()

    # ── Clip to study area ───────────────────────────────────────────────────
    log("Clipping to study area ...")
    try:
        clipped = gpd.clip(gdf, study_area)
    except Exception as exc:
        log(f"  gpd.clip failed ({exc}), falling back to intersection ...")
        # Fallback: manual intersection avoids TopologyException from clip
        study_geom = study_area.geometry.unary_union
        gdf["geometry"] = gdf.geometry.intersection(study_geom)
        clipped = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].copy()

    log(f"  {len(clipped):,} features after clip")

    if clipped.empty:
        log("ERROR: No features remain after clip. "
            "Do the buildings overlap the 9-county study area?")
        return

    # ── Buffer 150 ft ────────────────────────────────────────────────────────
    log(f"Buffering {BUFFER_FT} ft ...")
    clipped = clipped.copy()
    clipped["geometry"] = clipped.geometry.buffer(BUFFER_FT)

    # ── Union into a single exclusion polygon ────────────────────────────────
    log("Unioning all buffers into exclusion polygon ...")
    excl_geom = unary_union(clipped.geometry.tolist())
    log(f"  Union complete — {type(excl_geom).__name__}")

    # Clip union back to study area to remove buffer overhang
    study_geom = study_area.geometry.unary_union
    excl_geom  = excl_geom.intersection(study_geom)

    excl_acres = excl_geom.area / 43_560
    study_acres = study_geom.area / 43_560
    log(f"  Exclusion area: {excl_acres:,.0f} acres  "
        f"({100 * excl_acres / study_acres:.1f}% of study area)")

    result = gpd.GeoDataFrame({"geometry": [excl_geom]}, crs=CRS_PROJ)

    # ── Save EPSG:3435 shapefile ─────────────────────────────────────────────
    out_shp = OUTPUT_DIR / "residential_setback_EPSG3435.shp"
    result.to_file(str(out_shp))
    log(f"Saved: {out_shp}")

    # ── Save WGS84 GeoJSON (6 dp) ────────────────────────────────────────────
    out_geojson = OUTPUT_DIR / "residential_setback_WGS84.geojson"
    wgs = result.to_crs(CRS_WGS84).copy()
    # Round coordinates to 6 decimal places
    import re
    from shapely import wkt as shp_wkt
    def round_coords(geom, dp=6):
        return shp_wkt.loads(
            re.sub(r"-?\d+\.\d+", lambda m: str(round(float(m.group()), dp)), geom.wkt)
        )
    wgs["geometry"] = wgs.geometry.apply(round_coords)
    wgs.to_file(str(out_geojson), driver="GeoJSON")
    log(f"Saved: {out_geojson}")

    log("Done.")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    log("=" * 60)
    log("Residential Setback  |  150 ft  |  EPSG:3435")
    log("=" * 60)
    log("")
    log("[1/2] Loading county boundaries ...")
    study = load_study_area()
    log("")
    log("[2/2] Processing residential setback ...")
    run(study)
