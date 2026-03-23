#!/usr/bin/env python3
"""
Ingest parcel data from a local shapefile and export to CSV.

Usage:
    python ingest_parcels.py <path_to_shapefile> [options]

Examples:
    python ingest_parcels.py /data/parcels/parcels.shp
    python ingest_parcels.py /data/parcels/parcels.shp --output parcels_out.csv
    python ingest_parcels.py /data/parcels/parcels.shp --crs EPSG:4326 --chunk-size 5000
"""

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ingest parcel data from a shapefile and export to CSV."
    )
    parser.add_argument("shapefile", type=Path, help="Path to the input .shp file")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output CSV path (default: <shapefile_name>_parcels.csv beside the input file)",
    )
    parser.add_argument(
        "--crs",
        default="EPSG:4326",
        help="Target CRS to reproject geometry into (default: EPSG:4326)",
    )
    parser.add_argument(
        "--geometry-col",
        default="geometry_wkt",
        help="Column name for WKT geometry in the output CSV (default: geometry_wkt)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="If set, read and write in chunks of this many rows (useful for large files)",
    )
    parser.add_argument(
        "--no-geometry",
        action="store_true",
        help="Drop geometry column from output (attributes only)",
    )
    return parser.parse_args()


def default_output_path(shapefile: Path) -> Path:
    return shapefile.parent / f"{shapefile.stem}_parcels.csv"


def load_shapefile(shapefile: Path, target_crs: str) -> gpd.GeoDataFrame:
    print(f"Reading shapefile: {shapefile}")
    gdf = gpd.read_file(shapefile)
    print(f"  Loaded {len(gdf):,} parcels | CRS: {gdf.crs}")

    if gdf.crs is None:
        print("  WARNING: Shapefile has no CRS defined. Skipping reprojection.")
    elif str(gdf.crs) != target_crs:
        print(f"  Reprojecting from {gdf.crs} -> {target_crs}")
        gdf = gdf.to_crs(target_crs)

    return gdf


def gdf_to_csv(
    gdf: gpd.GeoDataFrame,
    output: Path,
    geometry_col: str,
    no_geometry: bool,
):
    df = pd.DataFrame(gdf)

    if no_geometry:
        df = df.drop(columns=["geometry"], errors="ignore")
    else:
        # Convert geometry objects to WKT strings
        df[geometry_col] = gdf.geometry.to_wkt()
        df = df.drop(columns=["geometry"], errors="ignore")

    df.to_csv(output, index=False)
    print(f"  Wrote {len(df):,} rows -> {output}")


def load_and_export_chunked(
    shapefile: Path,
    output: Path,
    target_crs: str,
    geometry_col: str,
    no_geometry: bool,
    chunk_size: int,
):
    """Memory-efficient chunked read using fiona row offsets."""
    import fiona

    with fiona.open(shapefile) as src:
        total = len(src)
        source_crs = src.crs_wkt
        print(f"Reading shapefile: {shapefile}")
        print(f"  Total features: {total:,} | CRS: {source_crs}")

        header_written = False
        offset = 0
        chunk_num = 0

        while offset < total:
            chunk_num += 1
            end = min(offset + chunk_size, total)
            print(f"  Processing chunk {chunk_num}: rows {offset}-{end - 1}")

            gdf = gpd.read_file(shapefile, rows=slice(offset, end))

            if gdf.crs is None:
                print("  WARNING: Shapefile has no CRS. Skipping reprojection.")
            elif str(gdf.crs) != target_crs:
                gdf = gdf.to_crs(target_crs)

            df = pd.DataFrame(gdf)
            if no_geometry:
                df = df.drop(columns=["geometry"], errors="ignore")
            else:
                df[geometry_col] = gdf.geometry.to_wkt()
                df = df.drop(columns=["geometry"], errors="ignore")

            mode = "w" if not header_written else "a"
            df.to_csv(output, index=False, mode=mode, header=not header_written)
            header_written = True
            offset += chunk_size

    print(f"  Done -> {output}")


def main():
    args = parse_args()

    shapefile = args.shapefile.resolve()
    if not shapefile.exists():
        print(f"ERROR: Shapefile not found: {shapefile}", file=sys.stderr)
        sys.exit(1)
    if shapefile.suffix.lower() != ".shp":
        print(f"WARNING: Expected a .shp file, got: {shapefile.suffix}")

    output = args.output or default_output_path(shapefile)
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.chunk_size:
        load_and_export_chunked(
            shapefile=shapefile,
            output=output,
            target_crs=args.crs,
            geometry_col=args.geometry_col,
            no_geometry=args.no_geometry,
            chunk_size=args.chunk_size,
        )
    else:
        gdf = load_shapefile(shapefile, args.crs)
        gdf_to_csv(gdf, output, args.geometry_col, args.no_geometry)

    print("Ingestion complete.")


if __name__ == "__main__":
    main()
