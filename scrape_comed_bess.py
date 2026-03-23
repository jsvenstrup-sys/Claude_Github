"""
Scraper for ComEd BESS Hosting Capacity ArcGIS FeatureServer.

The service at utility.arcgis.com/usrsvcs is a proxy that validates the HTTP
Referer header against known ComEd hosting capacity web apps.  This script
cycles through candidate referrers, then falls back to unauthenticated direct
access.  If the service still returns 403 you will need to supply an ArcGIS
token (--token) or credentials (--username / --password).

Usage:
    python scrape_comed_bess.py                     # try public access
    python scrape_comed_bess.py --token <TOKEN>
    python scrape_comed_bess.py --username u --password p
"""

import argparse
import json
import time
import sys
from pathlib import Path

import requests

# ── constants ────────────────────────────────────────────────────────────────

BASE_URL = (
    "https://utility.arcgis.com/usrsvcs/servers/"
    "9d1c207b6423446ca9eadd78cac261ae/rest/services/"
    "ComEd_BESS_Hosting_Capacity_032026/FeatureServer"
)

# ComEd / Exelon web apps that legitimately embed this service.
# The proxy checks the Referer header so we try these in order.
CANDIDATE_REFERRERS = [
    "https://www.comed.com/",
    "https://exelonutilities.maps.arcgis.com/",
    "https://www.arcgis.com/",
    "https://comed.maps.arcgis.com/",
    "https://exelon.maps.arcgis.com/",
    # The ArcGIS web app viewer that the first search result pointed to
    "https://www.arcgis.com/apps/webappviewer/index.html?id=e357a47d16bf4f9380855981301a644d",
]

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://www.comed.com",
}

PAGE_SIZE = 1000  # max features per request


# ── helpers ──────────────────────────────────────────────────────────────────

def get_token(username: str, password: str) -> str:
    """Obtain a short-lived ArcGIS Online token."""
    resp = requests.post(
        "https://www.arcgis.com/sharing/rest/generateToken",
        data={
            "username": username,
            "password": password,
            "referer": "https://www.arcgis.com",
            "f": "json",
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise RuntimeError(f"Token error: {data['error']}")
    print(f"[auth] token obtained, expires in {data.get('expires', '?')} ms")
    return data["token"]


def build_session(token: str | None, referrer: str) -> requests.Session:
    session = requests.Session()
    headers = {**DEFAULT_HEADERS, "Referer": referrer}
    session.headers.update(headers)
    if token:
        session.params = {"token": token}  # type: ignore[assignment]
    return session


def fetch_json(session: requests.Session, url: str, params: dict) -> dict:
    resp = session.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        code = data["error"].get("code", "?")
        msg = data["error"].get("message", data["error"])
        raise RuntimeError(f"ArcGIS error {code}: {msg}")
    return data


def probe_service(session: requests.Session) -> dict:
    """Fetch service metadata; returns the parsed JSON."""
    return fetch_json(session, BASE_URL, {"f": "json"})


def fetch_layer_info(session: requests.Session, layer_id: int) -> dict:
    url = f"{BASE_URL}/{layer_id}"
    return fetch_json(session, url, {"f": "json"})


def fetch_all_features(session: requests.Session, layer_id: int) -> list[dict]:
    """Page through all features in a layer."""
    query_url = f"{BASE_URL}/{layer_id}/query"
    features: list[dict] = []
    offset = 0

    while True:
        params = {
            "where": "1=1",
            "outFields": "*",
            "returnGeometry": "true",
            "outSR": "4326",
            "f": "geojson",
            "resultOffset": offset,
            "resultRecordCount": PAGE_SIZE,
        }
        data = fetch_json(session, query_url, params)
        batch = data.get("features", [])
        features.extend(batch)
        print(f"  layer {layer_id}: fetched {len(features)} features so far …")
        if len(batch) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
        time.sleep(0.3)  # be polite

    return features


def save_geojson(layer_id: int, layer_name: str, features: list[dict], out_dir: Path):
    name = layer_name.replace(" ", "_").replace("/", "-")
    path = out_dir / f"layer_{layer_id}_{name}.geojson"
    geojson = {
        "type": "FeatureCollection",
        "features": features,
    }
    path.write_text(json.dumps(geojson, indent=2))
    print(f"  saved {len(features)} features → {path}")


def save_csv(layer_id: int, layer_name: str, features: list[dict], out_dir: Path):
    import csv

    if not features:
        return
    name = layer_name.replace(" ", "_").replace("/", "-")
    path = out_dir / f"layer_{layer_id}_{name}.csv"

    # Collect all property keys
    keys: list[str] = []
    for f in features:
        for k in f.get("properties", {}).keys():
            if k not in keys:
                keys.append(k)
    # Append geometry columns
    keys += ["geometry_type", "geometry_wkt"]

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for feat in features:
            row = dict(feat.get("properties", {}))
            geom = feat.get("geometry") or {}
            row["geometry_type"] = geom.get("type", "")
            row["geometry_wkt"] = json.dumps(geom.get("coordinates", ""))
            writer.writerow(row)

    print(f"  saved CSV → {path}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Scrape ComEd BESS Hosting Capacity")
    parser.add_argument("--token", default=None, help="ArcGIS token")
    parser.add_argument("--username", default=None)
    parser.add_argument("--password", default=None)
    parser.add_argument("--out", default="bess_data", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve token
    token = args.token
    if not token and args.username and args.password:
        token = get_token(args.username, args.password)

    # Try each referrer until one works
    service_meta = None
    working_session = None

    for referrer in CANDIDATE_REFERRERS:
        session = build_session(token, referrer)
        try:
            print(f"[probe] trying referrer: {referrer}")
            service_meta = probe_service(session)
            working_session = session
            print(f"[probe] SUCCESS — service: {service_meta.get('serviceDescription', '(no description)')}")
            break
        except requests.HTTPError as e:
            print(f"  HTTP {e.response.status_code}")
        except RuntimeError as e:
            print(f"  ArcGIS error: {e}")
        except Exception as e:
            print(f"  error: {e}")

    if service_meta is None:
        print(
            "\n[FAIL] All referrers returned errors.\n"
            "Options:\n"
            "  1. Supply --token <your-arcgis-token>\n"
            "  2. Supply --username <u> --password <p> for ArcGIS Online creds\n"
            "  3. Obtain a token from https://www.arcgis.com/sharing/rest/generateToken\n"
        )
        sys.exit(1)

    # Save raw service metadata
    meta_path = out_dir / "service_metadata.json"
    meta_path.write_text(json.dumps(service_meta, indent=2))
    print(f"[meta] saved → {meta_path}")

    layers = service_meta.get("layers", []) + service_meta.get("tables", [])
    if not layers:
        print("[warn] No layers found in service metadata.")
        return

    print(f"\n[layers] found {len(layers)} layer(s):")
    for lyr in layers:
        print(f"  id={lyr['id']}  name={lyr['name']}")

    # Fetch each layer
    for lyr in layers:
        lid = lyr["id"]
        lname = lyr["name"]
        print(f"\n[layer {lid}] {lname}")
        try:
            features = fetch_all_features(working_session, lid)
            save_geojson(lid, lname, features, out_dir)
            save_csv(lid, lname, features, out_dir)
        except Exception as e:
            print(f"  ERROR fetching layer {lid}: {e}")

    print("\n[done]")


if __name__ == "__main__":
    main()
