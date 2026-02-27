"""
injection_updater_V4.py
Weekly incremental updater for TexNet disposal well injection data.

Fetches the last 30 days of data from the TexNet API, appends and
deduplicates it into the raw CSVs, then regenerates the regularized
GIST output files from the full combined history.

Arguments:
    --target-dir: Directory where all CSV output files will be written (this is always the same in production but gives us a way to test the script in a local environment)
    --dev: Append '_dev' suffix to all output filenames (safe testing mode so we don't accidentally modify production data)
    --debug: Enable verbose DEBUG logging

Usage:
    python injection_updater_V4.py --target-dir ./src/data
    python injection_updater_V4.py --target-dir ./src/data --dev
    python injection_updater_V4.py --target-dir ./src/data --debug --dev
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from io import StringIO
from logging.handlers import RotatingFileHandler
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import urllib3

import credentials
import injectionV3 as inj3

# Suppress SSL warnings (TexNet API uses self-signed cert)
requests.packages.urllib3.disable_warnings(
    requests.packages.urllib3.exceptions.InsecureRequestWarning
)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

def setup_logging(debug: bool) -> None:
    """
    Configure root logger with:
      - RotatingFileHandler: injection_updater_V4.log, 3 backups, 30 MB each
      - StreamHandler: mirrors output to console
    Debug flag lowers level to DEBUG; default is INFO.
    """
    log_level = logging.DEBUG if debug else logging.INFO
    log_file = Path(__file__).parent / "injection_updater_V4.log"
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    file_handler = RotatingFileHandler(
        log_file, maxBytes=30 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(log_level)
    root.addHandler(file_handler)
    root.addHandler(console_handler)


# ──────────────────────────────────────────────────────────────────────────────
# Path resolution
# ──────────────────────────────────────────────────────────────────────────────

def resolve_path(target_dir: Path, basename: str, dev: bool) -> Path:
    """
    Build an output file path, optionally injecting a '_dev' suffix.

    Example:
        resolve_path(Path('./src/data'), 'disposal_well.csv', dev=True)
        → Path('./src/data/disposal_well_dev.csv')
    """
    if dev:
        stem, ext = basename.rsplit(".", 1)
        basename = f"{stem}_dev.{ext}"
    return target_dir / basename


# ──────────────────────────────────────────────────────────────────────────────
# API helpers
# ──────────────────────────────────────────────────────────────────────────────

def authenticate(username: str, password: str, auth_url: str) -> str | None:
    """
    Authenticate with the TexNet API and return a bearer token.
    Returns None and logs an error on failure.
    """
    try:
        response = requests.post(
            auth_url,
            json={"username": username, "password": password},
            verify=False,
        )
        response.raise_for_status()
        token = response.json()["Token"]
        logger.info("Authentication successful.")
        return token
    except requests.exceptions.RequestException as e:
        logger.error("Authentication failed: %s", e)
        return None


def fetch_data(
    api_url: str,
    token: str,
    method: str = "GET",
    data=None,
    json_payload=None,
    params=None,
) -> str | None:
    """
    Fetch CSV text from the TexNet API endpoint.
    Returns the raw response text, or None on failure.
    """
    try:
        headers = {"Authorization": f"Bearer {token}"}
        if method.upper() == "POST" and json_payload is not None:
            headers["Content-Type"] = "application/json"

        if method.upper() == "GET":
            response = requests.get(api_url, headers=headers, params=params, verify=False)
        elif method.upper() == "POST":
            response = requests.post(
                api_url, headers=headers, data=data, json=json_payload, params=params, verify=False
            )
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")

        response.raise_for_status()
        logger.debug("Fetched %d bytes from %s", len(response.content), api_url)
        return response.text

    except requests.exceptions.RequestException as e:
        logger.error("API request to %s failed: %s", api_url, e)
        return None
    except ValueError as e:
        logger.error("fetch_data error: %s", e)
        return None


# ──────────────────────────────────────────────────────────────────────────────
# CSV update helpers (append + dedup)
# ──────────────────────────────────────────────────────────────────────────────

def update_well_csv(api_text: str, path: Path) -> pd.DataFrame:
    """
    Merge newly fetched well data with the existing CSV (if present).
    Deduplicates on 'Id'; the API value wins on conflict (kept last).
    Saves the merged result back to path and returns the DataFrame.
    """
    new_df = pd.read_csv(StringIO(api_text), low_memory=False)

    if path.exists():
        existing_df = pd.read_csv(path, low_memory=False)
        logger.debug("Loaded %d existing well rows from %s", len(existing_df), path)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        logger.info("No existing well file at %s — creating new.", path)
        combined = new_df

    # Keep last occurrence so the fresh API value wins
    combined.drop_duplicates(subset=["Id"], keep="last", inplace=True)
    combined.to_csv(path, index=False)
    logger.info("Well CSV updated: %d wells → %s", len(combined), path)
    return combined


def update_inj_csv(api_text: str, path: Path) -> None:
    """
    Merge newly fetched injection data with the existing CSV (if present).
    Deduplicates on ('Id', 'Date of Injection'); API value wins.
    Saves the merged result back to path.
    """
    new_df = pd.read_csv(StringIO(api_text), low_memory=False)

    if path.exists():
        existing_df = pd.read_csv(path, low_memory=False)
        logger.debug("Loaded %d existing injection rows from %s", len(existing_df), path)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        logger.info("No existing injection file at %s — creating new.", path)
        combined = new_df

    before = len(combined)
    combined.drop_duplicates(
        subset=["Id", "Date of Injection"], keep="last", inplace=True
    )
    logger.info(
        "Injection CSV updated: %d rows (removed %d duplicates) → %s",
        len(combined), before - len(combined), path,
    )
    combined.to_csv(path, index=False)


# ──────────────────────────────────────────────────────────────────────────────
# B3 format transformations
# ──────────────────────────────────────────────────────────────────────────────

WELL_HEADER_MAP = {
    "Id": "InjectionWellId",
    "Apinumber": "APINumber",
    "Uicnumber": "UICNumber",
    "SurfaceLatitude": "SurfaceHoleLatitude",
    "SurfaceLongitude": "SurfaceHoleLongitude",
    "OriginalPermitDate": "WellActivatedDate",
    "TotalBpdmax": "PermittedMaxLiquidBPD",
    "InjectionBottomInterval": "PermittedIntervalBottomFt",
    "InjectionTopInterval": "PermittedIntervalTopFt",
    "WellClassification": "CompletedWellDepthClassification",
}

INJ_HEADER_MAP = {
    "Id": "InjectionWellId",
    "Date of Injection": "Date",
    "Volume Injected (BBLs)": "InjectedLiquidBBL",
}

# Column rename applied to GIST well output after injTX writes it
GIST_WELL_MAP = {"InjectionWellId": "ID"}


def well_to_b3_format(input_path: Path, output_path: Path) -> None:
    """
    Transform raw well CSV to B3 format.
    Combines LeaseName + WellNumber into WellName, renames columns.
    """
    df = pd.read_csv(input_path, low_memory=False)
    df["WellName"] = df["LeaseName"].astype(str) + " " + df["WellNumber"].astype(str)
    df.drop(columns=["LeaseName", "WellNumber"], inplace=True)
    df.rename(columns=WELL_HEADER_MAP, inplace=True)
    df.to_csv(output_path, index=False)
    logger.info("B3 well file written: %d rows → %s", len(df), output_path)


def inj_to_b3_format(input_path: Path, output_path: Path) -> None:
    """Transform raw injection CSV to B3 format (column rename only)."""
    df = pd.read_csv(input_path, low_memory=False)
    df.rename(columns=INJ_HEADER_MAP, inplace=True)
    df.to_csv(output_path, index=False)
    logger.info("B3 injection file written: %d rows → %s", len(df), output_path)


def reformat_well_id_column(path: Path) -> None:
    """Rename 'InjectionWellId' → 'ID' in an already-written GIST well file (in place)."""
    df = pd.read_csv(path, low_memory=False)
    df.rename(columns=GIST_WELL_MAP, inplace=True)
    df.to_csv(path, index=False)


# ──────────────────────────────────────────────────────────────────────────────
# GIST processing pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_gist_pipeline(
    b3_well_file: Path,
    b3_inj_file: Path,
    well_file: Path,
    inj_file: Path,
    depth_label: str,
    depth_cutoff: float,
    end_date_str: str,
    verbose: int,
) -> None:
    """
    Run the full injTX → inj → processRates → outputReg pipeline for one
    depth class (Shallow or Deep).

    Parameters
    ----------
    b3_well_file  : path to the B3-format well CSV
    b3_inj_file   : path to the B3-format injection CSV
    well_file     : output path for the GIST well CSV
    inj_file      : output path for the GIST injection CSV
    depth_label   : 'Shallow' or 'Deep'
    depth_cutoff  : depth threshold in feet (7000.0)
    end_date_str  : end date string 'MM-DD-YYYY' passed to processRates
    verbose       : 0 = silent, 1 = info, 2 = debug (injectionV3 internal prints)
    """
    logger.info("Running GIST pipeline for %s wells…", depth_label)

    tx_inj = inj3.injTX(str(b3_well_file), depth_label, depth_cutoff, verbose=verbose)
    logger.debug("%s: injTX initialised with %d wells", depth_label, len(tx_inj.wellList))

    tx_inj.addDaily(str(b3_inj_file), 100_000, verbose=verbose)
    logger.debug("%s: addDaily complete", depth_label)

    wells = inj3.inj(None, tx_inj, "01-01-1970", str(well_file), verbose=verbose)
    reformat_well_id_column(well_file)

    wells.processRates(200_000.0, 10, end_date_str, False, verbose=verbose)
    logger.debug("%s: processRates complete", depth_label)

    wells.outputReg(str(inj_file), verbose=verbose)
    logger.info("%s GIST files written: wells → %s | injection → %s", depth_label, well_file, inj_file)


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Weekly incremental updater for TexNet injection data."
    )
    parser.add_argument(
        "--target-dir",
        required=True,
        type=Path,
        help="Directory where all CSV output files will be written.",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        default=False,
        help="Append '_dev' suffix to all output filenames (safe testing mode).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable verbose DEBUG logging and injectionV3 internal output.",
    )
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Orchestrate the weekly data fetch, merge, dedup, and GIST regeneration."""
    args = parse_args()
    setup_logging(args.debug)

    logger.info("=" * 60)
    logger.info("injection_updater_V4 started")
    logger.info("target-dir: %s | dev: %s | debug: %s", args.target_dir, args.dev, args.debug)

    # ── Validate target directory ────────────────────────────────────────────
    if not args.target_dir.exists():
        logger.error("Target directory does not exist: %s", args.target_dir)
        sys.exit(1)

    dev = args.dev
    tdir = args.target_dir

    # ── Resolve all file paths ───────────────────────────────────────────────
    well_raw      = resolve_path(tdir, "disposal_well.csv",             dev)
    inj_raw       = resolve_path(tdir, "disposal_inj.csv",              dev)
    well_b3       = resolve_path(tdir, "disposal_well_b3_format.csv",   dev)
    inj_b3        = resolve_path(tdir, "disposal_inj_b3_format.csv",    dev)
    shallow_well  = resolve_path(tdir, "gist_well_shallow.csv",         dev)
    shallow_inj   = resolve_path(tdir, "gist_injection_shallow.csv",    dev)
    deep_well     = resolve_path(tdir, "gist_well_deep.csv",            dev)
    deep_inj      = resolve_path(tdir, "gist_injection_deep.csv",       dev)

    logger.debug("Resolved paths: %s", {
        "well_raw": well_raw, "inj_raw": inj_raw,
        "well_b3": well_b3,   "inj_b3": inj_b3,
    })

    # ── Date range: last 30 days ─────────────────────────────────────────────
    now = datetime.now()
    start = now - timedelta(days=30)
    logger.info("Fetching injection data from %s to %s", start.strftime("%Y-%m-%d"), now.strftime("%Y-%m-%d"))

    # ── Authenticate ─────────────────────────────────────────────────────────
    auth_url = "https://injection.texnet.beg.utexas.edu/api/Users/Authenticate"
    token = authenticate(credentials.USERNAME, credentials.PASSWORD, auth_url)
    if not token:
        logger.error("Cannot proceed without a valid API token. Exiting.")
        sys.exit(1)

    # ── Step 1: Fetch & update well list ─────────────────────────────────────
    well_url = "https://injection.texnet.beg.utexas.edu/api/well/wellswithinjectioncsv"
    logger.info("Fetching well list from API…")
    well_text = fetch_data(well_url, token)
    if not well_text:
        logger.error("Failed to fetch well data. Exiting.")
        sys.exit(1)

    # Filter out wells with invalid coordinates before merging
    raw_well_df = pd.read_csv(StringIO(well_text), low_memory=False)
    valid_wells = raw_well_df[
        (raw_well_df["SurfaceLatitude"] != 0) & (raw_well_df["SurfaceLongitude"] != 0)
    ]
    logger.info(
        "Well filter: %d total → %d with valid coordinates",
        len(raw_well_df), len(valid_wells),
    )
    well_df = update_well_csv(valid_wells.to_csv(index=False), well_raw)

    # ── Step 2: Build injection payload from valid well IDs ──────────────────
    id_array = well_df["Id"].to_numpy()
    payload = {
        "BeginMonth": start.month,
        "BeginYear":  start.year,
        "EndMonth":   now.month,
        "EndYear":    now.year,
        "Format":     "excel",
        "IncludeWellIds": True,
        "WellIds":    id_array.tolist(),
    }

    # ── Step 3: Fetch & update injection data ────────────────────────────────
    inj_url = "https://injection.texnet.beg.utexas.edu/api/Export"
    logger.info("Fetching injection data from API…")
    inj_text = fetch_data(inj_url, token, method="POST", json_payload=payload)
    if not inj_text:
        logger.error("Failed to fetch injection data. Exiting.")
        sys.exit(1)

    update_inj_csv(inj_text, inj_raw)

    # ── Step 4: Transform to B3 format ───────────────────────────────────────
    logger.info("Transforming raw CSVs to B3 format…")
    well_to_b3_format(well_raw, well_b3)
    inj_to_b3_format(inj_raw, inj_b3)

    # ── Step 5: Run GIST pipeline (Shallow + Deep) ───────────────────────────
    verbose = 1 if args.debug else 0
    end_date_str = now.strftime("%m-%d-%Y")

    run_gist_pipeline(
        b3_well_file=well_b3,
        b3_inj_file=inj_b3,
        well_file=shallow_well,
        inj_file=shallow_inj,
        depth_label="Shallow",
        depth_cutoff=7000.0,
        end_date_str=end_date_str,
        verbose=verbose,
    )

    run_gist_pipeline(
        b3_well_file=well_b3,
        b3_inj_file=inj_b3,
        well_file=deep_well,
        inj_file=deep_inj,
        depth_label="Deep",
        depth_cutoff=7000.0,
        end_date_str=end_date_str,
        verbose=verbose,
    )

    logger.info("injection_updater_V4 completed successfully.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
