"""
get_rrc_well_injection_data.py
Fetches RRC UIC well location and H10 injection monitoring data from the
Texas Open Data Portal (Socrata SODA2 API), maps them to the B3 format
expected by injectionV3, and runs the GIST pipeline to produce regularized
shallow and deep injection output files.

Data sources:
  - Wells:     https://data.texas.gov/resource/givw-z9t4  (RRC-UIC well location)
  - Injection: https://data.texas.gov/resource/qq2j-f2zm  (RRC-UIC H10 Injection Monitoring)

Arguments:
    --target-dir : Directory where all CSV output files will be written
    --days       : Look-back window in days for injection data (required)
    --dev        : Append '_dev' suffix to all output filenames (safe testing mode)
    --debug      : Enable verbose DEBUG logging
    --backup-dir : Directory to back up existing CSVs before updating

Usage:
    python get_rrc_well_injection_data.py --target-dir ./src/data --days 90
    python get_rrc_well_injection_data.py --target-dir ./src/data --days 30 --dev
    python get_rrc_well_injection_data.py --target-dir ./src/data --days 365 --debug
"""

import argparse
import logging
import shutil
import sys
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import List, Optional

import pandas as pd
from sodapy import Socrata

import credentials
import injectionV3 as inj3

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

SOCRATA_DOMAIN = "data.texas.gov"
WELLS_DATASET_ID = "givw-z9t4"
INJECTION_DATASET_ID = "qq2j-f2zm"

# Depth cutoff (feet) separating Shallow from Deep wells
DEPTH_CUTOFF_FT = 7000.0

# Page size for sodapy get_all() calls (records per internal request)
PAGE_SIZE = 10_000

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

def setup_logging(debug: bool) -> None:
    """
    Configure root logger with:
      - RotatingFileHandler: get_rrc_well_injection_data.log, 3 backups, 30 MB each
      - StreamHandler: mirrors output to console
    Debug flag lowers level to DEBUG; default is INFO.
    """
    log_level = logging.DEBUG if debug else logging.INFO
    log_file = Path(__file__).parent / "get_rrc_well_injection_data.log"
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    file_handler = RotatingFileHandler(
        log_file, maxBytes=30 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    console_handler.setLevel(log_level)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(file_handler)
    root.addHandler(console_handler)


# ──────────────────────────────────────────────────────────────────────────────
# Path resolution
# ──────────────────────────────────────────────────────────────────────────────

def resolve_path(target_dir: Path, basename: str, dev: bool) -> Path:
    """
    Build an output file path, optionally injecting a '_dev' suffix.

    Example:
        resolve_path(Path('./src/data'), 'rrc_disposal_well.csv', dev=True)
        → Path('./src/data/rrc_disposal_well_dev.csv')
    """
    if dev:
        stem, ext = basename.rsplit(".", 1)
        basename = f"{stem}_dev.{ext}"
    return target_dir / basename


# ──────────────────────────────────────────────────────────────────────────────
# Backup
# ──────────────────────────────────────────────────────────────────────────────

def backup_files(files: List[Path], backup_dir: Path) -> None:
    """Copy a list of existing files to the backup directory."""
    if not backup_dir.exists():
        backup_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Created backup directory: %s", backup_dir)

    for f in files:
        if f.exists():
            dest = backup_dir / f.name
            shutil.copy2(f, dest)
            logger.info("Backed up %s → %s", f.name, dest)
        else:
            logger.debug("File %s does not exist, skipping backup.", f.name)


# ──────────────────────────────────────────────────────────────────────────────
# Socrata client
# ──────────────────────────────────────────────────────────────────────────────

def build_socrata_client(username: str, password: str) -> Socrata:
    """
    Create an authenticated Socrata client for data.texas.gov.
    App token is None — username/password auth is sufficient for these datasets.
    """
    client = Socrata(
        SOCRATA_DOMAIN,
        None,           # no app token; auth via username/password
        username=username,
        password=password,
        timeout=120,
    )
    logger.info("Socrata client created for domain: %s", SOCRATA_DOMAIN)
    return client


# ──────────────────────────────────────────────────────────────────────────────
# Data fetch
# ──────────────────────────────────────────────────────────────────────────────

def fetch_wells(client: Socrata) -> Optional[pd.DataFrame]:
    """
    Fetch all RRC UIC well location records (dataset givw-z9t4).
    Uses get_all() which paginates automatically.
    Returns a DataFrame, or None on failure.
    """
    try:
        logger.info("Fetching all wells from dataset %s …", WELLS_DATASET_ID)
        records = client.get_all(WELLS_DATASET_ID, limit=PAGE_SIZE)
        df = pd.DataFrame.from_records(records)
        logger.info("Fetched %d well records.", len(df))
        return df
    except Exception as e:
        logger.error("Failed to fetch wells: %s", e)
        return None


def fetch_injection(client: Socrata, days: int) -> Optional[pd.DataFrame]:
    """
    Fetch RRC H10 injection monitoring records (dataset qq2j-f2zm) for the
    last `days` days, using a $where date filter to keep the result set
    manageable. Uses get_all() which paginates automatically.
    Returns a DataFrame, or None on failure.
    """
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S")
    where_clause = f"formatted_date >= '{cutoff}'"
    try:
        logger.info(
            "Fetching injection records from dataset %s where %s …",
            INJECTION_DATASET_ID, where_clause,
        )
        records = client.get_all(
            INJECTION_DATASET_ID,
            where=where_clause,
            limit=PAGE_SIZE,
        )
        df = pd.DataFrame.from_records(records)
        logger.info("Fetched %d injection records.", len(df))
        return df
    except Exception as e:
        logger.error("Failed to fetch injection data: %s", e)
        return None


# ──────────────────────────────────────────────────────────────────────────────
# B3 format transformations
# ──────────────────────────────────────────────────────────────────────────────

def _parse_date_to_b3(series: pd.Series) -> pd.Series:
    """
    Parse a Socrata floating_timestamp column (ISO 8601 strings such as
    '2023-06-01T00:00:00.000' or '2023-06-01') to 'MM-DD-YYYY' strings
    as expected by the B3 / injectionV3 format.
    Unparseable values become '01-01-1970'.
    """
    def _convert(val):
        if pd.isnull(val) or str(val).strip() == "":
            return "01-01-1970"
        try:
            # pd.to_datetime handles ISO 8601 variants and epoch ms strings
            dt = pd.to_datetime(val, errors="coerce")
            if pd.isnull(dt):
                return "01-01-1970"
            return dt.strftime("%m-%d-%Y")
        except Exception:
            return "01-01-1970"

    return series.apply(_convert)


def _to_numeric_safe(series: pd.Series, default=0) -> pd.Series:
    """Coerce a series to numeric, filling errors with `default`."""
    return pd.to_numeric(series, errors="coerce").fillna(default)


def map_wells_to_b3(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map raw well DataFrame (from givw-z9t4) to the B3 column schema
    expected by injectionV3.injTX.

    Expected source columns (SODA field names):
        uic_number, api_no, lease_name, well_no_display,
        latitude_nad83, longitude_nad83, h1_date,
        top_inj_zone, bot_inj_zone

    B3 output columns:
        InjectionWellId, UICNumber, APINumber, WellName,
        SurfaceHoleLatitude, SurfaceHoleLongitude,
        WellActivatedDate, PermittedMaxLiquidBPD,
        PermittedIntervalTopFt, PermittedIntervalBottomFt,
        CompletedWellDepthClassification
    """
    required = [
        "uic_number", "api_no", "lease_name", "well_no_display",
        "latitude_nad83", "longitude_nad83", "h1_date",
        "top_inj_zone", "bot_inj_zone",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.warning(
            "Wells DataFrame is missing expected columns: %s. "
            "Available columns: %s",
            missing, list(df.columns),
        )

    out = pd.DataFrame()

    # Primary key and UIC number
    out["InjectionWellId"] = df.get("uic_number", pd.Series(dtype=str)).astype(str)
    out["UICNumber"] = out["InjectionWellId"]
    out["APINumber"] = df.get("api_no", pd.Series(dtype=str)).astype(str).str.strip()

    # Well name: lease_name + well_no_display
    lease = df.get("lease_name", pd.Series([""] * len(df))).fillna("").astype(str)
    well_no = df.get("well_no_display", pd.Series([""] * len(df))).fillna("").astype(str)
    out["WellName"] = (lease + " " + well_no).str.strip()

    # Coordinates — filter rows with no valid location later
    out["SurfaceHoleLatitude"] = _to_numeric_safe(
        df.get("latitude_nad83", pd.Series(dtype=float)), default=0.0
    )
    out["SurfaceHoleLongitude"] = _to_numeric_safe(
        df.get("longitude_nad83", pd.Series(dtype=float)), default=0.0
    )

    # Permit / activation date from H1 form date
    out["WellActivatedDate"] = _parse_date_to_b3(
        df.get("h1_date", pd.Series(dtype=str))
    )

    # No direct max BPD field available; default to 0
    out["PermittedMaxLiquidBPD"] = 0

    # Injection interval depths
    out["PermittedIntervalTopFt"] = _to_numeric_safe(
        df.get("top_inj_zone", pd.Series(dtype=float)), default=0.0
    ).astype(int)
    out["PermittedIntervalBottomFt"] = _to_numeric_safe(
        df.get("bot_inj_zone", pd.Series(dtype=float)), default=0.0
    ).astype(int)

    # Depth classification derived from bot_inj_zone vs the 7000 ft cutoff
    out["CompletedWellDepthClassification"] = out["PermittedIntervalBottomFt"].apply(
        lambda d: "Deep" if d >= DEPTH_CUTOFF_FT else "Shallow"
    )

    logger.info("Mapped %d wells to B3 format.", len(out))
    return out


def map_injection_to_b3(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map raw injection DataFrame (from qq2j-f2zm) to the B3 column schema
    expected by injectionV3.injTX.addDaily.

    Expected source columns (SODA field names):
        uic_no, formatted_date, vol_liq

    B3 output columns:
        InjectionWellId, Date, InjectedLiquidBBL
    """
    required = ["uic_no", "formatted_date", "vol_liq"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.warning(
            "Injection DataFrame is missing expected columns: %s. "
            "Available columns: %s",
            missing, list(df.columns),
        )

    out = pd.DataFrame()
    out["InjectionWellId"] = df.get("uic_no", pd.Series(dtype=str)).astype(str)
    out["Date"] = _parse_date_to_b3(df.get("formatted_date", pd.Series(dtype=str)))
    out["InjectedLiquidBBL"] = _to_numeric_safe(
        df.get("vol_liq", pd.Series(dtype=float)), default=0.0
    )

    logger.info("Mapped %d injection records to B3 format.", len(out))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# CSV update helpers (append + dedup)
# ──────────────────────────────────────────────────────────────────────────────

def update_csv(new_df: pd.DataFrame, path: Path, dedup_cols: List[str]) -> pd.DataFrame:
    """
    Merge new_df with the existing CSV at path (if present).
    Deduplicates on dedup_cols; the new value wins (kept last).
    Saves the merged result back to path and returns the DataFrame.
    """
    if path.exists():
        existing_df = pd.read_csv(path, low_memory=False)
        logger.debug("Loaded %d existing rows from %s", len(existing_df), path)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        logger.info("No existing file at %s — creating new.", path)
        combined = new_df.copy()

    before = len(combined)
    combined.drop_duplicates(subset=dedup_cols, keep="last", inplace=True)
    combined.to_csv(path, index=False)
    logger.info(
        "CSV updated: %d rows (removed %d duplicates) → %s",
        len(combined), before - len(combined), path,
    )
    return combined


# ──────────────────────────────────────────────────────────────────────────────
# GIST processing pipeline
# ──────────────────────────────────────────────────────────────────────────────

def reformat_well_id_column(path: Path) -> None:
    """Rename 'InjectionWellId' → 'ID' in an already-written GIST well file (in place)."""
    df = pd.read_csv(path, low_memory=False)
    df.rename(columns={"InjectionWellId": "ID"}, inplace=True)
    df.to_csv(path, index=False)


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
    logger.info(
        "%s GIST files written: wells → %s | injection → %s",
        depth_label, well_file, inj_file,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fetch RRC UIC well/injection data from Texas Open Data Portal and run GIST pipeline."
    )
    parser.add_argument(
        "--target-dir",
        required=True,
        type=Path,
        help="Directory where all CSV output files will be written.",
    )
    parser.add_argument(
        "--days",
        required=True,
        type=int,
        help="Number of days to look back when fetching injection monitoring data.",
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
    parser.add_argument(
        "--backup-dir",
        type=Path,
        help="Directory where existing CSV files will be backed up before updating.",
    )
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Orchestrate the RRC data fetch, merge, dedup, transform, and GIST pipeline."""
    args = parse_args()
    setup_logging(args.debug)

    logger.info("=" * 60)
    logger.info("get_rrc_well_injection_data started")
    logger.info(
        "target-dir: %s | days: %s | dev: %s | debug: %s",
        args.target_dir, args.days, args.dev, args.debug,
    )

    # ── Validate target directory ────────────────────────────────────────────
    if not args.target_dir.exists():
        logger.error("Target directory does not exist: %s", args.target_dir)
        sys.exit(1)

    dev = args.dev
    tdir = args.target_dir

    # ── Resolve all file paths ───────────────────────────────────────────────
    well_raw        = resolve_path(tdir, "rrc_disposal_well.csv",            dev)
    inj_raw         = resolve_path(tdir, "rrc_disposal_inj.csv",             dev)
    well_b3         = resolve_path(tdir, "rrc_disposal_well_b3_format.csv",  dev)
    inj_b3          = resolve_path(tdir, "rrc_disposal_inj_b3_format.csv",   dev)
    shallow_well    = resolve_path(tdir, "rrc_gist_well_shallow.csv",        dev)
    shallow_inj     = resolve_path(tdir, "rrc_gist_injection_shallow.csv",   dev)
    deep_well       = resolve_path(tdir, "rrc_gist_well_deep.csv",           dev)
    deep_inj        = resolve_path(tdir, "rrc_gist_injection_deep.csv",      dev)

    logger.debug(
        "Resolved paths: well_raw=%s | inj_raw=%s | well_b3=%s | inj_b3=%s",
        well_raw, inj_raw, well_b3, inj_b3,
    )

    # ── Step 0: Backup existing files ────────────────────────────────────────
    if args.backup_dir:
        files_to_backup = [
            well_raw, inj_raw, well_b3, inj_b3,
            shallow_well, shallow_inj, deep_well, deep_inj,
        ]
        backup_files(files_to_backup, args.backup_dir)

    # ── Build Socrata client ─────────────────────────────────────────────────
    client = build_socrata_client(credentials.RRC_USERNAME, credentials.RRC_PASSWORD)

    # ── Step 1: Fetch wells (full dataset) ───────────────────────────────────
    logger.info("Step 1: Fetching well locations…")
    raw_well_df = fetch_wells(client)
    if raw_well_df is None or raw_well_df.empty:
        logger.error("No well data returned. Exiting.")
        sys.exit(1)

    # Filter out wells with missing coordinates
    lat = pd.to_numeric(raw_well_df.get("latitude_nad83"), errors="coerce")
    lon = pd.to_numeric(raw_well_df.get("longitude_nad83"), errors="coerce")
    valid_mask = lat.notna() & lon.notna() & (lat != 0) & (lon != 0)
    valid_well_df = raw_well_df[valid_mask].copy()
    logger.info(
        "Well coordinate filter: %d total → %d with valid coordinates",
        len(raw_well_df), len(valid_well_df),
    )

    # ── Step 2: Map wells to B3 format ───────────────────────────────────────
    logger.info("Step 2: Mapping well data to B3 format…")
    b3_well_df = map_wells_to_b3(valid_well_df)

    # ── Step 3: Append + dedup well CSV ─────────────────────────────────────
    logger.info("Step 3: Updating well CSV…")
    update_csv(b3_well_df, well_raw, dedup_cols=["InjectionWellId"])

    # ── Step 4: Fetch injection data (date-filtered) ─────────────────────────
    logger.info("Step 4: Fetching injection data (last %d days)…", args.days)
    raw_inj_df = fetch_injection(client, args.days)
    if raw_inj_df is None or raw_inj_df.empty:
        logger.error("No injection data returned. Exiting.")
        sys.exit(1)

    # ── Step 5: Map injection to B3 format ───────────────────────────────────
    logger.info("Step 5: Mapping injection data to B3 format…")
    b3_inj_df = map_injection_to_b3(raw_inj_df)

    # ── Step 6: Append + dedup injection CSV ────────────────────────────────
    logger.info("Step 6: Updating injection CSV…")
    update_csv(b3_inj_df, inj_raw, dedup_cols=["InjectionWellId", "Date"])

    # ── Step 7: Write intermediate B3 CSVs for injectionV3 ──────────────────
    # Re-read the full (merged) raw CSVs to feed the GIST pipeline so that
    # the pipeline always operates on the complete accumulated history.
    logger.info("Step 7: Writing B3 intermediate files for GIST pipeline…")
    full_well_df = pd.read_csv(well_raw, low_memory=False)
    full_inj_df = pd.read_csv(inj_raw, low_memory=False)
    full_well_df.to_csv(well_b3, index=False)
    full_inj_df.to_csv(inj_b3, index=False)
    logger.info("B3 well file: %d rows → %s", len(full_well_df), well_b3)
    logger.info("B3 injection file: %d rows → %s", len(full_inj_df), inj_b3)

    # ── Step 8: Run GIST pipeline (Shallow + Deep) ───────────────────────────
    verbose = 1 if args.debug else 0
    now = datetime.now()
    # Extend end date by 7 days to cover the upcoming week (matches injection_updater_V4 behaviour)
    gist_end_date = now + timedelta(days=7)
    end_date_str = gist_end_date.strftime("%m-%d-%Y")

    logger.info("Step 8: Running GIST pipeline (end date: %s)…", end_date_str)

    run_gist_pipeline(
        b3_well_file=well_b3,
        b3_inj_file=inj_b3,
        well_file=shallow_well,
        inj_file=shallow_inj,
        depth_label="Shallow",
        depth_cutoff=DEPTH_CUTOFF_FT,
        end_date_str=end_date_str,
        verbose=verbose,
    )

    run_gist_pipeline(
        b3_well_file=well_b3,
        b3_inj_file=inj_b3,
        well_file=deep_well,
        inj_file=deep_inj,
        depth_label="Deep",
        depth_cutoff=DEPTH_CUTOFF_FT,
        end_date_str=end_date_str,
        verbose=verbose,
    )

    # ── Step 9: Cleanup intermediate B3 files ───────────────────────────────
    if not args.debug:
        logger.info("Step 9: Cleaning up intermediate B3 files…")
        for temp_file in [well_b3, inj_b3]:
            if temp_file.exists():
                temp_file.unlink()
                logger.debug("Deleted: %s", temp_file)

    logger.info("get_rrc_well_injection_data completed successfully.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
