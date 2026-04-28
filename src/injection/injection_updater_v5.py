"""
injection_updater_v5.py
Self-contained updater that fetches injection data from both the TexNet API
and the RRC Texas Open Data Portal (Socrata), standardizes both to B3 format,
runs the GIST regularization pipeline for each source, and merges the results
into the final shallow/deep well and injection CSVs consumed by GIST.

Key differences from V4 + get_rrc_well_injection_data.py:
  - Single file: TexNet fetch, RRC fetch, GIST pipeline, and merge in one place.
  - RRC monthly injection volumes are divided by days-in-month → bbl/day before
    feeding injectionV3, which expects a daily rate in InjectedLiquidBBL.
  - All TexNet raw/intermediate filenames are prefixed with 'texnet_'.
  - The --dev flag appends '_dev' to every output filename.

Data sources:
  TexNet wells:      https://injection.texnet.beg.utexas.edu/api/well/wellswithinjectioncsv
  TexNet injection:  https://injection.texnet.beg.utexas.edu/api/Export
  RRC wells:         https://data.texas.gov/resource/givw-z9t4  (RRC-UIC well location)
  RRC injection:     https://data.texas.gov/resource/qq2j-f2zm  (RRC-UIC H10 Injection Monitoring)

Usage:
    python injection_updater_v5.py --target-dir ./src/data --days 90
    python injection_updater_v5.py --target-dir ./src/data --days 90 --dev
    python injection_updater_v5.py --target-dir ./src/data --days 90 --dev --debug
"""

import argparse
import concurrent.futures
import logging
import shutil
import sys
from datetime import datetime, timedelta
from io import StringIO
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests
import urllib3
from sodapy import Socrata

import credentials
import injectionV3 as inj3
from injection_id_utils import normalize_uic_string, apply_uic_normalization
from permian_subbasin import print_permian_basins_for_wells

# Suppress SSL warnings (TexNet API uses a self-signed certificate)
requests.packages.urllib3.disable_warnings(
    requests.packages.urllib3.exceptions.InsecureRequestWarning
)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

TEXNET_AUTH_URL = "https://injection.texnet.beg.utexas.edu/api/Users/Authenticate"
TEXNET_WELL_URL = "https://injection.texnet.beg.utexas.edu/api/well/wellswithinjectioncsv"
TEXNET_INJ_URL  = "https://injection.texnet.beg.utexas.edu/api/Export"

SOCRATA_DOMAIN      = "data.texas.gov"
RRC_WELLS_DATASET   = "givw-z9t4"
RRC_INJ_DATASET     = "qq2j-f2zm"
SOCRATA_PAGE_SIZE   = 10_000

# Feet threshold separating Shallow from Deep wells
DEPTH_CUTOFF_FT = 7000.0

# Cap detailed debug samples so bad upstream data does not flood the log.
DEBUG_PARSE_SAMPLE_LIMIT = 12

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

def setup_logging(debug: bool) -> None:
    """
    Configure root logger:
      - RotatingFileHandler → injection_updater_v5.log (30 MB × 3 backups)
      - StreamHandler → stdout
    Debug flag lowers level to DEBUG; default is INFO.
    """
    log_level = logging.DEBUG if debug else logging.INFO
    log_file = Path(__file__).parent / "injection_updater_v5.log"
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    # File handler: Always log at least INFO, or DEBUG if requested
    file_handler = RotatingFileHandler(
        log_file, maxBytes=30 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    file_handler.setLevel(log_level)

    # Console handler: Show essential info (INFO level)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    console_handler.setLevel(logging.INFO)

    root = logging.getLogger()
    root.setLevel(log_level)
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    # Reduce noise from 3rd party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("sodapy").setLevel(logging.WARNING)


# ──────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ──────────────────────────────────────────────────────────────────────────────

def resolve_path(target_dir: Path, basename: str, dev: bool) -> Path:
    """
    Build an output file path, optionally inserting a '_dev' suffix.

    Example:
        resolve_path(Path('./src/data'), 'texnet_disposal_well.csv', dev=True)
        -> Path('./src/data/texnet_disposal_well_dev.csv')
    """
    if dev:
        stem, ext = basename.rsplit(".", 1)
        basename = f"{stem}_dev.{ext}"
    return target_dir / basename


def backup_files(files: List[Path], backup_dir: Path) -> None:
    """Copy existing files to backup_dir, creating it if necessary."""
    backup_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        if f.exists():
            dest = backup_dir / f.name
            shutil.copy2(f, dest)
            logger.info("Backed up %s -> %s", f.name, dest)
        else:
            logger.debug("Skipping backup for missing file: %s", f)


def _coerce_to_series(values, index=None) -> pd.Series:
    """Return values as a Series while preserving the caller's index when possible."""
    if isinstance(values, pd.Series):
        return values.copy()
    if index is None:
        return pd.Series(values)
    return pd.Series(values, index=index)


def _build_record_labels(df: pd.DataFrame, candidate_cols: List[str]) -> pd.Series:
    """
    Build a compact per-row label so debug logs can point back to the source row.

    The first non-empty candidate column wins; if none are present, a row index
    label is used instead.
    """
    labels = pd.Series("", index=df.index, dtype="object")
    for col in candidate_cols:
        if col not in df.columns:
            continue
        values = df[col].fillna("").astype(str).str.strip()
        valid = values.ne("") & values.str.lower().ne("nan")
        fill_mask = labels.eq("") & valid
        labels.loc[fill_mask] = f"{col}=" + values.loc[fill_mask]
    fallback = pd.Series([f"row#{idx}" for idx in df.index], index=df.index, dtype="object")
    return labels.mask(labels.eq(""), fallback)


def _format_debug_value(value) -> str:
    """Render a compact debug-safe representation of a raw value."""
    if pd.isna(value):
        return "<NA>"
    text = str(value).strip()
    if text == "":
        return '""'
    return repr(text)


def _log_masked_samples(
    *,
    context: str,
    field_name: str,
    row_labels: Optional[pd.Series],
    raw_series: pd.Series,
    mask: pd.Series,
    issue_label: str,
    default_value=None,
) -> None:
    """
    Emit a capped DEBUG log with sample rows for a parsing/defaulting issue.
    """
    if not logger.isEnabledFor(logging.DEBUG):
        return

    mask = _coerce_to_series(mask, index=raw_series.index).fillna(False).astype(bool)
    count = int(mask.sum())
    if count == 0:
        return
    total = len(mask)
    pct = (100.0 * count / total) if total else 0.0

    labels = (
        _coerce_to_series(row_labels, index=raw_series.index)
        if row_labels is not None
        else pd.Series([f"row#{idx}" for idx in raw_series.index], index=raw_series.index, dtype="object")
    )
    samples_df = pd.DataFrame(
        {
            "label": labels,
            "raw": raw_series.astype("object"),
        },
        index=raw_series.index,
    )[mask].head(DEBUG_PARSE_SAMPLE_LIMIT)

    sample_text = "; ".join(
        f"{row.label} raw={_format_debug_value(row.raw)}"
        for row in samples_df.itertuples()
    )
    default_suffix = f" -> defaulted to {default_value!r}" if default_value is not None else ""
    logger.debug(
        "%s: field '%s' had %d/%d %s value(s) (%.1f%%)%s. Sample rows: %s",
        context,
        field_name,
        count,
        total,
        issue_label,
        pct,
        default_suffix,
        sample_text,
    )


def _log_missing_required_columns(df: pd.DataFrame, required_cols: List[str], context: str) -> None:
    """
    Raise a clear error when a required upstream column is absent.
    """
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        available = ", ".join(sorted(df.columns.tolist())[:25])
        raise ValueError(
            f"{context} missing required columns: {missing}. "
            f"Available columns sample: {available}"
        )


def _to_numeric_safe(
    series: pd.Series,
    default=0,
    *,
    field_name: str = "unknown",
    context: str = "dataset",
    row_labels: Optional[pd.Series] = None,
) -> pd.Series:
    """
    Coerce a series to numeric, logging invalid or missing values in DEBUG mode.
    """
    raw = _coerce_to_series(series)
    stripped = raw.astype("object").astype(str).str.strip()
    missing_mask = raw.isna() | stripped.eq("") | stripped.str.lower().eq("nan")
    parsed = pd.to_numeric(raw, errors="coerce")
    invalid_mask = parsed.isna() & ~missing_mask

    _log_masked_samples(
        context=context,
        field_name=field_name,
        row_labels=row_labels,
        raw_series=raw,
        mask=invalid_mask,
        issue_label="invalid numeric",
        default_value=default,
    )
    _log_masked_samples(
        context=context,
        field_name=field_name,
        row_labels=row_labels,
        raw_series=raw,
        mask=missing_mask,
        issue_label="missing/blank",
        default_value=default,
    )
    return parsed.fillna(default)


def _parse_date_to_b3(
    series: pd.Series,
    *,
    field_name: str = "unknown",
    context: str = "dataset",
    row_labels: Optional[pd.Series] = None,
    default_date: str = "01-01-1970",
) -> pd.Series:
    """
    Parse date-like values to 'MM-DD-YYYY', logging rows that defaulted.
    """
    raw = _coerce_to_series(series)
    stripped = raw.astype("object").astype(str).str.strip()
    missing_mask = raw.isna() | stripped.eq("") | stripped.str.lower().eq("nan")
    parsed = pd.to_datetime(raw, errors="coerce")
    invalid_mask = parsed.isna() & ~missing_mask

    _log_masked_samples(
        context=context,
        field_name=field_name,
        row_labels=row_labels,
        raw_series=raw,
        mask=invalid_mask,
        issue_label="invalid date",
        default_value=default_date,
    )
    _log_masked_samples(
        context=context,
        field_name=field_name,
        row_labels=row_labels,
        raw_series=raw,
        mask=missing_mask,
        issue_label="missing/blank",
        default_value=default_date,
    )

    formatted = parsed.dt.strftime("%m-%d-%Y")
    return formatted.fillna(default_date)


def _normalize_uic_series(
    series: pd.Series,
    *,
    field_name: str,
    context: str,
    row_labels: Optional[pd.Series] = None,
) -> pd.Series:
    """
    Normalize UIC-like identifiers and log rows that collapse to blank IDs.
    """
    raw = _coerce_to_series(series)
    normalized = raw.apply(normalize_uic_string)
    blank_mask = normalized.eq("")
    _log_masked_samples(
        context=context,
        field_name=field_name,
        row_labels=row_labels,
        raw_series=raw,
        mask=blank_mask,
        issue_label="missing/blank identifier",
    )
    return normalized


def _filter_critical_columns(
    df: pd.DataFrame,
    columns: List[str],
    context: str,
    row_labels: pd.Series,
    is_date: bool = False,
) -> pd.DataFrame:
    """
    Filter out rows with missing or invalid values in critical columns.
    Returns the filtered DataFrame.
    """
    drop_mask = pd.Series(False, index=df.index)
    for col in columns:
        if col not in df.columns:
            continue
        raw = df[col]
        if is_date:
            parsed = pd.to_datetime(raw, errors="coerce")
            bad = parsed.isna()
        else:
            parsed = pd.to_numeric(raw, errors="coerce")
            bad = parsed.isna() | (parsed == 0)

        _log_masked_samples(
            context=context,
            field_name=col,
            row_labels=row_labels,
            raw_series=raw,
            mask=bad,
            issue_label="excluded (missing/invalid)",
        )
        drop_mask |= bad

    before = len(df)
    df = df[~drop_mask].copy()
    if before > len(df):
        logger.info(
            "%s: excluded %d/%d rows (%.1f%%) due to missing/invalid critical fields in %s.",
            context,
            before - len(df),
            before,
            100.0 * (before - len(df)) / before,
            columns,
        )
    return df


def update_csv(new_df: pd.DataFrame, path: Path, dedup_cols: List[str]) -> pd.DataFrame:
    """
    Merge new_df with the existing CSV at path (if present), deduplicate on
    dedup_cols keeping the newest value, save, and return the combined DataFrame.
    """
    new_df = apply_uic_normalization(new_df.copy(), dedup_cols)

    if path.exists():
        dtypes = {col: str for col in dedup_cols}
        existing_df = pd.read_csv(path, low_memory=False, dtype=dtypes)
        existing_df = apply_uic_normalization(existing_df, dedup_cols)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        logger.info("No existing file at %s — creating new.", path)
        combined = new_df.copy()

    before = len(combined)
    combined.drop_duplicates(subset=dedup_cols, keep="last", inplace=True)
    combined.to_csv(path, index=False)
    logger.info(
        "CSV updated: %d rows (removed %d duplicates) -> %s",
        len(combined), before - len(combined), path,
    )
    return combined


def reformat_well_id_column(path: Path) -> None:
    """Rename 'InjectionWellId' → 'ID' in a GIST well file (in place)."""
    df = pd.read_csv(path, dtype={"InjectionWellId": str}, low_memory=False)
    df["InjectionWellId"] = _normalize_uic_series(
        df["InjectionWellId"],
        field_name="InjectionWellId",
        context=f"GIST well file {path.name}",
    )
    df.rename(columns={"InjectionWellId": "ID"}, inplace=True)
    df.to_csv(path, index=False)


# ──────────────────────────────────────────────────────────────────────────────
# TexNet — API helpers
# ──────────────────────────────────────────────────────────────────────────────

def texnet_authenticate(username: str, password: str) -> Optional[str]:
    """
    Authenticate with the TexNet API and return a bearer token.
    Returns None and logs an error on failure.
    """
    try:
        response = requests.post(
            TEXNET_AUTH_URL,
            json={"username": username, "password": password},
            verify=False,
        )
        response.raise_for_status()
        token = response.json()["Token"]
        logger.info("TexNet authentication successful.")
        return token
    except requests.exceptions.RequestException as e:
        logger.error("TexNet authentication failed: %s", e)
        return None


def texnet_fetch(api_url: str, token: str, method: str = "GET",
                 json_payload=None, params=None) -> Optional[str]:
    """
    Fetch CSV text from a TexNet API endpoint.
    Returns raw response text or None on failure.
    """
    try:
        headers = {"Authorization": f"Bearer {token}"}
        if method.upper() == "POST" and json_payload is not None:
            headers["Content-Type"] = "application/json"

        if method.upper() == "GET":
            resp = requests.get(api_url, headers=headers, params=params, verify=False)
        else:
            resp = requests.post(api_url, headers=headers, json=json_payload,
                                 params=params, verify=False)
        resp.raise_for_status()
        logger.debug("TexNet: fetched %d bytes from %s", len(resp.content), api_url)
        return resp.text
    except requests.exceptions.RequestException as e:
        logger.error("TexNet API request to %s failed: %s", api_url, e)
        return None


# ──────────────────────────────────────────────────────────────────────────────
# TexNet — raw CSV update helpers
# ──────────────────────────────────────────────────────────────────────────────

def texnet_update_well_csv(api_text: str, path: Path) -> pd.DataFrame:
    """
    Merge freshly-fetched TexNet well data with the existing CSV.
    Deduplicates on 'Uicnumber'; the API value wins. Returns the DataFrame.
    """
    new_df = pd.read_csv(
        StringIO(api_text),
        dtype={"Uicnumber": str, "Apinumber": str, "Id": int},
        low_memory=False,
    )

    if path.exists():
        existing_df = pd.read_csv(
            path,
            dtype={"Uicnumber": str, "Apinumber": str, "Id": int},
            low_memory=False,
        )
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        logger.info("No existing TexNet well file at %s — creating new.", path)
        combined = new_df

    combined = apply_uic_normalization(combined, ["Uicnumber"])
    combined.drop_duplicates(subset=["Uicnumber"], keep="last", inplace=True)
    combined.to_csv(path, index=False)
    logger.info("TexNet well CSV updated: %d wells -> %s", len(combined), path)
    return combined


def texnet_update_inj_csv(api_text: str, path: Path) -> None:
    """
    Merge freshly-fetched TexNet injection data with the existing CSV.
    Deduplicates on ('Uicnumber', 'Date of Injection'); API value wins.
    """
    new_df = pd.read_csv(StringIO(api_text), dtype={"UIC Number": str, "Id": int},
                         low_memory=False)
    if "UIC Number" in new_df.columns:
        new_df.rename(columns={"UIC Number": "Uicnumber"}, inplace=True)

    if path.exists():
        existing_df = pd.read_csv(path, dtype={"Uicnumber": str, "Id": int},
                                  low_memory=False)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        logger.info("No existing TexNet injection file at %s — creating new.", path)
        combined = new_df

    before = len(combined)
    combined = apply_uic_normalization(combined, ["Uicnumber"])
    combined.drop_duplicates(subset=["Uicnumber", "Date of Injection"], keep="last",
                             inplace=True)
    logger.info(
        "TexNet injection CSV updated: %d rows (removed %d duplicates) -> %s",
        len(combined), before - len(combined), path,
    )
    combined.to_csv(path, index=False)


# ──────────────────────────────────────────────────────────────────────────────
# TexNet — B3 format transformations
# ──────────────────────────────────────────────────────────────────────────────

# Column renames: raw TexNet well → B3
_TEXNET_WELL_B3_MAP = {
    "Uicnumber":               "InjectionWellId",
    "Apinumber":               "APINumber",
    "SurfaceLatitude":         "SurfaceHoleLatitude",
    "SurfaceLongitude":        "SurfaceHoleLongitude",
    "OriginalPermitDate":      "WellActivatedDate",
    "TotalBpdmax":             "PermittedMaxLiquidBPD",
    "InjectionBottomInterval": "PermittedIntervalBottomFt",
    "InjectionTopInterval":    "PermittedIntervalTopFt",
    "WellClassification":      "CompletedWellDepthClassification",
}

# Column renames: raw TexNet injection → B3
_TEXNET_INJ_B3_MAP = {
    "Uicnumber":             "InjectionWellId",
    "Date of Injection":     "Date",
    "Volume Injected (BBLs)": "InjectedLiquidBBL",
}


def texnet_well_to_b3(input_path: Path, output_path: Path) -> None:
    """
    Transform raw TexNet well CSV to B3 format.
    Combines LeaseName + WellNumber into WellName, then renames columns.
    """
    df = pd.read_csv(
        input_path,
        dtype={"Uicnumber": str, "Apinumber": str, "Id": int},
        low_memory=False,
    )
    _log_missing_required_columns(
        df,
        [
            "Uicnumber",
            "Apinumber",
            "SurfaceLatitude",
            "SurfaceLongitude",
            "OriginalPermitDate",
            "TotalBpdmax",
            "InjectionBottomInterval",
            "InjectionTopInterval",
            "WellClassification",
            "LeaseName",
            "WellNumber",
        ],
        "TexNet well CSV",
    )
    row_labels = _build_record_labels(df, ["Uicnumber", "Apinumber", "Id"])

    # Exclude wells with missing coordinates or depth intervals
    df = _filter_critical_columns(
        df,
        ["SurfaceLatitude", "SurfaceLongitude", "InjectionTopInterval", "InjectionBottomInterval"],
        "TexNet wells",
        row_labels,
    )
    row_labels = row_labels.loc[df.index]

    df["WellName"] = df["LeaseName"].astype(str) + " " + df["WellNumber"].astype(str)
    df.drop(columns=["LeaseName", "WellNumber"], errors="ignore", inplace=True)
    df.rename(columns=_TEXNET_WELL_B3_MAP, inplace=True)
    df["InjectionWellId"] = _normalize_uic_series(
        df["InjectionWellId"],
        field_name="Uicnumber",
        context="TexNet wells",
        row_labels=row_labels,
    )
    df["UICNumber"] = df["InjectionWellId"]
    df["SurfaceHoleLatitude"] = _to_numeric_safe(
        df["SurfaceHoleLatitude"],
        default=0.0,
        field_name="SurfaceLatitude",
        context="TexNet wells",
        row_labels=row_labels,
    )
    df["SurfaceHoleLongitude"] = _to_numeric_safe(
        df["SurfaceHoleLongitude"],
        default=0.0,
        field_name="SurfaceLongitude",
        context="TexNet wells",
        row_labels=row_labels,
    )
    df["WellActivatedDate"] = _parse_date_to_b3(
        df["WellActivatedDate"],
        field_name="OriginalPermitDate",
        context="TexNet wells",
        row_labels=row_labels,
    )
    df["PermittedMaxLiquidBPD"] = _to_numeric_safe(
        df["PermittedMaxLiquidBPD"],
        default=0.0,
        field_name="TotalBpdmax",
        context="TexNet wells",
        row_labels=row_labels,
    )
    df["PermittedIntervalTopFt"] = _to_numeric_safe(
        df["PermittedIntervalTopFt"],
        default=0.0,
        field_name="InjectionTopInterval",
        context="TexNet wells",
        row_labels=row_labels,
    )
    df["PermittedIntervalBottomFt"] = _to_numeric_safe(
        df["PermittedIntervalBottomFt"],
        default=0.0,
        field_name="InjectionBottomInterval",
        context="TexNet wells",
        row_labels=row_labels,
    )

    try:
        print_permian_basins_for_wells(
            df["InjectionWellId"],
            df["SurfaceHoleLatitude"],
            df["SurfaceHoleLongitude"],
        )
    except Exception as exc:
        logger.warning("Permian sub-basin reporting skipped: %s", exc)

    df.to_csv(output_path, index=False)
    logger.info("TexNet B3 well file written: %d rows -> %s", len(df), output_path)


def texnet_inj_to_b3(input_path: Path, output_path: Path) -> None:
    """
    Transform raw TexNet injection CSV to B3 format.

    Steps:
      1. Rename columns via _TEXNET_INJ_B3_MAP.
      2. Normalize UIC well IDs.
      3. Reformat Date to MM-DD-YYYY (same as RRC path) so both sources are
         consistent in the B3 injection files.
      4. Write only the three required B3 columns to keep the file compact.

    TexNet data is already daily (one row per well per day), so no rate
    conversion is applied to InjectedLiquidBBL.
    """
    df = pd.read_csv(input_path, dtype={"Uicnumber": str, "Id": int}, low_memory=False)
    _log_missing_required_columns(
        df,
        ["Uicnumber", "Date of Injection", "Volume Injected (BBLs)"],
        "TexNet injection CSV",
    )
    row_labels = _build_record_labels(df, ["Uicnumber", "Id"])

    # Exclude records with missing volume
    df = _filter_critical_columns(
        df,
        ["Volume Injected (BBLs)"],
        "TexNet injection",
        row_labels,
    )
    row_labels = row_labels.loc[df.index]

    df.rename(columns=_TEXNET_INJ_B3_MAP, inplace=True)
    df["InjectionWellId"] = _normalize_uic_series(
        df["InjectionWellId"],
        field_name="Uicnumber",
        context="TexNet injection",
        row_labels=row_labels,
    )

    # Standardize to MM-DD-YYYY so TexNet and RRC injection files share the
    # same date format (TexNet raw dates arrive as YYYY-MM-DD ISO strings).
    df["Date"] = _parse_date_to_b3(
        df["Date"],
        field_name="Date of Injection",
        context="TexNet injection",
        row_labels=row_labels,
    )
    df["InjectedLiquidBBL"] = _to_numeric_safe(
        df["InjectedLiquidBBL"],
        default=0.0,
        field_name="Volume Injected (BBLs)",
        context="TexNet injection",
        row_labels=row_labels,
    )

    # Keep only the three B3 columns; the raw file carries 50+ extra columns
    # (pressure, operator, lat/lon, etc.) that are never used downstream.
    df[["InjectionWellId", "Date", "InjectedLiquidBBL"]].to_csv(output_path, index=False)
    logger.info("TexNet B3 injection file written: %d rows -> %s", len(df), output_path)


# ──────────────────────────────────────────────────────────────────────────────
# RRC — Socrata client and data fetch
# ──────────────────────────────────────────────────────────────────────────────

def build_socrata_client(username: str, password: str, app_token: str) -> Socrata:
    """Create an authenticated Socrata client for data.texas.gov."""
    client = Socrata(
        SOCRATA_DOMAIN,
        app_token,
        username=username,
        password=password,
        timeout=200,
    )
    logger.info("Socrata client created for domain: %s", SOCRATA_DOMAIN)
    return client


def rrc_fetch_wells(client: Socrata) -> Optional[pd.DataFrame]:
    """
    Fetch all RRC UIC well location records (givw-z9t4) where
    uic_type_injection is 1 or 2. Returns a DataFrame or None on failure.
    """
    try:
        where = "uic_type_injection = 1 OR uic_type_injection = 2"
        logger.info("Fetching RRC wells from %s where %s …", RRC_WELLS_DATASET, where)
        records = client.get_all(RRC_WELLS_DATASET, where=where, limit=SOCRATA_PAGE_SIZE)
        df = pd.DataFrame.from_records(records)
        logger.info("Fetched %d RRC well records.", len(df))
        return df
    except Exception as e:
        logger.error("Failed to fetch RRC wells: %s", e)
        return None


def rrc_fetch_injection(client: Socrata, days: int) -> Optional[pd.DataFrame]:
    """
    Fetch RRC H10 injection monitoring records (qq2j-f2zm) for the last
    `days` days via a $where date filter. Returns a DataFrame or None.
    """
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S")
    where = f"formatted_date >= '{cutoff}'"
    try:
        logger.info(
            "Fetching RRC injection from %s where %s …",
            RRC_INJ_DATASET, where,
        )
        records = client.get_all(RRC_INJ_DATASET, where=where, limit=SOCRATA_PAGE_SIZE)
        df = pd.DataFrame.from_records(records)
        logger.info("Fetched %d RRC injection records.", len(df))
        return df
    except Exception as e:
        logger.error("Failed to fetch RRC injection data: %s", e)
        return None


# ──────────────────────────────────────────────────────────────────────────────
# RRC — B3 format transformations
# ──────────────────────────────────────────────────────────────────────────────

def rrc_map_wells_to_b3(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map raw RRC well DataFrame (givw-z9t4 SODA columns) to the B3 schema
    expected by injectionV3.injTX.

    WellName = LEASE_NAME + ' ' + WELL_NO_DISPLAY (concatenated so the
    displayed name is unique, since LEASE_NAME alone is not).

    Source columns  → B3 columns
    uic_number      → InjectionWellId, UICNumber
    api_no          → APINumber
    lease_name +
    well_no_display → WellName
    latitude_nad83  → SurfaceHoleLatitude
    longitude_nad83 → SurfaceHoleLongitude
    h1_date         → WellActivatedDate
    top_inj_zone    → PermittedIntervalTopFt
    bot_inj_zone    → PermittedIntervalBottomFt
    (derived)       → CompletedWellDepthClassification
    """
    expected = [
        "uic_number", "api_no", "lease_name", "well_no_display",
        "latitude_nad83", "longitude_nad83", "h1_date",
        "top_inj_zone", "bot_inj_zone",
    ]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        logger.warning("RRC wells DataFrame missing expected columns: %s", missing)

    row_labels = _build_record_labels(df, ["uic_number", "api_no", "lease_name"])

    # Exclude wells with missing coordinates
    df = _filter_critical_columns(
        df,
        ["latitude_nad83", "longitude_nad83"],
        "RRC wells",
        row_labels,
    )
    row_labels = row_labels.loc[df.index]

    # Exclude wells with missing h1_date
    df = _filter_critical_columns(
        df,
        ["h1_date"],
        "RRC wells",
        row_labels,
        is_date=True,
    )
    row_labels = row_labels.loc[df.index]

    out = pd.DataFrame()
    out["InjectionWellId"] = _normalize_uic_series(
        df.get("uic_number", pd.Series(dtype=str)).astype(str),
        field_name="uic_number",
        context="RRC wells",
        row_labels=row_labels,
    )
    out["UICNumber"] = out["InjectionWellId"]
    out["APINumber"] = (
        df.get("api_no", pd.Series(dtype=str)).astype(str).str.strip()
    )

    # Unique display name: lease + well number
    lease   = df.get("lease_name",      pd.Series([""] * len(df))).fillna("").astype(str)
    well_no = df.get("well_no_display",  pd.Series([""] * len(df))).fillna("").astype(str)
    out["WellName"] = (lease + " " + well_no).str.strip()

    out["SurfaceHoleLatitude"] = _to_numeric_safe(
        df.get("latitude_nad83", pd.Series(dtype=float)),
        default=0.0,
        field_name="latitude_nad83",
        context="RRC wells",
        row_labels=row_labels,
    )
    out["SurfaceHoleLongitude"] = _to_numeric_safe(
        df.get("longitude_nad83", pd.Series(dtype=float)),
        default=0.0,
        field_name="longitude_nad83",
        context="RRC wells",
        row_labels=row_labels,
    )
    out["WellActivatedDate"] = _parse_date_to_b3(
        df.get("h1_date", pd.Series(dtype=str)),
        field_name="h1_date",
        context="RRC wells",
        row_labels=row_labels,
    )
    # RRC dataset does not include a permitted max injection rate column.
    # TODO: If the RRC open-data portal ever exposes a max-rate field, map it here.
    out["PermittedMaxLiquidBPD"] = 0.0
    out["PermittedIntervalTopFt"] = _to_numeric_safe(
        df.get("top_inj_zone", pd.Series(dtype=float)),
        default=0.0,
        field_name="top_inj_zone",
        context="RRC wells",
        row_labels=row_labels,
    ).astype(int)
    out["PermittedIntervalBottomFt"] = _to_numeric_safe(
        df.get("bot_inj_zone", pd.Series(dtype=float)),
        default=0.0,
        field_name="bot_inj_zone",
        context="RRC wells",
        row_labels=row_labels,
    ).astype(int)
    out["CompletedWellDepthClassification"] = out["PermittedIntervalBottomFt"].apply(
        lambda d: "Deep" if d >= DEPTH_CUTOFF_FT else "Shallow"
    )

    try:
        print_permian_basins_for_wells(
            out["InjectionWellId"],
            out["SurfaceHoleLatitude"],
            out["SurfaceHoleLongitude"],
        )
    except Exception as exc:
        logger.warning("Permian sub-basin reporting skipped: %s", exc)

    logger.info("Mapped %d RRC wells to B3 format.", len(out))
    return out


def rrc_map_injection_to_b3(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map raw RRC injection DataFrame (qq2j-f2zm SODA columns) to B3 format.

    The RRC H10 dataset reports monthly injection totals.  injectionV3
    expects InjectedLiquidBBL as a daily rate (bbl/day), so each monthly
    volume is divided by the number of days in that month.

    Days-in-month is derived vectorially via pandas datetime accessors
    (dt.days_in_month), which is faster than row-wise apply() on large frames.

    Source columns  → B3 columns
    uic_no          → InjectionWellId
    formatted_date  → Date  (MM-DD-YYYY)
    vol_liq / days  → InjectedLiquidBBL  (bbl/day)
    """
    expected = ["uic_no", "formatted_date", "vol_liq"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        logger.warning("RRC injection DataFrame missing expected columns: %s", missing)

    row_labels = _build_record_labels(df, ["uic_no", "well_no", "api_no"])
    out = pd.DataFrame()
    out["InjectionWellId"] = _normalize_uic_series(
        df.get("uic_no", pd.Series(dtype=str)).astype(str),
        field_name="uic_no",
        context="RRC injection",
        row_labels=row_labels,
    )
    out["Date"] = _parse_date_to_b3(
        df.get("formatted_date", pd.Series(dtype=str)),
        field_name="formatted_date",
        context="RRC injection",
        row_labels=row_labels,
    )

    monthly_vol = _to_numeric_safe(
        df.get("vol_liq", pd.Series(dtype=float)),
        default=0.0,
        field_name="vol_liq",
        context="RRC injection",
        row_labels=row_labels,
    )

    # Vectorized: parse the B3 date strings we just produced, then use
    # dt.days_in_month.  Unparseable originals became "01-01-1970" (January,
    # 31 days) which is a safe divisor.  fillna(31) guards any remaining NaT.
    parsed_dates = pd.to_datetime(out["Date"], format="%m-%d-%Y", errors="coerce")
    days_per_month = parsed_dates.dt.days_in_month.fillna(31).astype(int)
    out["InjectedLiquidBBL"] = monthly_vol / days_per_month

    n_epoch = (out["Date"] == "01-01-1970").sum()
    if n_epoch > 0:
        logger.debug(
            "RRC injection: %d/%d records had unparseable dates → defaulted to 01-01-1970",
            n_epoch, len(out),
        )

    logger.info("Mapped %d RRC injection records to B3 format (monthly→daily).", len(out))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Shared GIST pipeline
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
    Run the full injTX → addDaily → inj → processRates → outputReg pipeline
    for one depth class ('Shallow' or 'Deep').

    Parameters
    ----------
    b3_well_file : B3-format well CSV
    b3_inj_file  : B3-format injection CSV
    well_file    : output path for GIST well CSV
    inj_file     : output path for GIST injection CSV
    depth_label  : 'Shallow' or 'Deep'
    depth_cutoff : depth threshold in feet (7000.0)
    end_date_str : end date 'MM-DD-YYYY' passed to processRates
    verbose      : 0 = silent, 1 = info, 2 = debug (injectionV3 internal prints)
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
# Merge TexNet + RRC GIST outputs into final files
# ──────────────────────────────────────────────────────────────────────────────

def merge_datasets(
    texnet_file: Path,
    rrc_file: Path,
    output_file: Path,
    dedup_cols: List[str],
) -> None:
    """
    Concatenate the TexNet GIST output and the RRC GIST output, normalize
    the 'ID' column, deduplicate (TexNet wins on conflict), sort, and write
    the merged result to output_file.
    """
    dfs = []
    for label, path in [("TexNet", texnet_file), ("RRC", rrc_file)]:
        if path.exists():
            df = pd.read_csv(path, dtype={"ID": str}, low_memory=False)
            dfs.append(df)
            logger.info("Loaded %d rows from %s (%s)", len(df), path.name, label)
        else:
            logger.warning("%s file not found, skipping: %s", label, path)

    if not dfs:
        logger.error("No data to merge for %s — skipping.", output_file.name)
        return

    combined = pd.concat(dfs, ignore_index=True)
    before = len(combined)

    if "ID" in combined.columns:
        combined = apply_uic_normalization(combined, ["ID"])

    # TexNet is appended last, so keep="last" gives TexNet priority on conflicts
    combined.drop_duplicates(subset=dedup_cols, keep="last", inplace=True)

    sort_cols = [c for c in ["ID", "Days", "Date"] if c in combined.columns]
    if sort_cols:
        combined.sort_values(by=sort_cols, inplace=True)

    combined.to_csv(output_file, index=False)
    logger.info(
        "Merged %s: %d rows (removed %d duplicates) -> %s",
        output_file.name, len(combined), before - len(combined), output_file,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Fetch TexNet and RRC injection data, standardize to B3 format, "
            "run the GIST pipeline, and merge into final shallow/deep CSVs."
        )
    )
    parser.add_argument(
        "--target-dir",
        required=True,
        type=Path,
        help="Directory where all CSV output files will be written.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=1826,
        help="Number of days to look back when fetching RRC injection data (default: 1826 ≈ 5 years).",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        default=False,
        help="Append '_dev' suffix to all output filenames (prevents overwriting production data).",
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
        default=None,
        help="Copy existing CSVs here before updating (optional).",
    )
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Section runners (called in parallel from main)
# ──────────────────────────────────────────────────────────────────────────────

def run_texnet_section(
    tdir: Path,
    dev: bool,
    debug: bool,
    start: datetime,
    now: datetime,
    end_date_str: str,
    verbose: int,
) -> None:
    """
    Fetch TexNet wells + injection, transform to B3 format, and run the GIST
    pipeline (Shallow + Deep). Writes all texnet_* and texnet_gist_* CSVs.
    Raises RuntimeError on any unrecoverable failure so the caller can react.
    """
    logger.info("-" * 40)
    logger.info("Section 1 (TexNet): starting")

    rp = lambda name: resolve_path(tdir, name, dev)  # noqa: E731
    tx_well_raw     = rp("texnet_disposal_well.csv")
    tx_inj_raw      = rp("texnet_disposal_inj.csv")
    tx_well_b3      = rp("texnet_disposal_well_b3_format.csv")
    tx_inj_b3       = rp("texnet_disposal_inj_b3_format.csv")
    tx_shallow_well = rp("texnet_gist_well_shallow.csv")
    tx_shallow_inj  = rp("texnet_gist_injection_shallow.csv")
    tx_deep_well    = rp("texnet_gist_well_deep.csv")
    tx_deep_inj     = rp("texnet_gist_injection_deep.csv")

    # Step 1: Authenticate
    token = texnet_authenticate(credentials.USERNAME, credentials.PASSWORD)
    if not token:
        raise RuntimeError("TexNet authentication failed.")

    # Step 2: Fetch + update well list
    logger.info("TexNet Step 1: Fetching well list…")
    well_text = texnet_fetch(TEXNET_WELL_URL, token)
    if not well_text:
        raise RuntimeError("Failed to fetch TexNet well data.")

    raw_well_df = pd.read_csv(
        StringIO(well_text),
        dtype={"Uicnumber": str, "Apinumber": str, "Id": int},
        low_memory=False,
    )
    _log_missing_required_columns(
        raw_well_df,
        ["Uicnumber", "Apinumber", "Id", "SurfaceLatitude", "SurfaceLongitude"],
        "TexNet wells API response",
    )
    texnet_labels = _build_record_labels(raw_well_df, ["Uicnumber", "Apinumber", "Id"])
    texnet_lat = _to_numeric_safe(
        raw_well_df["SurfaceLatitude"],
        default=0.0,
        field_name="SurfaceLatitude",
        context="TexNet wells API response",
        row_labels=texnet_labels,
    )
    texnet_lon = _to_numeric_safe(
        raw_well_df["SurfaceLongitude"],
        default=0.0,
        field_name="SurfaceLongitude",
        context="TexNet wells API response",
        row_labels=texnet_labels,
    )
    invalid_coord_mask = texnet_lat.eq(0) | texnet_lon.eq(0)
    _log_masked_samples(
        context="TexNet wells API response",
        field_name="SurfaceLatitude/SurfaceLongitude",
        row_labels=texnet_labels,
        raw_series=raw_well_df["SurfaceLatitude"].astype(str) + ", " + raw_well_df["SurfaceLongitude"].astype(str),
        mask=invalid_coord_mask,
        issue_label="rejected zero/invalid coordinate",
    )
    valid_wells = raw_well_df[
        texnet_lat.ne(0) & texnet_lon.ne(0)
    ].copy()
    valid_wells["SurfaceLatitude"] = texnet_lat.loc[valid_wells.index]
    valid_wells["SurfaceLongitude"] = texnet_lon.loc[valid_wells.index]
    logger.info(
        "TexNet well filter: %d total fetched → %d with valid coordinates",
        len(raw_well_df),
        len(valid_wells),
    )
    well_df = texnet_update_well_csv(valid_wells.to_csv(index=False), tx_well_raw)

    # Step 3: Fetch + update injection data
    logger.info("TexNet Step 2: Fetching injection data…")
    id_array = well_df["Id"].to_numpy()
    payload = {
        "BeginMonth":     start.month,
        "BeginYear":      start.year,
        "EndMonth":       (now.month % 12) + 1,
        "EndYear":        now.year if now.month < 12 else now.year + 1,
        "Format":         "excel",
        "IncludeWellIds": True,
        "WellIds":        id_array.tolist(),
    }
    inj_text = texnet_fetch(TEXNET_INJ_URL, token, method="POST", json_payload=payload)
    if not inj_text:
        raise RuntimeError("Failed to fetch TexNet injection data.")
    texnet_update_inj_csv(inj_text, tx_inj_raw)

    # Step 4: Transform to B3 format
    logger.info("TexNet Step 3: Transforming to B3 format…")
    texnet_well_to_b3(tx_well_raw, tx_well_b3)
    texnet_inj_to_b3(tx_inj_raw, tx_inj_b3)

    # Step 5: Run GIST pipeline (Shallow + Deep)
    logger.info("TexNet Step 4: Running GIST pipeline…")
    run_gist_pipeline(
        b3_well_file=tx_well_b3, b3_inj_file=tx_inj_b3,
        well_file=tx_shallow_well, inj_file=tx_shallow_inj,
        depth_label="Shallow", depth_cutoff=DEPTH_CUTOFF_FT,
        end_date_str=end_date_str, verbose=verbose,
    )
    run_gist_pipeline(
        b3_well_file=tx_well_b3, b3_inj_file=tx_inj_b3,
        well_file=tx_deep_well, inj_file=tx_deep_inj,
        depth_label="Deep", depth_cutoff=DEPTH_CUTOFF_FT,
        end_date_str=end_date_str, verbose=verbose,
    )

    # Step 6: Cleanup B3 intermediates (skip in debug mode to aid inspection)
    if not debug:
        for temp in [tx_well_b3, tx_inj_b3]:
            if temp.exists():
                temp.unlink()
                logger.debug("Deleted TexNet intermediate: %s", temp)

    logger.info("Section 1 (TexNet): complete")


def run_rrc_section(
    tdir: Path,
    dev: bool,
    days: int,
    end_date_str: str,
    verbose: int,
) -> None:
    """
    Fetch RRC wells + injection via Socrata, transform to B3 format (with
    monthly→daily conversion), and run the GIST pipeline (Shallow + Deep).
    Writes all rrc_* and rrc_gist_* CSVs.
    Raises RuntimeError on any unrecoverable failure so the caller can react.
    """
    logger.info("-" * 40)
    logger.info("Section 2 (RRC): starting")

    rp = lambda name: resolve_path(tdir, name, dev)  # noqa: E731
    rrc_well_raw     = rp("rrc_disposal_well.csv")
    rrc_inj_raw      = rp("rrc_disposal_inj.csv")
    rrc_shallow_well = rp("rrc_gist_well_shallow.csv")
    rrc_shallow_inj  = rp("rrc_gist_injection_shallow.csv")
    rrc_deep_well    = rp("rrc_gist_well_deep.csv")
    rrc_deep_inj     = rp("rrc_gist_injection_deep.csv")

    client = build_socrata_client(
        credentials.RRC_USERNAME,
        credentials.RRC_PASSWORD,
        credentials.RRC_APP_TOKEN,
    )

    # Step 1: Fetch wells → B3 → update raw well CSV
    logger.info("RRC Step 1: Fetching well locations…")
    raw_rrc_wells = rrc_fetch_wells(client)
    if raw_rrc_wells is None or raw_rrc_wells.empty:
        raise RuntimeError("No RRC well data returned.")

    rrc_labels = _build_record_labels(raw_rrc_wells, ["uic_number", "api_no", "lease_name"])
    lat = _to_numeric_safe(
        raw_rrc_wells.get("latitude_nad83"),
        default=0.0,
        field_name="latitude_nad83",
        context="RRC wells API response",
        row_labels=rrc_labels,
    )
    lon = _to_numeric_safe(
        raw_rrc_wells.get("longitude_nad83"),
        default=0.0,
        field_name="longitude_nad83",
        context="RRC wells API response",
        row_labels=rrc_labels,
    )
    invalid_rrc_coord_mask = lat.eq(0) | lon.eq(0)
    _log_masked_samples(
        context="RRC wells API response",
        field_name="latitude_nad83/longitude_nad83",
        row_labels=rrc_labels,
        raw_series=raw_rrc_wells.get("latitude_nad83", pd.Series(index=raw_rrc_wells.index, dtype="object")).astype(str)
        + ", "
        + raw_rrc_wells.get("longitude_nad83", pd.Series(index=raw_rrc_wells.index, dtype="object")).astype(str),
        mask=invalid_rrc_coord_mask,
        issue_label="rejected zero/invalid coordinate",
    )
    valid_rrc_wells = raw_rrc_wells[
        lat.ne(0) & lon.ne(0)
    ].copy()
    logger.info(
        "RRC well filter: %d total fetched → %d with valid coordinates",
        len(raw_rrc_wells),
        len(valid_rrc_wells),
    )

    logger.info("RRC Step 2: Mapping well data to B3 format…")
    b3_rrc_well_df = rrc_map_wells_to_b3(valid_rrc_wells)
    update_csv(b3_rrc_well_df, rrc_well_raw, dedup_cols=["InjectionWellId"])

    # Step 2: Fetch injection → B3 (monthly→daily) → update raw inj CSV
    logger.info("RRC Step 3: Fetching injection data (last %d days)…", days)
    raw_rrc_inj = rrc_fetch_injection(client, days)
    if raw_rrc_inj is None:
        raise RuntimeError("Failed to fetch RRC injection data.")

    if raw_rrc_inj.empty:
        logger.warning(
            "No RRC injection data for the last %d days — proceeding with existing CSV.",
            days,
        )
    else:
        logger.info("RRC Step 4: Mapping injection data to B3 format (monthly→daily)…")
        b3_rrc_inj_df = rrc_map_injection_to_b3(raw_rrc_inj)
        update_csv(b3_rrc_inj_df, rrc_inj_raw, dedup_cols=["InjectionWellId", "Date"])

    # Step 3: Run GIST pipeline (Shallow + Deep)
    logger.info("RRC Step 5: Running GIST pipeline…")
    run_gist_pipeline(
        b3_well_file=rrc_well_raw, b3_inj_file=rrc_inj_raw,
        well_file=rrc_shallow_well, inj_file=rrc_shallow_inj,
        depth_label="Shallow", depth_cutoff=DEPTH_CUTOFF_FT,
        end_date_str=end_date_str, verbose=verbose,
    )
    run_gist_pipeline(
        b3_well_file=rrc_well_raw, b3_inj_file=rrc_inj_raw,
        well_file=rrc_deep_well, inj_file=rrc_deep_inj,
        depth_label="Deep", depth_cutoff=DEPTH_CUTOFF_FT,
        end_date_str=end_date_str, verbose=verbose,
    )

    logger.info("Section 2 (RRC): complete")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Orchestrate the full TexNet + RRC fetch → B3 → GIST → merge pipeline.
    Sections 1 (TexNet) and 2 (RRC) run concurrently in separate threads;
    Section 3 (merge) waits for both before writing the final CSVs.
    """
    args = parse_args()
    setup_logging(args.debug)

    logger.info("=" * 60)
    logger.info("injection_updater_v5 started")
    logger.info(
        "target-dir: %s | days: %s | dev: %s | debug: %s",
        args.target_dir, args.days, args.dev, args.debug,
    )

    if not args.target_dir.exists():
        logger.error("Target directory does not exist: %s", args.target_dir)
        sys.exit(1)

    dev  = args.dev
    tdir = args.target_dir
    rp   = lambda name: resolve_path(tdir, name, dev)  # noqa: E731 — local shorthand

    # ── Optional backup (before any writes) ─────────────────────────────────
    if args.backup_dir:
        files_to_backup = [
            rp("texnet_disposal_well.csv"), rp("texnet_disposal_inj.csv"),
            rp("texnet_gist_well_shallow.csv"), rp("texnet_gist_injection_shallow.csv"),
            rp("texnet_gist_well_deep.csv"),    rp("texnet_gist_injection_deep.csv"),
            rp("rrc_disposal_well.csv"),         rp("rrc_disposal_inj.csv"),
            rp("rrc_gist_well_shallow.csv"),     rp("rrc_gist_injection_shallow.csv"),
            rp("rrc_gist_well_deep.csv"),        rp("rrc_gist_injection_deep.csv"),
            rp("gist_well_shallow.csv"),         rp("gist_injection_shallow.csv"),
            rp("gist_well_deep.csv"),            rp("gist_injection_deep.csv"),
        ]
        backup_files(files_to_backup, args.backup_dir)

    now          = datetime.now()
    start        = datetime(2016, 1, 1)
    end_date_str = (now + timedelta(days=7)).strftime("%m-%d-%Y")
    verbose      = 1 if args.debug else 0

    logger.info(
        "TexNet date range: %s → %s | RRC lookback: %d days",
        start.strftime("%Y-%m-%d"), now.strftime("%Y-%m-%d"), args.days,
    )

    # ════════════════════════════════════════════════════════════════════════
    # SECTIONS 1 & 2 — run TexNet and RRC concurrently
    # Each section writes to non-overlapping files so there are no race
    # conditions. ThreadPoolExecutor is used (vs ProcessPoolExecutor) to
    # avoid pickling overhead and keep shared logging straightforward; the
    # work is predominantly I/O-bound (API calls + file reads/writes).
    # ════════════════════════════════════════════════════════════════════════
    logger.info("Starting TexNet and RRC sections concurrently…")
    errors = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        texnet_future = executor.submit(
            run_texnet_section,
            tdir, dev, args.debug, start, now, end_date_str, verbose,
        )
        rrc_future = executor.submit(
            run_rrc_section,
            tdir, dev, args.days, end_date_str, verbose,
        )

        for label, future in [("TexNet", texnet_future), ("RRC", rrc_future)]:
            try:
                future.result()
            except Exception as exc:
                logger.error("%s section failed: %s", label, exc, exc_info=True)
                errors.append(label)

    if errors:
        logger.error(
            "The following sections failed: %s. Merging may be incomplete.",
            ", ".join(errors),
        )

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 3 — Merge TexNet + RRC into final GIST files
    # ════════════════════════════════════════════════════════════════════════
    logger.info("-" * 40)
    logger.info("Section 3: Merging TexNet + RRC outputs into final GIST files")

    merge_datasets(
        rp("texnet_gist_well_shallow.csv"), rp("rrc_gist_well_shallow.csv"),
        rp("gist_well_shallow.csv"), dedup_cols=["ID"],
    )
    merge_datasets(
        rp("texnet_gist_well_deep.csv"), rp("rrc_gist_well_deep.csv"),
        rp("gist_well_deep.csv"), dedup_cols=["ID"],
    )
    merge_datasets(
        rp("texnet_gist_injection_shallow.csv"), rp("rrc_gist_injection_shallow.csv"),
        rp("gist_injection_shallow.csv"), dedup_cols=["ID", "Days"],
    )
    merge_datasets(
        rp("texnet_gist_injection_deep.csv"), rp("rrc_gist_injection_deep.csv"),
        rp("gist_injection_deep.csv"), dedup_cols=["ID", "Days"],
    )

    if errors:
        logger.warning("injection_updater_v5 completed with errors in: %s", ", ".join(errors))
    else:
        logger.info("injection_updater_v5 completed successfully.")
    logger.info("=" * 60)

    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
