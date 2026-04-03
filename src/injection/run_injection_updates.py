"""
run_injection_updates.py
Wrapper script to run RRC and TexNet injection data updates sequentially
and merge their outputs into a single dataset for GIST.

Usage:
    python run_injection_updates.py --target-dir ./src/data --days 90
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path
import pandas as pd
from logging.handlers import RotatingFileHandler
from injection_id_utils import apply_uic_normalization

logger = logging.getLogger(__name__)


def setup_logging(debug: bool) -> None:
    log_level = logging.DEBUG if debug else logging.INFO
    log_file = Path(__file__).parent / "run_injection_updates.log"
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

def start_script(script_path: Path, args: list) -> subprocess.Popen:
    cmd = [sys.executable, str(script_path)] + args
    logger.info(f"Starting: {' '.join(cmd)}")
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def wait_script(proc: subprocess.Popen, script_path: Path) -> bool:
    stdout, stderr = proc.communicate()
    if proc.returncode == 0:
        logger.debug(stdout)
        return True
    logger.error(f"Script {script_path.name} failed with exit code {proc.returncode}")
    logger.error(stderr)
    return False

def merge_datasets(rrc_file: Path, texnet_file: Path, output_file: Path, dedup_cols: list) -> None:
    """Merge RRC and TexNet dataframes, deduplicate, and save to output_file."""
    dfs = []
    if rrc_file.exists():
        dfs.append(pd.read_csv(rrc_file, dtype={"ID": str}, low_memory=False))
        logger.info(f"Loaded {len(dfs[-1])} rows from {rrc_file.name}")
    else:
        logger.warning(f"RRC file not found: {rrc_file}")

    if texnet_file.exists():
        dfs.append(pd.read_csv(texnet_file, dtype={"ID": str}, low_memory=False))
        logger.info(f"Loaded {len(dfs[-1])} rows from {texnet_file.name}")
    else:
        logger.warning(f"TexNet file not found: {texnet_file}")

    if not dfs:
        logger.error(f"No data to merge for {output_file.name}")
        return

    combined = pd.concat(dfs, ignore_index=True)
    before = len(combined)

    # Normalize ID before deduplication to ensure RRC and TexNet IDs align
    if "ID" in combined.columns:
        combined = apply_uic_normalization(combined, ["ID"])

    # Deduplicate: keep the last record (TexNet usually runs second, so it would win if there's a collision)
    combined.drop_duplicates(subset=dedup_cols, keep="last", inplace=True)
    
    # Sort by ID and Date if applicable
    sort_cols = []
    if "ID" in combined.columns: sort_cols.append("ID")
    if "Days" in combined.columns: sort_cols.append("Days")
    elif "Date" in combined.columns: sort_cols.append("Date")

    if sort_cols:
        combined.sort_values(by=sort_cols, inplace=True)

    combined.to_csv(output_file, index=False)
    logger.info(f"Merged {output_file.name}: {len(combined)} rows (removed {before - len(combined)} duplicates)")

def main():
    parser = argparse.ArgumentParser(description="GIST Injection Data Update Wrapper")
    parser.add_argument("--target-dir", required=True, type=Path, help="Directory for CSV files")
    parser.add_argument("--days", type=int, default=90, help="Look-back days for RRC data")
    parser.add_argument("--dev", action="store_true", help="Use dev suffix for filenames")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    setup_logging(args.debug)
    logger.info("=" * 60)
    logger.info("GIST Injection Update Wrapper Started")

    script_dir = Path(__file__).parent
    rrc_script = script_dir / "get_rrc_well_injection_data.py"
    texnet_script = script_dir / "injection_updater_V4.py"

    # 1. Launch RRC and TexNet scripts in parallel (they write to non-overlapping files)
    # Note: --debug is intentionally NOT forwarded to subscripts; their injectionV3 verbose
    # print() output is too noisy. Run subscripts directly if per-well verbosity is needed.
    rrc_args = ["--target-dir", str(args.target_dir), "--days", str(args.days)]
    if args.dev: rrc_args.append("--dev")

    texnet_args = ["--target-dir", str(args.target_dir)]
    if args.dev: texnet_args.append("--dev")

    rrc_proc = start_script(rrc_script, rrc_args)
    texnet_proc = start_script(texnet_script, texnet_args)

    # 2. Wait for both to finish before merging
    if not wait_script(rrc_proc, rrc_script):
        logger.error("RRC update failed. Merging may be incomplete.")
    if not wait_script(texnet_proc, texnet_script):
        logger.error("TexNet update failed. Merging may be incomplete.")

    # 3. Merge results from both scripts
    suffix = "_dev" if args.dev else ""
    
    file_pairs = [
        # (RRC Filename, TexNet Filename, Final Filename, Dedup Columns)
        (f"rrc_gist_well_shallow{suffix}.csv", f"gist_well_shallow{suffix}.csv", f"gist_well_shallow{suffix}.csv", ["ID"]),
        (f"rrc_gist_well_deep{suffix}.csv", f"gist_well_deep{suffix}.csv", f"gist_well_deep{suffix}.csv", ["ID"]),
        (f"rrc_gist_injection_shallow{suffix}.csv", f"gist_injection_shallow{suffix}.csv", f"gist_injection_shallow{suffix}.csv", ["ID", "Days"]),
        (f"rrc_gist_injection_deep{suffix}.csv", f"gist_injection_deep{suffix}.csv", f"gist_injection_deep{suffix}.csv", ["ID", "Days"]),
    ]

    for rrc_name, texnet_name, final_name, dedup_cols in file_pairs:
        # Note: injection_updater_V4 already outputs to gist_well_shallow.csv etc.
        # We need to be careful not to overwrite the TexNet output before we merge it.
        # However, the RRC script outputs to rrc_gist_well_shallow.csv etc.
        # So we merge rrc_gist_well_shallow + gist_well_shallow -> gist_well_shallow.
        
        rrc_path = args.target_dir / rrc_name
        texnet_path = args.target_dir / texnet_name
        final_path = args.target_dir / final_name
        
        merge_datasets(rrc_path, texnet_path, final_path, dedup_cols)

    logger.info("GIST Injection Update Wrapper Completed Successfully")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
