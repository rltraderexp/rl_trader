"""
High-performance data ingestion pipeline using DuckDB.

This module provides a function to build and append to a columnar database
from a directory of raw, gzipped CSV files. It handles schema creation,
de-duplication, and data quality checks like gap detection. It is robust to
various line endings, column name variations ('time' vs 'timestamp'), and
other common data quality issues like malformed files, API error messages,
and non-standard empty line separators.
"""
import duckdb
import pandas as pd
from pathlib import Path
import re
from typing import Dict, Any, Tuple, Optional
import gzip

def _inspect_csv_header(file_path: Path) -> Tuple[bool, Optional[str], list]:
    """
    Inspects the CSV header to find the time column, check for validity,
    and return all column names.
    """
    try:
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            header_line = ""
            while not header_line.strip():
                header_line = f.readline()
                if not header_line: return False, None, []

            columns_lower = {col.strip().lower() for col in header_line.split(',')}
            original_columns = [col.strip() for col in header_line.split(',')]
            
            required_cols = {'open', 'high', 'low', 'close', 'volume'}
            if not required_cols.issubset(columns_lower):
                return False, None, []

            time_col_name_lower = 'timestamp' if 'timestamp' in columns_lower else 'time' if 'time' in columns_lower else None
            if not time_col_name_lower:
                return False, None, []

            original_time_col = next((col for col in original_columns if col.lower() == time_col_name_lower), None)
            
            return True, original_time_col, original_columns
            
    except (EOFError, StopIteration, ValueError, gzip.BadGzipFile):
        return False, None, []

def ingest_raw_data_to_duckdb(raw_dir: str, db_path: str, source_timezone: str = 'US/Eastern') -> Dict[str, Any]:
    """
    Ingests raw OHLCV data from gzipped CSV files into a DuckDB database.
    """
    summary = {
        'files_found': 0, 'files_processed': 0, 'files_skipped_errors': 0,
        'rows_added': 0, 'gaps_found': 0, 'status': 'started'
    }
    
    raw_path = Path(raw_dir)
    if not raw_path.is_dir():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    con = duckdb.connect(database=db_path, read_only=False)
    con.execute(f"SET TimeZone='{source_timezone}';")
    
    con.execute("""
        CREATE TABLE IF NOT EXISTS ohlcv (
            timestamp TIMESTAMPTZ, asset VARCHAR, open DOUBLE, high DOUBLE,
            low DOUBLE, close DOUBLE, volume DOUBLE, UNIQUE(timestamp, asset)
        );
    """)
    con.execute("CREATE TABLE IF NOT EXISTS ingestion_log (filename VARCHAR PRIMARY KEY, ingested_at TIMESTAMPTZ);")
    
    all_files = list(raw_path.glob('*.csv.gz'))
    summary['files_found'] = len(all_files)
    
    processed_files = set(con.execute("SELECT filename FROM ingestion_log").df()['filename'])
    files_to_process = [f for f in all_files if f.name not in processed_files]
    
    initial_rows = con.execute("SELECT COUNT(*) FROM ohlcv").fetchone()[0]
    
    for f_path in files_to_process:
        asset_match = re.match(r"([A-Z0-9_]+)__\d{4}-\d{2}\.csv\.gz", f_path.name)
        if not asset_match:
            print(f"Skipping file with unexpected name format: {f_path.name}")
            continue
        
        asset_symbol = asset_match.group(1)
        
        is_valid, time_col, all_cols = _inspect_csv_header(f_path)
        if not is_valid:
            print(f"  [WARNING] File {f_path.name} has an invalid header, is empty, or is not a valid CSV. Skipping.")
            summary['files_skipped_errors'] += 1
            continue

        print(f"Processing {f_path.name} for asset {asset_symbol} (time_col: '{time_col}')")

        column_types_varchar = {col: 'VARCHAR' for col in all_cols}

        query = f"""
        INSERT INTO ohlcv
        WITH raw_data AS (
            SELECT *
            FROM read_csv(
                '{str(f_path)}',
                auto_detect=false,
                strict_mode=false,
                delim=',',
                header=false,
                skip=1,
                columns={column_types_varchar},
                ignore_errors=true
            )
        )
        SELECT
            strptime("{time_col}", '%Y-%m-%d %H:%M:%S') AS timestamp,
            '{asset_symbol}' AS asset,
            TRY_CAST(open AS DOUBLE) AS open,
            TRY_CAST(high AS DOUBLE) AS high,
            TRY_CAST(low AS DOUBLE) AS low,
            TRY_CAST(close AS DOUBLE) AS close,
            TRY_CAST(volume AS DOUBLE) AS volume
        FROM raw_data
        WHERE strptime("{time_col}", '%Y-%m-%d %H:%M:%S') IS NOT NULL
        ON CONFLICT (timestamp, asset) DO NOTHING;
        """
        
        try:
            con.execute(query)
            con.execute("INSERT INTO ingestion_log VALUES (?, now())", [f_path.name])
            summary['files_processed'] += 1
        except (duckdb.InvalidInputException, duckdb.BinderException, duckdb.ConversionException) as e:
            print(f"  [WARNING] DuckDB could not process file {f_path.name}. Skipping. Error: {e}")
            summary['files_skipped_errors'] += 1
            continue

    final_rows = con.execute("SELECT COUNT(*) FROM ohlcv").fetchone()[0]
    summary['rows_added'] = final_rows - initial_rows
    
    gap_query = """
    WITH ordered_ts AS (
        SELECT asset, timestamp, LAG(timestamp, 1) OVER (PARTITION BY asset ORDER BY timestamp) as prev_timestamp
        FROM ohlcv
    )
    SELECT asset, prev_timestamp, timestamp, (timestamp - prev_timestamp) as gap
    FROM ordered_ts
    WHERE (timestamp - prev_timestamp) > INTERVAL '1 minute'
    ORDER BY asset, prev_timestamp;
    """
    gaps_df = con.execute(gap_query).df()
    summary['gaps_found'] = len(gaps_df)
    if not gaps_df.empty:
        print("\n--- Detected Data Gaps ---")
        print(gaps_df.to_string())
    
    con.close()
    summary['status'] = 'completed'
    return summary