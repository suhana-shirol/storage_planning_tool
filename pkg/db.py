from __future__ import annotations
import os
from pathlib import Path
from typing import List, Tuple, Union
import re
import duckdb
from pkg.etl_loaders import base_table_specs

# Path resolution
PKG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PKG_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"
SEEDS_DIR = MODELS_DIR / "seeds"
SCHEMA_SQL = MODELS_DIR / "schema.sql"

DB_DIR = PROJECT_ROOT / "db"
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DB_DIR / "storage_planning.duckdb"

# Single shared connection
_con = None

def get_con(read_only: bool = False):
    """
    Return a DuckDB connection.
    - Default: shared read/write connection (cached) for normal app usage.
    - read_only=True: open a new read-only connection to avoid file locks across processes.
    """
    global _con
    # A read-only request always returns a fresh connection to avoid locking the shared handle.
    if read_only:
        return duckdb.connect(str(DB_PATH), read_only=True)
    # Lazily create and cache a shared connection for the process.
    if _con is None:
        _con = duckdb.connect(str(DB_PATH))
    return _con

def close_con():
    # Close and clear the cached connection, if it exists.
    global _con
    if _con is not None:
        _con.close()
        _con = None

# Schema + seeding helpers

def _exec_schema(con):
    # Execute the SQL schema file inside a transaction to build all tables.
    if not SCHEMA_SQL.exists():
        raise FileNotFoundError(f"schema.sql not found at: {SCHEMA_SQL}")
    sql_text = SCHEMA_SQL.read_text(encoding="utf-8")
    con.execute("BEGIN")
    try:
        con.execute(sql_text)
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise

def _seed_via_select(con, table_name: str, select_sql: str, params: List[str]) -> None:
    # Generic helper to replace table contents using a SELECT query (usually read_csv_auto).
    con.execute("BEGIN")
    try:
        con.execute(f"DELETE FROM {table_name}")
        con.execute(f"INSERT INTO {table_name} {select_sql}", params)
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise

def _load_all_seeds(con) -> List[Tuple[str, Union[int, str]]]:
    # Iterate over all base seed specs and load any CSV that exists.
    summary: List[Tuple[str, Union[int, str]]] = []
    for table_name, select_sql, csv_path in base_table_specs(SEEDS_DIR):
        if not csv_path.exists():
            summary.append((table_name, "skipped (missing seed file)"))
            continue
        _seed_via_select(con, table_name, select_sql, [str(csv_path)])
        # Record how many rows were inserted for reporting.
        row_count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        summary.append((table_name, row_count))
    return summary


def reset_and_seed_db():
    """Recreate DB from schema and load seed CSVs. Returns a seed summary list."""
    # Drop the DB file to guarantee a clean slate, then rebuild schema and load seeds.
    close_con()
    if DB_PATH.exists():
        os.remove(DB_PATH)
    con = get_con()  # creates fresh DB
    _exec_schema(con)
    summary = _load_all_seeds(con)
    return summary


def build_relationship_tables():
    """
    Prepare relationship tables using seed CSVs, but only insert rows whose
    foreign keys exist in the current database.
    """
    con = get_con()
    summary: List[Tuple[str, Union[int, str]]] = []

    # location_storage_type
    lst_seed = SEEDS_DIR / "location_storage_type.csv"
    if lst_seed.exists():
        con.execute("DELETE FROM location_storage_type")
        con.execute(
            """
            INSERT INTO location_storage_type (location_id, type_id, code, type_current_units)
            SELECT lst.location_id,
                   lst.type_id,
                   lst.code,
                   CAST(lst.type_current_units AS INTEGER)
            FROM read_csv_auto(?, HEADER=TRUE) AS lst
            JOIN storage_location sl ON lst.location_id = sl.location_id
            JOIN storage_type st ON lst.type_id = st.type_id
            """,
            [str(lst_seed)],
        )
        count = con.execute("SELECT COUNT(*) FROM location_storage_type").fetchone()[0]
        summary.append(("location_storage_type", count))
    else:
        summary.append(("location_storage_type", "missing seed file"))

    # tech_location_usage
    tlu_seed = SEEDS_DIR / "tech_location_usage.csv"
    if tlu_seed.exists():
        con.execute("DELETE FROM tech_location_usage")
        con.execute(
            """
            INSERT INTO tech_location_usage (tech_id, location_id)
            SELECT tlu.tech_id,
                   tlu.location_id
            FROM read_csv_auto(?, HEADER=TRUE) AS tlu
            JOIN tech t ON tlu.tech_id = t.tech_id
            JOIN storage_location sl ON tlu.location_id = sl.location_id
            """,
            [str(tlu_seed)],
        )
        count = con.execute("SELECT COUNT(*) FROM tech_location_usage").fetchone()[0]
        summary.append(("tech_location_usage", count))
    else:
        summary.append(("tech_location_usage", "missing seed file"))

    # type_size_compat
    tsc_seed = SEEDS_DIR / "type_size_compat.csv"
    if tsc_seed.exists():
        con.execute("DELETE FROM type_size_compat")
        con.execute(
            """
            INSERT INTO type_size_compat (type_id, size_id, max_sku_per_unit)
            SELECT tsc.type_id,
                   tsc.size_id,
                   CAST(tsc.max_sku_per_unit AS DOUBLE)
            FROM read_csv_auto(?, HEADER=TRUE) AS tsc
            JOIN storage_type st ON tsc.type_id = st.type_id
            JOIN size s ON tsc.size_id = s.size_id
            """,
            [str(tsc_seed)],
        )
        count = con.execute("SELECT COUNT(*) FROM type_size_compat").fetchone()[0]
        summary.append(("type_size_compat", count))
    else:
        summary.append(("type_size_compat", "missing seed file"))

    return summary


def clear_relationship_tables():
    """Remove derived relationship data so base tables can be edited safely."""
    con = get_con()
    # Simple truncate of each relationship table; keeps base tables untouched.
    for table in ("location_storage_type", "tech_location_usage", "type_size_compat"):
        con.execute(f"DELETE FROM {table}")


def _sanitize_indicator_name(name: str) -> str:
    # Uppercase, strip non-alphanumerics, and collapse to underscores.
    cleaned = re.sub(r"[^A-Z0-9_]+", "_", name.upper()).strip("_")
    if not cleaned:
        raise ValueError("Tech name must include at least one alphanumeric character.")
    return cleaned


def ensure_indicator_columns_for_tech(tech_name: str):
    """
    Ensure parts and cost_center tables have boolean columns for the given tech.
    """
    column = _sanitize_indicator_name(tech_name)
    con = get_con()
    for table in ("parts", "cost_center"):
        # Add the column if missing, then coalesce NULLs to False so booleans stay consistent.
        existing_cols = {row[1] for row in con.execute(f"PRAGMA table_info('{table}')").fetchall()}
        if column not in existing_cols:
            con.execute(f'ALTER TABLE "{table}" ADD COLUMN "{column}" BOOLEAN DEFAULT FALSE')
        con.execute(f'UPDATE "{table}" SET "{column}" = COALESCE("{column}", FALSE)')


def fetch_tech_map():
    # Return the tech map for UI/processing, ordered by value for stability.
    con = get_con()
    return con.execute("SELECT value, tech_name FROM tech_map ORDER BY value").fetchdf()


def upsert_tech_map_entry(value: str, tech_name: str):
    # Insert or update a mapping row atomically via ON CONFLICT.
    con = get_con()
    con.execute(
        """
        INSERT INTO tech_map (value, tech_name)
        VALUES (?, ?)
        ON CONFLICT (value) DO UPDATE SET tech_name = excluded.tech_name
        """,
        [value, tech_name],
    )


def delete_tech_map_entry(value: str):
    # Remove a tech_map entry matching the provided value.
    con = get_con()
    con.execute("DELETE FROM tech_map WHERE value = ?", [value])
