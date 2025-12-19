from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd
from .aggregate_parts import build_i_sku, build_i_sku_user, build_i_sku_type

def _get_con():
    # Lazy import to avoid circular dependency; then return shared connection.
    from .db import get_con as _gc
    return _gc()

SeedSpec = Tuple[str, str, Path]


def base_table_specs(seeds_dir: Path) -> List[SeedSpec]:
    """Tables that can be safely loaded up-front (no FK fan-out)."""
    return [
        (
            "system",
            "SELECT team, line_down_cost, plan_budget, space_factor, reserve_cost FROM read_csv_auto(?, HEADER=TRUE)",
            seeds_dir / "system.csv",
        ),
        (
            "tech",
            "SELECT tech_id, tech AS tech_name FROM read_csv_auto(?, HEADER=TRUE)",
            seeds_dir / "tech.csv",
        ),
        (
            "location_priority",
            "SELECT location_priority, travel_time, reserve_allowed FROM read_csv_auto(?, HEADER=TRUE)",
            seeds_dir / "location_priority.csv",
        ),
        (
            "storage_location",
            "SELECT location_id, location, floor_space, current_storage_floor_space, location_priority, travel_time, reserve_allowed FROM read_csv_auto(?, HEADER=TRUE)",
            seeds_dir / "storage_location.csv",
        ),
        (
            "storage_type",
            "SELECT type_id, type, sqft_req, buy_cost, buy_invest, reloc_cost, reloc_invest, cubic_capacity_per_unit FROM read_csv_auto(?, HEADER=TRUE)",
            seeds_dir / "storage_type.csv",
        ),
        (
            "tech_map",
            "SELECT value, tech_name FROM read_csv_auto(?, HEADER=TRUE)",
            seeds_dir / "tech_map.csv",
        ),
        (
            "size",
            "SELECT size_id, size, min_vol, total_cuft FROM read_csv_auto(?, HEADER=TRUE)",
            seeds_dir / "size.csv",
        )
    ]

# Generic table replace helper
def _replace_table(df: pd.DataFrame, table_name: str, rename_map: dict | None = None) -> None:
    con = _get_con()
    # Work on a copy to avoid mutating caller's frame.
    df_copy = df.copy()
    # Optional rename to align DataFrame columns with table columns.
    if rename_map:
        df_copy = df_copy.rename(columns=rename_map)

    # Collect table column order from DuckDB and pre-create missing columns in the DataFrame.
    table_cols = [row[1] for row in con.execute(f"PRAGMA table_info('{table_name}')").fetchall()]
    if not table_cols:
        raise RuntimeError(f"Table '{table_name}' not found in DuckDB.")

    for col in table_cols:
        if col not in df_copy.columns:
            df_copy[col] = pd.NA

    # Keep only columns that exist in the table and honor their order.
    write_cols = [c for c in table_cols if c in df_copy.columns]
    tmp_view = f"__tmp_{table_name}"
    con.register(tmp_view, df_copy[write_cols])
    col_sql = ", ".join(f'"{c}"' for c in write_cols)

    # Replace table contents in a transaction to keep the DB consistent.
    con.execute("BEGIN")
    try:
        con.execute(f"DELETE FROM {table_name}")
        if not df_copy.empty:
            con.execute(f"INSERT INTO {table_name} ({col_sql}) SELECT {col_sql} FROM {tmp_view}")
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    finally:
        con.unregister(tmp_view)


def relationship_table_specs(seeds_dir: Path) -> List[SeedSpec]:
    """Tables that depend on confirmed base parameters (populated on demand)."""
    return [
        (
            "location_storage_type",
            "SELECT location_id, type_id, code, type_current_units FROM read_csv_auto(?, HEADER=TRUE)",
            seeds_dir / "location_storage_type.csv",
        ),
        (
            "tech_location_usage",
            "SELECT tech_id, location_id FROM read_csv_auto(?, HEADER=TRUE)",
            seeds_dir / "tech_location_usage.csv",
        ),
                (
            "type_size_compat",
            "SELECT type_id, size_id, max_sku_per_unit FROM read_csv_auto(?, HEADER=TRUE)",
            seeds_dir / "type_size_compat.csv",
        )
    ]


def load_lx03_seed(df_out: pd.DataFrame) -> int:
    if df_out is None or df_out.empty:
        return 0

    # Defensive checks for required columns and basic cleanup.
    df_parts = df_out.copy()
    if "Material" not in df_parts.columns:
        raise ValueError("LX03 output missing 'Material' column required for parts load.")

    # Drop rows lacking a material identifier and deduplicate by material.
    # df_parts = df_parts[df_parts["Material"].notna()]
    # print(len(df_parts))
    # df_parts = df_parts.drop_duplicates(subset=["Material"])
    # print(len(df_parts))
    if df_parts.empty:
        return 0
    
    con = _get_con()
    # Inspect table schema to align columns dynamically.
    table_cols = [row[1] for row in con.execute("PRAGMA table_info('parts')").fetchall()]
    if not table_cols:
        raise RuntimeError("parts table is not defined in DuckDB schema.")
    
    # Indicator columns are conventionally uppercase in the table schema.
    indicator_cols = [col for col in table_cols if col.isupper()]

    parts_df = pd.DataFrame(
        {
            "material": df_parts["Material"].astype("string"),
            # "priority" : pd.Series(0, index=df_parts.index, dtype="string"),
            # "movement" : pd.Series(0, index=df_parts.index, dtype="string"),
            "size": df_parts.get("Size", pd.Series(pd.NA, index=df_parts.index)).astype("string"),
            # "orders": pd.Series(0, index=df_parts.index, dtype="int64"),
            # "line_down_orders" : pd.Series(0, index=df_parts.index, dtype="int64"),
            "total_stock": df_parts.get("Total Stock", pd.Series(pd.NA, index=df_parts.index)).astype("int64"),
            # "num_users" : pd.Series(0, index=df_parts.index, dtype="int64"),
            # "ASSEMBLY" : pd.Series(0, index=df_parts.index, dtype="int64"),
            # "BODY" : pd.Series(0, index=df_parts.index, dtype="int64"),
            # "PAINT" : pd.Series(0, index=df_parts.index, dtype="int64"),
            # "UNK" : pd.Series(0, index[df_parts.index, dtype="int64"),
            "size_id": df_parts.get("size_id", pd.Series(pd.NA, index=df_parts.index)).astype("int64"),
            "storage_type": df_parts.get("storage_type", pd.Series(pd.NA, index=df_parts.index)).astype("string")
        }
    )

    # Fill in indicator columns (default to 0) and coerce to boolean.
    for col in indicator_cols:
        column = df_parts.get(col)
        if column is None:
            parts_df[col] = 0
        else:
            parts_df[col] = column.fillna(0).astype(int).clip(lower=0).astype(bool)

    # Keep only columns that exist on the target table (except updated_at).
    insert_cols = [c for c in parts_df.columns if c in table_cols and c.lower() != "updated_at"]
    if not insert_cols:
        raise RuntimeError("No matching columns between MC46 output and parts table definition.")

    parts_df = parts_df[insert_cols]

    # Register a temporary view for bulk insert.
    tmp_view = "__lx03_parts_tmp"
    con.register(tmp_view, parts_df)
    col_sql = ", ".join(f'"{c}"' for c in insert_cols)

    # Replace parts table contents atomically.
    con.execute("BEGIN")
    try:
        con.execute("DELETE FROM parts")
        con.execute(f"INSERT INTO parts ({col_sql}) SELECT {col_sql} FROM {tmp_view}")
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    finally:
        con.unregister(tmp_view)

    return len(parts_df)


def load_mc46_seed(df_out: pd.DataFrame) -> int:
    """
    Update existing parts rows (seeded via LX03) with MC46-derived attributes.
    """
    if df_out is None or df_out.empty:
        return 0

    # Copy and validate presence of the join key.
    df_parts = df_out.copy()
    if "Material" not in df_parts.columns:
        raise ValueError("MC46 output missing 'Material' column required for parts load.")

    # Only keep unique, non-null materials to reduce duplicate updates.
    df_parts = df_parts[df_parts["Material"].notna()]
    df_parts = df_parts.drop_duplicates(subset=["Material"])
    if df_parts.empty:
        return 0

    con = _get_con()
    # Capture the existing materials in the DB to avoid inserting new ones.
    table_cols = [row[1] for row in con.execute("PRAGMA table_info('parts')").fetchall()]
    if not table_cols:
        raise RuntimeError("parts table is not defined in DuckDB schema.")

    # Keep only MC46 rows whose materials already exist in parts.
    existing_materials = {row[0] for row in con.execute("SELECT material FROM parts").fetchall()}
    df_parts["Material"] = df_parts["Material"].astype("string")
    df_parts = df_parts[df_parts["Material"].isin(existing_materials)]
    if df_parts.empty:
        return 0

    # Determine which indicator columns are usable for this update.
    indicator_cols = [col for col in table_cols if col.isupper() and col in df_parts.columns]
    base_indicator_cols = ["ASSEMBLY", "BODY", "PAINT", "UNK"]

    # Build a DataFrame containing only the columns we intend to update.
    parts_df = pd.DataFrame(
        {
            "material": df_parts["Material"].astype("string"),
            "priority": df_parts.get("Priority", pd.Series(pd.NA, index=df_parts.index)).astype("string"),
            "movement": df_parts.get("Movement", pd.Series(pd.NA, index=df_parts.index)).astype("string"),
        }
    )

    # Normalize indicator columns to boolean flags (0/1) for safety.
    def normalized_indicator(col: str) -> pd.Series:
        series = df_parts.get(col, pd.Series(0, index=df_parts.index))
        return series.fillna(0).astype(int).clip(lower=0, upper=1).astype(bool)

    for col in base_indicator_cols:
        parts_df[col] = normalized_indicator(col)

    extra_indicator_cols = [col for col in indicator_cols if col not in base_indicator_cols]
    for col in extra_indicator_cols:
        parts_df[col] = normalized_indicator(col)

    # Preserve column order: material first, followed by the update set.
    update_cols = ["priority", "movement"] + base_indicator_cols + extra_indicator_cols
    parts_df = parts_df[["material"] + update_cols]

    # Register temporary view for the UPDATE ... FROM pattern.
    tmp_view = "__mc46_parts_tmp"
    con.register(tmp_view, parts_df)

    def q(col: str) -> str:
        return f'"{col}"'

    # Build SET clause only for columns that actually exist in the table.
    set_clause = ", ".join(f"{q(col)} = src.{q(col)}" for col in update_cols if col in table_cols)
    if not set_clause:
        con.unregister(tmp_view)
        return 0

    # Perform updates inside a transaction to keep data consistent.
    con.execute("BEGIN")
    try:
        con.execute(
            f"""
            UPDATE parts AS tgt
            SET {set_clause}
            FROM {tmp_view} AS src
            WHERE tgt.material = src.material
            """
        )
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    finally:
        con.unregister(tmp_view)

    return len(parts_df)

def load_cost_center(df_out: pd.DataFrame) -> int:
    """
    Load processed cost center rows into the `cost_center` table.

    Args:
        df_out: Output from `preprocess_cost_center`, containing cost centers
                and indicator columns (e.g., ASSEMBLY/BODY/PAINT).

    Returns:
        Number of cost center rows written.
    """
    if df_out is None or df_out.empty:
        return 0

    if "Cost Center" not in df_out.columns:
        raise ValueError("Cost Center output missing 'Cost Center' column required for load.")

    # Normalize to numeric, drop invalid entries, and deduplicate.
    df = df_out.copy()
    df["cost_center"] = pd.to_numeric(df["Cost Center"], errors="coerce")
    df = df[df["cost_center"].notna()]
    df = df.drop_duplicates(subset=["cost_center"])
    if df.empty:
        return 0

    # Introspect table columns so we only write compatible fields.
    con = _get_con()
    table_info = con.execute("PRAGMA table_info('cost_center')").fetchall()
    if not table_info:
        raise RuntimeError("cost_center table is not defined in DuckDB schema.")

    table_cols = [row[1] for row in table_info]
    indicator_cols = [col for col in table_cols if col.isupper()]

    # Build a frame with cost center id plus indicator flags.
    cost_df = pd.DataFrame({"cost_center": df["cost_center"].astype("Int64")})
    for col in indicator_cols:
        if col in df.columns:
            series = df[col].fillna(0)
        else:
            series = pd.Series(0, index=df.index)
        cost_df[col] = series.astype(int).clip(lower=0, upper=1).astype(bool)

    # Limit to columns that actually exist on the table (ignoring updated_at).
    insert_cols = [c for c in cost_df.columns if c in table_cols and c.lower() != "updated_at"]
    if not insert_cols:
        raise RuntimeError("No matching columns between Cost Center output and cost_center table definition.")

    cost_df = cost_df[insert_cols]

    # Stage data in a temporary view and replace existing rows atomically.
    tmp_view = "__cost_center_tmp"
    con.register(tmp_view, cost_df)
    col_sql = ", ".join(f'"{c}"' for c in insert_cols)

    con.execute("BEGIN")
    try:
        con.execute("DELETE FROM cost_center")
        con.execute(f"INSERT INTO cost_center ({col_sql}) SELECT {col_sql} FROM {tmp_view}")
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    finally:
        con.unregister(tmp_view)

    return len(cost_df)


def load_priority_updates(df_out: pd.DataFrame) -> int:
    """
    Fill missing priority values in parts table using derived priority data.
    """
    if df_out is None or df_out.empty:
        return 0

    if "Material" not in df_out.columns or "Priority" not in df_out.columns:
        raise ValueError("Priority updates require Material and Priority columns.")


    df = df_out.copy()
    df = df[df["Priority"].notna()]
    if df.empty:
        return 0

    # Only update rows that already exist in parts.
    con = _get_con()
    existing_materials = {row[0] for row in con.execute("SELECT material FROM parts").fetchall()}
    df["Material"] = df["Material"].astype("string")
    df = df[df["Material"].isin(existing_materials)]
    if df.empty:
        return 0

    # Register temporary view for the update operations.
    tmp_view = "__consumption_priority_tmp"
    con.register(tmp_view, df[["Material", "Priority", "line_down_orders"]])

    # Update priorities and line_down_orders transactionally.
    con.execute("BEGIN")
    try:
        con.execute(
            f"""
            UPDATE parts AS tgt
            SET priority = src.Priority,
            FROM {tmp_view} AS src
            WHERE tgt.material = src.Material
              AND tgt.priority IS NULL
            """
        )

        con.execute(
            """
            UPDATE parts
            SET priority = 'L'
            WHERE priority IS NULL
            """
        )

        con.execute(
            f"""
            UPDATE parts AS tgt
            SET line_down_orders = src.line_down_orders
            FROM {tmp_view} AS src
            WHERE tgt.material = src.Material
            """
        )

        con.execute(
            """
            UPDATE parts
            SET line_down_orders = COALESCE(line_down_orders, 0)
            """
        )
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    finally:
        con.unregister(tmp_view)

    return len(df)


def load_movement_updates(df_out: pd.DataFrame) -> int:
    """
    Fill missing movement values in parts table using derived movement data.
    """
    if df_out is None or df_out.empty:
        return 0

    required_cols = {"Material", "Movement", "Orders"}
    if not required_cols.issubset(df_out.columns):
        raise ValueError("Movement updates require Material, Movement, and Orders columns.")

    df = df_out.copy()
    df = df[df["Movement"].notna()]
    if df.empty:
        return 0

    # Only apply updates to materials already in parts.
    con = _get_con()
    existing_materials = {row[0] for row in con.execute("SELECT material FROM parts").fetchall()}
    df["Material"] = df["Material"].astype("string")
    df = df[df["Material"].isin(existing_materials)]
    if df.empty:
        return 0

    # Stage incoming data in a temporary view.
    tmp_view = "__consumption_movement_tmp"
    con.register(tmp_view, df[["Material", "Movement", "Orders"]])

    # Update movement and orders (fallbacks handled afterwards) transactionally.
    con.execute("BEGIN")
    try:
        con.execute(
            f"""
            UPDATE parts AS tgt
            SET movement = src.Movement,
                orders = src.Orders
            FROM {tmp_view} AS src
            WHERE tgt.material = src.Material
              AND tgt.movement IS NULL
            """
        )

        con.execute(
            """
            UPDATE parts
            SET movement = 'L'
            WHERE movement IS NULL
            """
        )

        con.execute(
            """
            UPDATE parts
            SET orders = COALESCE(orders, 0)
            """
        )
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    finally:
        con.unregister(tmp_view)

    return len(df)


def load_tech_updates(df_out: pd.DataFrame) -> int:
    """
    Update tech indicator columns on parts using derived consumption data.
    """
    if df_out is None or df_out.empty:
        return 0

    if "Material" not in df_out.columns:
        raise ValueError("Tech updates require Material column.")

    # Filter down to materials present in the database.
    df = df_out.copy()
    con = _get_con()
    table_cols = {row[1] for row in con.execute("PRAGMA table_info('parts')").fetchall()}
    indicator_cols = [col for col in df.columns if col.isupper() and col in table_cols]
    if not indicator_cols:
        raise ValueError("No valid indicator columns found for tech updates.")

    existing_materials = {row[0] for row in con.execute("SELECT material FROM parts").fetchall()}
    df["Material"] = df["Material"].astype("string")
    df = df[df["Material"].isin(existing_materials)]
    if df.empty:
        return 0

    # Normalize indicator flags to clean 0/1 integers.
    for col in indicator_cols:
        df[col] = df[col].fillna(0).astype(int).clip(lower=0, upper=1)

    # Stage the incoming indicator updates.
    tmp_view = "__consumption_tech_tmp"
    con.register(tmp_view, df[["Material"] + indicator_cols])

    # Apply updates per indicator inside a transaction for atomicity.
    con.execute("BEGIN")
    try:
        # Set concrete user columns true wherever the source is true.
        concrete_cols = [c for c in indicator_cols if c != "UNK"]
        for col in concrete_cols:
            con.execute(
                f"""
                UPDATE parts AS tgt
                SET {col} = true
                FROM {tmp_view} AS src
                WHERE tgt.material = src.Material
                  AND tgt.{col} = false
                  AND src.{col} = true
                """
            )
        # If any concrete flag is true, UNK must be cleared.
        if "UNK" in indicator_cols and concrete_cols:
            or_clause = " OR ".join(f"src.{c} = true" for c in concrete_cols)
            con.execute(
                f"""
                UPDATE parts AS tgt
                SET UNK = false
                FROM {tmp_view} AS src
                WHERE tgt.material = src.Material
                  AND tgt.UNK = true
                  AND ({or_clause})
                """
            )
            con.execute(
                f"""
                UPDATE parts
                SET UNK = false
                WHERE UNK = true AND ({' OR '.join(f'{c} = true' for c in concrete_cols)})
                """
            )

        # Recompute num_users dynamically as the sum of indicator columns.
        sum_expr = " + ".join(f"CAST({col} AS INTEGER)" for col in indicator_cols)
        con.execute(f"UPDATE parts SET num_users = {sum_expr}")
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    finally:
        con.unregister(tmp_view)

    return len(df)


def load_consumption_updates(tuple_of_dfs: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> None:
    # Tuple ordering: priority, movement, tech; apply each updater sequentially.
    priority_df, movement_df, tech_df = tuple_of_dfs
    load_priority_updates(priority_df)
    load_movement_updates(movement_df)
    load_tech_updates(tech_df)


def save_i_sku_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build i_sku, i_sku_user, i_sku_type and persist them to DuckDB tables.
    Returns the three DataFrames.
    """
    # Build each derived table in memory first.
    i_sku_df = build_i_sku()
    i_sku_user_df = build_i_sku_user()
    i_sku_type_df = build_i_sku_type()

    # Align column names to table schema where needed
    # Handle legacy column name line_down_order if present in existing DB
    con = _get_con()
    i_sku_cols = [row[1] for row in con.execute("PRAGMA table_info('i_sku')").fetchall()]
    rename_map = None
    if "line_down_order" in i_sku_cols and "line_down_orders" in i_sku_df.columns:
        rename_map = {"line_down_orders": "line_down_order"}

    # Replace tables atomically with the newly built data.
    _replace_table(i_sku_df, "i_sku", rename_map=rename_map)
    _replace_table(i_sku_user_df, "i_sku_user")
    _replace_table(i_sku_type_df, "i_sku_type")

    return i_sku_df, i_sku_user_df, i_sku_type_df


def normalize_part_priorities() -> None:
    """
    Ensure parts.priority only contains H/M/L (defaulting others to L).
    """
    con = _get_con()
    # Normalize priority strings to uppercase H/M/L; default anything else to L.
    con.execute(
        """
        UPDATE parts
        SET priority =
            CASE
                WHEN UPPER(priority) IN ('H', 'M', 'L') THEN UPPER(priority)
                ELSE 'L'
            END
        """
    )

    # Flag UNK when all other tech indicators are false (dynamic)
    cols = [row[1] for row in con.execute("PRAGMA table_info('parts')").fetchall()]
    indicator_cols = [c for c in cols if c.isupper()]
    if "UNK" in indicator_cols:
        other_cols = [c for c in indicator_cols if c != "UNK"]
        if other_cols:
            # If any concrete user flag is true, UNK must be false; else true.
            sum_expr = " + ".join(f"COALESCE(CAST({c} AS INTEGER), 0)" for c in other_cols)
            con.execute(
                f"""
                UPDATE parts
                SET UNK = CASE WHEN ({sum_expr}) > 0 THEN FALSE ELSE TRUE END
                """
            )
        else:
            con.execute("UPDATE parts SET UNK = TRUE")



