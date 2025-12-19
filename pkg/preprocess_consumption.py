import pandas as pd
import numpy as np
import re
from typing import Dict, Tuple

from .validate import validate, ValidationError, load_schema, enforce_output_schema
from .db import get_con  


def _coerce_dtypes(df: pd.DataFrame, dtypes: Dict[str, str]) -> pd.DataFrame:
    """
    Coerce dataframe columns to pandas dtypes declared in a schema.

    This helper is intentionally non-fatal: if a declared column is not present
    or a conversion fails, the function skips that column and continues. This
    mirrors typical ETL behavior where partial type alignment is preferable to
    stopping the pipeline.

    Args:
        df: Input dataframe to convert.
        dtypes: Mapping of column name -> pandas dtype string (e.g., "string",
            "Int64", "float64", "datetime64[ns]").

    Returns:
        A copy of `df` with best-effort dtype coercions applied.
    """
    out = df.copy()
    for col, dtype in dtypes.items():
        # Skip any declared dtype when the column is not present.
        if col not in out.columns:
            continue
        if dtype.startswith("datetime"):
            # Datetime conversion with coercion to preserve pipeline flow.
            out[col] = pd.to_datetime(out[col], errors="coerce")
        elif dtype in ("int64", "Int64"):  # Int64 = pandas NA-int
            # Strip non-numeric characters before int conversion.
            cleaned = out[col].astype(str).str.replace(r"[^\d\.\-]", "", regex=True)
            converted = pd.to_numeric(cleaned, errors='coerce')
            target_dtype = "Int64" if dtype == "Int64" else "int64"
            try:
                if target_dtype == "Int64":
                    out[col] = converted.round().astype("Int64")
                else:
                    out[col] = converted.round().astype("int64")
            except (TypeError, ValueError):
                out[col] = converted
        elif dtype in ("float64", "Float64"):
            # Remove thousands separators and coerce to float.
            cleaned = out[col].astype(str).str.replace(',', '')
            converted = pd.to_numeric(cleaned, errors="coerce")
            target_dtype = "Float64" if dtype == "Float64" else "float64"
            try:
                out[col] = converted.astype(target_dtype)
            except (TypeError, ValueError):
                out[col] = converted
        elif dtype == "string":
            out[col] = out[col].astype("string")
        else:
            # Fallback: best-effort cast, ignore on failure.
            try:
                out[col] = out[col].astype(dtype)
            except Exception:
                pass
    return out

def get_known_materials() -> pd.Series:
    # Query the canonical parts list so consumption rows can be filtered.
    con = get_con()
    try:
        df_materials = con.execute(
            """
            SELECT material
            FROM parts
            """
        ).df()
    except Exception as exc:
        raise RuntimeError(
            "Failed to read parts from DuckDB. Ensure schema is initialized."
        ) from exc
    return df_materials["material"].astype("string")

def get_cost_center_to_user() -> pd.DataFrame:
    # Pull the cost center table to map centers to tech/user indicators.
    con = get_con()
    try:
        df_cc_to_user = con.execute(
            """
            SELECT *
            FROM cost_center
            """
        ).df()
    except Exception as exc:
        raise RuntimeError(
            "Failed to read cost_center from DuckDB. Ensure schema is initialized."
        ) from exc

    # Normalize column naming and dtype for joining.
    if "cost_center" in df_cc_to_user.columns and "Cost Center" not in df_cc_to_user.columns:
        df_cc_to_user = df_cc_to_user.rename(columns={"cost_center": "Cost Center"})
    if "Cost Center" in df_cc_to_user.columns:
        df_cc_to_user["Cost Center"] = df_cc_to_user["Cost Center"].astype("string").str.strip()
    
    return df_cc_to_user


# -------- Priority Functions --------

def derive_priority(df_in: pd.DataFrame) -> pd.DataFrame:
    # Materials with at least one line down order over the past two years get Priority category H (High)
    # Otherwise material will get Priority category 'L' (Low)
    # Keep only the fields needed to compute the metric.
    df = df_in.copy()
    df = df[["Material", "Order Type"]].dropna(subset=["Material"])

    # Count line-down orders per material.
    counts = (
        df.groupby("Material")["Order Type"]
        .apply(lambda x: (x == "OR09").sum())
        .rename("line_down_orders")
        .reset_index()
    )
    # Assign H if any line-down orders exist, else L.
    counts["Priority"] = np.where(counts["line_down_orders"] > 0, "H", "L")    
    # counts.to_csv("priority_check.csv")
    return counts


# -------- Movement Functions --------

def derive_movement(df_in: pd.DataFrame) -> pd.DataFrame:
    # Copy and keep only the columns needed for order counting.
    df = df_in.copy()
    df = df[["Material", "Order"]]

    # per-material order count metric -> "Orders"
    orders_by_mat = (
        df.groupby("Material")["Order"]
          .nunique()
        .rename("Orders")
        .reset_index()
    )

    # Pareto on Orders to assign Movement
    # Sort descending, compute cumulative count, and derive percent of total.
    pareto_df = (
        orders_by_mat
        .sort_values(by="Orders", ascending=False)
        .assign(**{"Cumulative Orders": lambda x: x["Orders"].cumsum()})
    )
    total = int(pareto_df["Cumulative Orders"].iloc[-1]) if len(pareto_df) else 0
    pareto_df["% of Transactions"] = 0.0 if total == 0 else pareto_df["Cumulative Orders"] / total
    pareto_df["Movement"] = np.where(pareto_df["% of Transactions"] <= 0.80, "H", "M")

    # IMPORTANT: do NOT drop "Order" (schema requires it).
    # Merge Movement + Orders (metric) back to each row
    df = pareto_df[["Material", "Movement", "Orders"]]
    return df


# -------- Technologies Functions --------

def _prune_unk(df: pd.DataFrame, indicator_cols: list[str]) -> pd.DataFrame:
    """Force UNK=0 whenever any concrete user column is 1."""
    if "UNK" not in indicator_cols:
        return df

    # Separate known user columns from the UNK placeholder.
    concrete_cols = [col for col in indicator_cols if col != "UNK"]
    if not concrete_cols:
        return df

    # When any concrete user is flagged, UNK must be 0 for that row.
    normalized = df[concrete_cols].fillna(0).astype(int).clip(lower=0, upper=1)
    mask = normalized.any(axis=1)
    df.loc[mask, "UNK"] = 0
    df["UNK"] = df["UNK"].fillna(0).astype(int)
    return df


def keep_only_known_materials(df_in):
    # Filter consumption rows to materials that exist in the parts table.
    known_materials = set(get_known_materials())
    df = df_in.copy()
    df = df[df["Material"].astype("string").isin(known_materials)]
    return df

def derive_tech(df_in):
    # Start from a narrowed view containing the join key.
    df = df_in.copy()
    df = df[["Material", "Cost Center"]].copy()
    df["Cost Center"] = df["Cost Center"].astype("string").str.strip()
    # Lookup cost center -> user indicators from the database.
    cost_center = get_cost_center_to_user().copy()

    # Normalize the join key so it matches the consumption frame.
    if "Cost Center" not in cost_center.columns:
        if "cost_center" in cost_center.columns:
            cost_center = cost_center.rename(columns={"cost_center": "Cost Center"})
        else:
            raise KeyError("cost_center table missing Cost Center column.")

    # The indicator columns are the uppercase ones on the cost center table.
    indicator_cols = [col for col in cost_center.columns if col.isupper()]
    if not indicator_cols:
        raise ValueError("No indicator columns (e.g., ASSEMBLY/BODY/PAINT/UNK) found in cost_center table.")

    # Keep only the join key plus indicator columns.
    cc_users = cost_center[["Cost Center"] + indicator_cols].copy()

    # Normalize indicators to 0/1 ints so downstream code gets clean columns.
    for col in indicator_cols:
        series = pd.to_numeric(cc_users[col], errors="coerce").fillna(0)
        cc_users[col] = series.astype(int).clip(lower=0, upper=1)

    # Attach user flags to each consumption row based on cost center.
    df = df.merge(cc_users, on="Cost Center", how="left")

    # Track rows where the cost center lookup failed so we can flag UNK later.
    missing_cc_mask = df[indicator_cols].isna().all(axis=1)

    # Fill any cost centers missing in the lookup with zeros.
    for col in indicator_cols:
        df[col] = df[col].fillna(0).astype(int)

    if "UNK" in indicator_cols:
        df.loc[missing_cc_mask, "UNK"] = 1

    # Consolidate duplicate materials by OR-ing the indicator columns.
    agg_map = {col: "max" for col in indicator_cols}
    df = (
        df.groupby("Material", dropna=False)
        .agg(agg_map)
        .reset_index()
    )

    df = _prune_unk(df, indicator_cols)

    return df

def preprocess_consumption(df_in: pd.DataFrame, schema_name: str = "consumption") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Load the input schema so we can validate and coerce types.
    schema = load_schema(schema_name)

    # Operate on a copy of the raw input.
    df = df_in.copy()

    if "dtypes" in schema:
        # Align input dtypes (best effort) prior to validation.
        df = _coerce_dtypes(df, schema["dtypes"])

    # Validate and trim to required columns.
    df, missing, extra = validate(df, schema_name=schema_name)
    df = df.copy()

    # Derive each of the three downstream artifacts from the cleaned frame.
    priority = derive_priority(df)
    movement = derive_movement(df)
    techs = derive_tech(df)

    # Enforce schemas for each outgoing table.
    priority_out_schema = load_schema("material_to_priority")
    material_to_priority = enforce_output_schema(priority, priority_out_schema)

    movement_out_schema = load_schema("material_to_movement")
    material_to_movement = enforce_output_schema(movement, movement_out_schema)

    techs_out_schema = load_schema("material_to_tech")
    material_to_tech = enforce_output_schema(techs, techs_out_schema)

    return (material_to_priority, material_to_movement, material_to_tech)
