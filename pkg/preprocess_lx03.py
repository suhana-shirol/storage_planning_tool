import pandas as pd
import numpy as np
import re
from typing import Dict, Optional

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
        # Skip coercion when declared column is missing.
        if col not in out.columns:
            continue
        if dtype.startswith("datetime"):
            # Convert to datetime, invalid strings become NaT.
            out[col] = pd.to_datetime(out[col], errors="coerce")
        elif dtype in ("int64", "Int64"):  # Int64 = pandas NA-int
            # Strip non-numeric characters before integer conversion.
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
            # Remove commas and coerce to float.
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
            # fallback
            try:
                out[col] = out[col].astype(dtype)
            except Exception:
                pass
    return out

def _load_location_storage_type_codes() -> pd.DataFrame:
    """
    Fetch the location/type/code relationships from DuckDB.

    Returns:
        DataFrame with at least `location_id`, `type_id`, and `code`.
    """
    con = get_con()
    try:
        df_codes = con.execute(
            """
            SELECT type_id, code
            FROM location_storage_type
            """
        ).df()
    except Exception as exc:
        raise RuntimeError(
            "Failed to read location_storage_type codes from DuckDB. Ensure schema is initialized."
        ) from exc

    return df_codes

def _load_size() -> pd.DataFrame:
    con = get_con()
    try:
        df_size = con.execute(
            """
            SELECT size_id, size
            FROM size
            """
        ).df()
    except Exception as exc:
        raise RuntimeError(
            "Failed to read sizes from DuckDB. Ensure schema is initialized."
        ) from exc
    return df_size

# You can move these to a config file if you want them tunable
SHELF_CUTOFF = 20
REMSTAR_CUTOFF = 30


def preprocess_lx03(lx03_df: pd.DataFrame, schema_name: str = "lx03") -> pd.DataFrame:
    """
    Transform raw LX03 data into a clean table with Material, Total Stock, and Size.

    Parameters
    ----------
    lx03_df : DataFrame
        Raw LX03 data (what you previously read from 'LX03.csv').
        Must have at least `Material`, `Storage Type`, and `Total Stock` columns
        (names can be overridden via parameters).
    storage_type_dim : DataFrame
        Dimension table for storage types, typically loaded from your DB's
        `storage_type` table. Must contain at least:
          - `storage_type_code_col` (e.g. 'code', 'Storage_Type', ...)
          - `storage_type_group_col` (e.g. 'type_group' = 'Shelf'/'Remstar'/'Rack'/'Bulk')

    Returns
    -------
    DataFrame
        Columns: Material, Total Stock, Size, Type, Storage_Type
        (you can trim columns before inserting into your own table).
    """
    schema = load_schema(schema_name)

    # --- 1. Start from a copy and basic cleaning ---
    df = lx03_df.copy()

    if "dtypes" in schema:
        # Best-effort dtype alignment guided by schema.
        df = _coerce_dtypes(df, schema["dtypes"])
        if "Total Stock" in df.columns:
            # Strip non-numeric characters before converting to integer stock.
            cleaned_stock = (
                df["Total Stock"]
                .astype(str)
                .str.replace(r"[^\d\.\-]", "", regex=True)
            )
            df["Total Stock"] = pd.to_numeric(cleaned_stock, errors="coerce").astype("Int64")

    # Validate columns against schema and trim to required set.
    df, missing, extra = validate(df, schema_name=schema_name)
    df = df.copy()

    # Drop rows with missing Material (same as notebook)
    df = df.dropna(subset=["Material"])

    # Keep only the columns we actually use
    df = df[["Material", "Storage Type", "Total Stock"]]

    # get type to code from location_storage_type
    type_to_code = _load_location_storage_type_codes()

    # Standardize name for joining
    df = df.rename(columns={"Storage Type": "code"})

    # --- 2. Join to storage_type dimension (replaces code_to_type.csv) ---
    # Expect storage_type_dim to have columns like:
    #   - code          (raw Storage_Type code, e.g. 'NS1', 'RE5', ...)
    #   - type_group    (high-level type: 'Shelf', 'Remstar', 'Rack', 'Bulk')

    df = df.merge(type_to_code, on="code", how="left")

    

    # --- 5. Initialize Size and apply sizing rules (exactly like LX03 section) ---
    df["Size"] = pd.NA

    # Remap type_id to size buckets based on business rules.
    df.loc[(df["type_id"] == 1), "Size"] = "S"

    # Shelf with Total Stock > 20 -> S
    df.loc[(df["type_id"] == 3) & (df["Total Stock"] > SHELF_CUTOFF), "Size"] = "S"

    # Remstar with Total Stock > 30 -> S
    df.loc[(df["type_id"] == 4) & (df["Total Stock"] > REMSTAR_CUTOFF), "Size"] = "S"

    # Shelf or Remstar that weren't already S -> M
    df.loc[
        df["Size"].isna() & df["type_id"].isin([3, 4]),
        "Size"
    ] = "M"

    # Rack -> L
    df.loc[df["type_id"] == 5, "Size"] = "L"

    # size to size_id map
    size_df = _load_size()
    size_map = dict(zip(size_df["size"], size_df["size_id"]))

    # Attach size_id for downstream compatibility logic.
    df["size_id"] = df["Size"].map(size_map)

    # Drop rows lacking a material identifier and deduplicate by material.
    df = df[df["Material"].notna()]
    df = df.drop_duplicates(subset=["Material"], keep="first")

    # Align naming with downstream expectation.
    df = df.rename(columns = {"code": "storage_type"})

    # Enforce output schema to guarantee column presence/order before return.
    out_schema = load_schema("lx03_output")
    df = enforce_output_schema(df, out_schema)
    # If you want to keep only the minimal set, uncomment the next line:
    # df = df[["Material", "Total Stock", "Size"]]

    return df
