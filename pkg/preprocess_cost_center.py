from typing import Dict
import pandas as pd
import numpy as np
import re

from .validate import validate, ValidationError, load_schema, enforce_output_schema

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
        # Skip dtype coercion if the declared column is absent.
        if col not in out.columns:
            continue
        if dtype.startswith("datetime"):
            # Convert to datetime, coercing invalid values to NaT.
            out[col] = pd.to_datetime(out[col], errors="coerce")
        elif dtype in ("int64", "Int64"):  # Int64 = pandas NA-int
            # Remove thousands separators before integer conversion.
            cleaned = out[col].astype(str).str.replace(',', '')
            converted = pd.to_numeric(cleaned, errors='coerce')
            target_dtype = "Int64" if dtype == "Int64" else "int64"
            try:
                out[col] = converted.astype(target_dtype)
            except (TypeError, ValueError):
                out[col] = converted
        elif dtype in ("float64", "Float64"):
            # Coerce to float and keep best effort on failure.
            converted = pd.to_numeric(out[col], errors="coerce")
            target_dtype = "Float64" if dtype == "Float64" else "float64"
            try:
                out[col] = converted.astype(target_dtype)
            except (TypeError, ValueError):
                out[col] = converted
        elif dtype == "string":
            out[col] = out[col].astype("string")
        else:
            # Fallback: attempt cast and ignore failures.
            try:
                out[col] = out[col].astype(dtype)
            except Exception:
                pass
    return out

def preprocess_cost_center(df_in: pd.DataFrame, schema_name: str = "cost_center"):
    # Read the schema definition so we know required columns and dtypes.
    schema = load_schema(schema_name)
    # Work on a copy to avoid mutating the caller's DataFrame.
    df = df_in.copy()
    # Try to align incoming column types with the schema hints.
    df = _coerce_dtypes(df, schema["dtypes"])

    # Validate the incoming frame; `validate` also trims to required columns.
    cost_center_req, missing, extra = validate(df, schema_name)

    # Use the validated frame as the new working DataFrame.
    df = cost_center_req.copy()

    # Keep only departments whose codes start with the TX-* prefixes of interest.
    df = df[df["Department"].astype(str).str.startswith(("TX-2", "TX-3", "TX-4"), na=False)]

    # Build boolean masks for each department prefix and map them to user labels.
    conditions = [
        df["Department"].str.startswith("TX-2", na=False),
        df["Department"].str.startswith("TX-3", na=False),
        df["Department"].str.startswith("TX-4", na=False)
    ]
    choices = ["BODY", "PAINT", "ASSEMBLY"]

    # Assign a user value per row; unknown prefixes become "UNK".
    df = df.copy()
    df["User"] = np.select(conditions, choices, default="UNK")

    # Only keep rows for users we explicitly track as indicator columns.
    indicator_cols = ["ASSEMBLY", "BODY", "PAINT", "UNK"]
    df = df[df["User"].isin(indicator_cols)]

    if df.empty:
        # If nothing matched, return an empty table with the expected columns.
        df = pd.DataFrame(columns=["Cost Center"] + indicator_cols)
    else:
        # Mark each row for pivoting into indicator columns.
        df["value"] = 1
        # Pivot to one row per cost center with 0/1 flags for each user bucket.
        df = (
            df.pivot_table(
                index="Cost Center",
                columns="User",
                values="value",
                aggfunc="max",
                fill_value=0,
            )
            .reset_index()
        )
        # Clean up pivot column naming.
        df.columns.name = None
        # Ensure all indicator columns exist even if absent from the pivot result.
        for col in indicator_cols:
            if col not in df.columns:
                df[col] = 0
        # Reorder columns and coerce indicator flags to ints.
        df = df[["Cost Center"] + indicator_cols]
        df[indicator_cols] = df[indicator_cols].astype(int)

    # Enforce the declared output schema before returning.
    out_schema = load_schema("cost_center_output")
    df = enforce_output_schema(df, out_schema)

    return df
