import pandas as pd
import numpy as np
import re
from typing import Dict

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
        # Skip coercion if the column does not exist on the frame.
        if col not in out.columns:
            continue
        if dtype.startswith("datetime"):
            # Convert to datetime, coercing invalid entries to NaT.
            out[col] = pd.to_datetime(out[col], errors="coerce")
        elif dtype in ("int64", "Int64"):  # Int64 = pandas NA-int
            # Drop comma separators before integer conversion.
            cleaned = out[col].astype(str).str.replace(',', '')
            converted = pd.to_numeric(cleaned, errors='coerce')
            target_dtype = "Int64" if dtype == "Int64" else "int64"
            try:
                out[col] = converted.astype(target_dtype)
            except (TypeError, ValueError):
                out[col] = converted
        elif dtype in ("float64", "Float64"):
            # Convert to float, coercing errors to NaN.
            converted = pd.to_numeric(out[col], errors="coerce")
            target_dtype = "Float64" if dtype == "Float64" else "float64"
            try:
                out[col] = converted.astype(target_dtype)
            except (TypeError, ValueError):
                out[col] = converted
        elif dtype == "string":
            out[col] = out[col].astype("string")
        else:
            # Fallback: attempt cast and swallow failures.
            try:
                out[col] = out[col].astype(dtype)
            except Exception:
                pass
    return out


def preprocess_mc46(df_in: pd.DataFrame, schema_name: str = "mc46"):
    """
    Clean and normalize MC46 input data and enforce the output schema.

    Steps performed:
    - Load and validate against the declared input schema.
    - Best-effort dtype coercion using schema hints.
    - Normalize and split the free-text user field "Ind. Std Desc.".
    - Map split tokens to standardized user groups and aggregate to `Users`.
    - Apply business rule (MRP Controller 899 -> Movement 'O') and drop helper columns.
    - Enforce the final output schema before returning.

    Args:
        df_in: Raw MC46 dataframe to process.
        schema_name: Input schema name used for validation (default: "mc46").

    Returns:
        A normalized dataframe conforming to the "mc46_output" schema.
    """
    # Load schema metadata (columns, optional dtypes, constraints, etc.).
    schema = load_schema(schema_name)

    # Work on a copy so caller's frame is not mutated.
    df_work = df_in.copy()
    
    # Align dtypes when declared by schema (best-effort, non-fatal).
    if "dtypes" in schema:
        df_work = _coerce_dtypes(df_work, schema["dtypes"])

    # Validate and standardize the input frame; capture missing/extra columns.
    df, missing, extra = validate(df_work, schema_name=schema_name)
    df = df.copy()

    # fill null "Ind. Std Desc" with "UNK" = unknown user
    df['Ind. Std Desc.'] = df['Ind. Std Desc.'].fillna("UNK")
    # If user '0' or '?' fill with "UNK"
    df.loc[(df['Ind. Std Desc.'] == '0') | (df['Ind. Std Desc.'] == '?') , 'Ind. Std Desc.'] = 'UNK'

    # split 'Ind. Std Desc.' string and create new columns for them
    pat = r'[,;/]'
    split_users_df = df['Ind. Std Desc.'].str.split(pat, expand = True, regex=True).apply(lambda col: col.str.strip())
    # rename those columns
    split_users_df.columns = [f'Split_{i+1}' for i in range(split_users_df.shape[1])]

    def load_tech_map_df() -> pd.DataFrame:
        # Provide sensible defaults, then attempt to read the tech_map table.
        defaults = pd.DataFrame({
            'Value': ['A', 'B', 'P', 'BODY', 'LC', 'UNK'],
            'Tech':  ['ASSEMBLY', 'BODY', 'PAINT', 'BODY', 'UNK', 'UNK']
        })
        try:
            con = get_con()
            result = con.execute("SELECT value AS Value, tech_name AS Tech FROM tech_map").fetchdf()
            if result.empty:
                return defaults
            return result
        except Exception:
            return defaults

    tech_map_df = load_tech_map_df()

    # Normalize lookup
    tech_map_df = tech_map_df.assign(
        Value=tech_map_df['Value'].str.strip().str.upper(),
        Tech=tech_map_df['Tech'].str.strip().str.upper(),
    )
    # Create a mapping dict keyed by value code (case-insensitive).
    lookup_map = tech_map_df.drop_duplicates('Value').set_index('Value')['Tech'].to_dict()

    # Detect split columns automatically
    split_cols = split_users_df.columns

    def is_missing_like(x):
        """Return True for pd.NA/NaN/None and common stringified missings."""
        if pd.isna(x):
            return True
        if isinstance(x, str) and x.strip().upper() in {"<NA>", "NAN", "NONE", ""}:
            return True
        return False

    def map_keys(val):
        # 1) If Ind. Std Desc. code is Null return none for user
        if is_missing_like(val):
            return None

        # 2) Normalize to string for substring matching
        s = val if isinstance(val, str) else str(val)
        su = s.upper()

        # 3) Substring match (case-insensitive)
        matched = [v for k, v in lookup_map.items()
                if isinstance(k, str) and k.upper() in su]

        # 4) No matches → "UNK"; else joined unique labels
        if not matched:
            return "UNK"
        # optional: de-duplicate while preserving order
        seen, out = set(), []
        for v in matched:
            if v not in seen:
                seen.add(v)
                out.append(v)
        return ", ".join(out)
    
    # Map user codes using helper functions
    user_cols = []
    for i, col in enumerate(split_cols):
        # Convert each split token to a normalized user code.
        mc46_split_col = split_users_df[f'{col}']
        split_users_df[f'User_{i+1}'] = mc46_split_col.astype(str).apply(map_keys)

        if f'User_{i+1}' not in user_cols:
            user_cols.append(f'User_{i+1}')
    
    def extract_users(row):
        """Collect normalized list of distinct users for a given material row."""
        collected = []
        for user_col in user_cols:
            val = row[user_col]
            if pd.isna(val):
                continue
            tokens = [tok.strip().upper() for tok in str(val).split(",") if tok.strip()]
            for token in tokens:
                if token not in collected:
                    collected.append(token)
        return collected

    # Build per-row list of distinct users derived from split columns.
    user_lists = split_users_df[user_cols].apply(extract_users, axis=1)
    # All unique indicator labels derived from the lookup, plus UNK as default.
    indicator_users = sorted({val for val in lookup_map.values() if val}) or ["UNK"]
    if "UNK" not in indicator_users:
        indicator_users.append("UNK")

    # Expand list membership into indicator columns.
    for user_label in indicator_users:
        df[user_label] = user_lists.apply(lambda users, label=user_label: int(label in users))

    # Ensure Movement column exists, defaulting to NA; apply business override.
    if "Movement" not in df.columns:
        df["Movement"] = pd.NA
    df.loc[df['MRP Controller'] == 899, 'Movement'] = 'L'

    # Keep only the columns we expect and ensure they exist.
    keep_cols = ["Material", "Priority", "Movement"] + indicator_users
    for col in keep_cols:
        if col not in df.columns:
            df[col] = 0 if col in indicator_users else pd.NA
    df = df[keep_cols]
    out_schema = load_schema("mc46_output")
    df = enforce_output_schema(df, out_schema)

    return df
