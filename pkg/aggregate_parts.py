import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import json
from typing import List, Dict

from .validate import validate, ValidationError, load_schema, enforce_output_schema


def _get_con():
    # Lazy import to avoid top-level import cycle with db module.
    from .db import get_con as _gc
    return _gc()


def fetch_parts() -> pd.DataFrame:
    # Pull the full parts table from DuckDB.
    con = _get_con()
    try:
        df_parts = con.execute(
            """
            SELECT *
            FROM parts
            """
        ).df()
    except Exception as exc:
        raise RuntimeError(
            "Failed to read parts from DuckDB. Ensure schema is initialized."
        ) from exc
    return df_parts

def fetch_min_vol() -> pd.DataFrame:
    # Fetch minimum volume per size, needed for SKU cube feet calculation.
    con = _get_con()
    try:
        size_to_vol = con.execute(
            """
            SELECT size, min_vol
            FROM size
            """
        ).df()
    except Exception as exc:
        raise RuntimeError(
            "Failed to read size from DuckDB. Ensure schema is initialized."
        ) from exc
    return size_to_vol

def fetch_total_cuft() -> pd.DataFrame:
    # Fetch cubic feet per unit per size, also for volume math.
    con = _get_con()
    try:
        size_to_tot_cu_ft = con.execute(
            """
            SELECT size, total_cuft
            FROM size
            """
        ).df()
    except Exception as exc:
        raise RuntimeError(
            "Failed to read size from DuckDB. Ensure schema is initialized."
        ) from exc
    return size_to_tot_cu_ft


def fetch_tech_map():
    con = _get_con()
    return con.execute("SELECT value, tech_name FROM tech_map ORDER BY value").fetchdf()


def fetch_storage_types() -> pd.DataFrame:
    # Retrieve storage types with capacity info for later compatibility joins.
    con = _get_con()
    return con.execute(
        """
        SELECT type_id, type, cubic_capacity_per_unit
        FROM storage_type
        """
    ).df()


def fetch_type_size_compat() -> pd.DataFrame:
    # Pull compatibility matrix so we can compute max SKUs per storage type/size.
    con = _get_con()
    return con.execute(
        """
        SELECT type_id, size_id, max_sku_per_unit
        FROM type_size_compat
        """
    ).df()

# def fetch_tech():
#     con = get_con()
#     return con.execute("SELECT tech_name FROM tech").fetchdf()

def prep_parts() -> pd.DataFrame:
    # Start from the raw parts table.
    parts_df = fetch_parts()

    # Enrich with min_vol and total_cuft dimensional attributes.
    minvol_df = fetch_min_vol()
    parts_df = parts_df.merge(minvol_df)
    total_cuft_df = fetch_total_cuft()
    parts_df = parts_df.merge(total_cuft_df)

    if "material" not in parts_df.columns:
        raise ValueError("parts table is missing required column 'material'.")
    
    # Compute cubic feet requirement per SKU using business rule max(min_vol, stock * cuft).
    parts_df['SKU_cubic_ft'] = np.maximum(parts_df["min_vol"], parts_df["total_stock"] * parts_df["total_cuft"])

    # Avoid division by zero; keep NaN where num_users is zero/NaN
    denom = parts_df["num_users"].replace(0, np.nan)
    #splitting orders across users that use the material
    parts_df['orders_per_user'] = parts_df['orders'] / denom
    parts_df['ld_orders_per_user'] = np.ceil(parts_df['line_down_orders'] / denom * 0.5833)
    parts_df[['orders_per_user', 'ld_orders_per_user']] = parts_df[['orders_per_user', 'ld_orders_per_user']].fillna(0)

    parts_df['category'] = parts_df.get('priority', "").fillna("") + parts_df.get('movement', "").fillna("") + parts_df.get('size', "").fillna("")

    return parts_df

def build_i_sku() -> pd.DataFrame:
    # Prepare base parts data with derived metrics.
    parts_df = prep_parts()

    # Aggregate by category to compute counts and sums required by i_sku.
    i_sku = parts_df.groupby('category').agg(
        {'priority':'first',
         'movement':'first',
         'size':'first',
         'size_id':'first',
         'material': 'nunique',
         'total_stock': 'sum',
         'orders': 'sum',
         'line_down_orders':'sum',
         'min_vol' : 'first',
         'SKU_cubic_ft' : 'sum'
        }).reset_index().rename(columns = {'material':'numSKU'})

    # minVol_df = fetch_min_vol()
    # i_sku = i_sku.merge(minVol_df, how="left", on="size_id")

    # total_cuft_df = fetch_total_cuft()
    # i_sku = i_sku.merge(total_cuft_df, how="left", on="size_id")
    # i_sku["SKU_cubic_ft"] = i_sku['total_stock'] * i_sku['total_cuft']
    # i_sku = i_sku.drop(['total_cuft'], axis=1)

    i_sku = i_sku.rename(columns={'orders': 'total_orders'})

    # Assign category_id deterministically from the grouped order
    i_sku = i_sku.reset_index(drop=True)
    i_sku.insert(0, "category_id", i_sku.index + 1)

    # Enforce outbound schema to guarantee required columns and ordering.
    i_sku_schema = load_schema("i_sku")
    i_sku_out = enforce_output_schema(i_sku, i_sku_schema)

    return i_sku_out

def build_i_sku_user() -> pd.DataFrame:
    # Needed for tech lookup when constructing melted view.
    con = _get_con()
    # Start from enriched parts data.
    parts_df = prep_parts()

    # Build category lookup from i_sku
    i_sku_df = build_i_sku()
    cat_lookup = {row.category: int(row.category_id) for _, row in i_sku_df.iterrows()}

    # Tech lookup from tech table (indicator columns use uppercased names)
    tech_df = con.execute("SELECT tech_id, tech_name FROM tech").df()
    tech_lookup = {row.tech_name.upper(): int(row.tech_id) for _, row in tech_df.iterrows()}

    indicator_cols = [col for col in parts_df.columns if col.isupper() and col != "MATERIAL"]

    if not indicator_cols:
        return pd.DataFrame(columns=load_schema("i_sku_user")["required_cols"])

    # Melt indicator columns to long form to compute per-tech aggregations.
    long_df = parts_df.melt(
        id_vars=[
            "category",
            "priority",
            "movement",
            "size",
            "size_id",
            "material",
            "total_stock",
            "orders_per_user",
            "ld_orders_per_user",
        ],
        value_vars=indicator_cols,
        var_name="tech_name",
        value_name="tech_flag",
    )

    # Keep only rows where the tech is present
    # long_df = long_df[long_df["tech_flag"] > 0]

    # Attach IDs and drop rows without valid FKs
    long_df["category_id"] = long_df["category"].map(cat_lookup)
    long_df["tech_id"] = long_df["tech_name"].map(lambda x: tech_lookup.get(str(x).upper()))
    long_df = long_df.dropna(subset=["category_id", "tech_id"])

    long_df

    # Cast indicator to int and weight metrics by tech presence.
    long_df['tech_flag'] = long_df['tech_flag'].astype(int)

    long_df["total_stock"] = long_df["tech_flag"] * long_df["total_stock"]
    long_df["orders_per_user"] = long_df["tech_flag"] * long_df["orders_per_user"]
    long_df["ld_orders_per_user"] = long_df["tech_flag"] * long_df["ld_orders_per_user"]

    long_df.to_csv("sku_user.csv")

    # Aggregate back to category x tech level.
    i_sku_user = (
        long_df.groupby(["category_id", "tech_id"], dropna=False)
        .agg(
            category = ("category", "first"),
            tech_name = ("tech_name", "first"),
            numSKU=("tech_flag", "sum"),
            ld_orders_per_user=("ld_orders_per_user", "sum"),
            orders_per_user=("orders_per_user", "sum"),
        )
        .reset_index()
    )


    i_sku_user_schema = load_schema("i_sku_user")
    i_sku_user_out = enforce_output_schema(i_sku_user, i_sku_user_schema)

    return i_sku_user_out

def build_i_sku_type() -> pd.DataFrame:
    # Build category info from i_sku
    i_sku_df = build_i_sku()
    if i_sku_df.empty:
        return pd.DataFrame(columns=load_schema("i_sku_type")["required_cols"])

    # Fetch storage types
    types_df = fetch_storage_types()
    if types_df.empty:
        return pd.DataFrame(columns=load_schema("i_sku_type")["required_cols"])

    # Fetch compat table
    compat_df = fetch_type_size_compat()
    # Build a dictionary for O(1) lookup of compatibility caps by (type_id, size_id).
    compat_lookup = {
        (int(row.type_id), int(row.size_id)): float(row.max_sku_per_unit)
        for _, row in compat_df.iterrows()
    }

    # Cross join categories x types
    i_sku_df["key"] = 1
    types_df["key"] = 1
    combo = i_sku_df.merge(types_df, on="key").drop(columns=["key"])

    def max_sku(row):
        # Return max_sku_per_unit when the combination appears in the compat map.
        return compat_lookup.get((int(row.type_id), int(row.size_id)))

    combo["max_sku_per_unit"] = combo.apply(max_sku, axis=1)
    combo["max_sku_per_unit"] = combo["max_sku_per_unit"].fillna(0)
    # compatibility flag is simply presence of a non-null max_sku_per_unit.
    combo["compatible"] = combo["max_sku_per_unit"].notna().astype(int)

    # Apply penalty rule for S-size items in type_id 4 (Remstar).
    combo["penalty"] = ((combo["size_id"] == 1) & (combo["type_id"] == 4)).astype(int)

    # Select and order columns per schema expectations.
    i_sku_type = combo[
        [
            "category_id",
            "category",
            "type_id",
            "type",
            "size_id",
            "cubic_capacity_per_unit",
            "max_sku_per_unit",
            "compatible",
            "penalty",
        ]
    ]

    i_sku_type_schema = load_schema("i_sku_type")
    i_sku_type_out = enforce_output_schema(i_sku_type, i_sku_type_schema)

    return i_sku_type_out
