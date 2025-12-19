import pandas as pd
import numpy as np
import math
import itertools
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path

from .retrieve import *

def _get_con():
    from pkg.db import get_con as _gc
    return _gc(read_only=True)

BASE_DIR = Path(__file__).resolve().parent
DECISION_VARS_CSV_PATH = BASE_DIR /  "non_zero_decision_variables.xlsx"
PIECEWISE_PARAMS_PATH = BASE_DIR / "piecewiseparams.json"

# --------- Get model input parameters and output --------------

opti_df = pd.read_excel(DECISION_VARS_CSV_PATH)

i_loc = get_i_loc()
i_locpriority = get_i_locpriority()
i_sku = get_i_sku()
i_sku_type = get_i_sku_type()
i_type = get_i_type()
i_type_loc = get_i_type_loc()
i_user_loc = get_i_loc_user()
i_sku_user = get_i_sku_user()


all_params = {'iloc': i_loc,
              'ilocpriority': i_locpriority,
              'isku': i_sku,
              'iskutype': i_sku_type,
              'itype': i_type,
              'itypeloc': i_type_loc,
              'i_user_loc': i_user_loc,
              'i_sku_user': i_sku_user
              }

Params = {k: v for outer in all_params.values() for k, v in outer.items()}

system_params = get_system()
Params['Linedowncost'] = system_params['Linedowncost'].iloc[0]
Params['Budget'] = system_params['Budget'].iloc[0]
Params['SpaceFactor'] = system_params['SpaceFactor'].iloc[0]
Params['reservecost'] = system_params['reservecost'].iloc[0]


if not PIECEWISE_PARAMS_PATH.exists():
    raise FileNotFoundError(f"Piecewise params not found at {PIECEWISE_PARAMS_PATH}")

with PIECEWISE_PARAMS_PATH.open("r") as f:
    piecewiseparams = json.load(f)

def strings_to_tuples(d):
    return {eval(k): v for k, v in d.items()}

piecewiseparams = {
    "breakpoints": strings_to_tuples(piecewiseparams["breakpoints"]),
    "pieceval": strings_to_tuples(piecewiseparams["pieceval"]),
    "resbreakpoints": strings_to_tuples(piecewiseparams["resbreakpoints"]),
    "respieceval": strings_to_tuples(piecewiseparams["respieceval"]),
}
Params.update(piecewiseparams)

def fetch_parts():
    con = _get_con()

    try:
        parts_df = con.execute(
            """
            SELECT * FROM parts
            """
        ).df()
    except Exception as exc:
        raise RuntimeError(
            "Failed to read parts from database. Ensure schema is initialized."
        ) from exc
    
    parts_df = parts_df.drop(["size_id", "updated_at"], axis=1)
    return parts_df

def fetch_location_storage_type():
    con = _get_con()

    try:
        location_storage_type = con.execute(
            """
            SELECT lst.location_id,
                   sl.location,
                   lst.type_id,
                   st.type,
                   lst.code,
                   lst.type_current_units
            FROM location_storage_type lst
            LEFT JOIN storage_location sl ON lst.location_id = sl.location_id
            LEFT JOIN storage_type st ON lst.type_id = st.type_id
            ORDER BY lst.location_id, lst.type_id
            """
        ).df()
    except Exception as exc:
        raise RuntimeError(
            "Failed to read location_storage_type from database. Ensure schema is initialized."
        ) from exc
    
    return location_storage_type

def fetch_loc():
    con = _get_con()

    try:
        location_df = con.execute(
            """
            SELECT location_id,
                   location,
                   floor_space,
                   current_storage_floor_space,
                   travel_time
            FROM storage_location
            """
        ).df()
    except Exception as exc:
        raise RuntimeError(
            "Failed to read storage_location from database. Ensure schema is initialized."
        ) from exc
    
    return location_df

def fetch_min_vol() -> pd.DataFrame:
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

def prep_parts():
    parts_df = fetch_parts()
    loc_storage_type = fetch_location_storage_type()

    minvol_df = fetch_min_vol()
    parts_df = parts_df.merge(minvol_df)
    total_cuft_df = fetch_total_cuft()
    parts_df = parts_df.merge(total_cuft_df)


    parts_df = parts_df.rename(columns={"storage_type": "code"})
    loc_storage_type = loc_storage_type.drop(["location_id", "type_id"], axis=1)

    parts_df['SKU_cubic_ft'] = np.maximum(parts_df["min_vol"], parts_df["total_stock"] * parts_df["total_cuft"])

    parts_df = parts_df.merge(loc_storage_type, how="left", on="code")
    parts_df['category'] = parts_df.get('priority', "").fillna("") + parts_df.get('movement', "").fillna("") + parts_df.get('size', "").fillna("")

    return parts_df

# ---------- Prepare for visualizations

def _format_percent(series: pd.Series) -> pd.Series:
    """Round percentages and keep small non-zero values visible."""
    return series.replace(0, None).apply(
        lambda x: 0.01 if x is not None and 0 < x < 0.01 else round(x, 2) if x is not None else None
    )



def build_store_units():
    cstore_df = pd.DataFrame(
        [(k[0], k[1], v) for k, v in Params["cStoreUnits"].items()],
        columns=["Index_1", "Index_2", "cStoreUnits"]
    ).dropna()

    results_df = opti_df[(opti_df["Variable"] == "StoreUnits")]
    
    buy_units_df = opti_df[(opti_df["Variable"] == "BuyUnits")]
    buy_units_df = buy_units_df.drop(["Index_3", "Index_4"], axis = 1)
    buy_units_df = buy_units_df.rename(columns = {"Index_1" : "Storage Type", 
                                                  "Index_2" : "Location", 
                                                  "Value" : "Number of Units"
                                                })

    reloc_units_df = opti_df[(opti_df["Variable"] == "RelocUnits")]
    reloc_units_df = reloc_units_df.drop(["Index_4"], axis = 1)
    reloc_units_df = reloc_units_df.rename(columns = {"Index_1" : "Storage Type",
                                                      "Index_2" : "Current Location",
                                                      "Index_3" : "Recommended Location",
                                                      "Value" : "Number of Units"
                                                    })

    units_change_df = results_df.merge(
                cstore_df,
                on=["Index_1", "Index_2"],
                how="left"
            ).fillna(0).infer_objects(copy=False)
    units_change_df = units_change_df.drop(["Index_3", "Index_4"], axis = 1)
    units_change_df = units_change_df.rename(columns = {"Index_1" : "Storage Type", 
                                                        "Index_2" : "Location", 
                                                        "cStoreUnits" : "Current # of Units",
                                                        "Value" : "Recommended # of Units"
                                                        })
    units_change_df = units_change_df.drop(["Variable"], axis=1)

    if buy_units_df.empty:
        return units_change_df, reloc_units_df

    return units_change_df, buy_units_df, reloc_units_df

def build_SKUalloc():
    unique_df = opti_df[opti_df["Variable"] == "SKUuniquePct"].copy()
    forward_cuft_rec_df = opti_df[opti_df["Variable"] == "SKUcubicPct"].copy()
    res_unique_df = opti_df[opti_df["Variable"] == "ResSKUuniquePct"].copy()
    res_cuft_req_df = opti_df[opti_df["Variable"] == "ResSKUcubicPct"].copy()
    parts_df = prep_parts()

    # ------------ SKU Category Forward Allocation in Count ---------------

    unique_df = unique_df.drop(["Variable", "Index_4"], axis = 1)
    unique_df = unique_df.rename(columns = {"Index_1": "Category",
                                            "Index_2": "Storage Type",
                                            "Index_3": "Location",
                                            "Value": "Recommended % Forward"
                                            })
    
    cat_counts = (
        parts_df.groupby("category")["material"]
        .nunique()
        .reset_index()
        .rename(columns={"category": "Category", "material": "Total Category Count"})
    )

    counts = (
        parts_df.groupby(["category", "type", "location"])["material"]
        .nunique()
        .reset_index()
        .rename(columns={"category": "Category", "type": "Storage Type", "location": "Location", "material": "Current Count"})
    )
    counts = counts.merge(cat_counts, on="Category", how="left")
    counts["Current % of Category"] = counts["Current Count"] / counts["Total Category Count"]
    current_df = counts

    unique_SKU = current_df.merge(unique_df, how="outer", on=["Category", "Storage Type", "Location"], suffixes=("_curr", "_rec")).fillna(0)
    # Always derive total per category from cat_counts to avoid zero carry-over after outer merge
    cat_total_map = dict(zip(cat_counts["Category"], cat_counts["Total Category Count"]))
    unique_SKU["Total Category Count"] = unique_SKU["Category"].map(cat_total_map).fillna(0)
    unique_SKU["Recommended Count"] = unique_SKU["Recommended % Forward"] * unique_SKU["Total Category Count"]


    # ------------ SKU Category Reserve Allocation in Count ---------------

    res_unique_df = res_unique_df.drop(["Variable", "Index_4"], axis = 1)
    res_unique_df = res_unique_df.rename(columns = {"Index_1": "Category",
                                            "Index_2": "Storage Type",
                                            "Index_3": "Location",
                                            "Value": "Recommended % Reserve"
                                            })
    res_unique_SKU = res_unique_df.merge(cat_counts, on="Category", how="left")
    # Ensure Total Category Count populated even if merge produced zeros
    res_unique_SKU["Total Category Count"] = res_unique_SKU["Category"].map(dict(zip(cat_counts["Category"], cat_counts["Total Category Count"]))).fillna(0)
    res_unique_SKU["Recommended Reserve Count"] = res_unique_SKU["Recommended % Reserve"] * res_unique_SKU["Total Category Count"]

    # ------------ SKU Category Forward Allocation in Cubic Feet ---------------
    forward_cuft_rec_df = forward_cuft_rec_df.drop(["Variable", "Index_4"], axis = 1)
    forward_cuft_rec_df = forward_cuft_rec_df.rename(columns = {"Index_1": "Category",
                                            "Index_2": "Storage Type",
                                            "Index_3": "Location",
                                            "Value": "Recommended % Cubic Feet"
                                            })
    
    cat_cubic = (
        parts_df.groupby("category")["SKU_cubic_ft"]
        .sum()
        .reset_index()
        .rename(columns={"category": "Category", "SKU_cubic_ft": "Total Cubic Feet"})
    )
    cubic_alloc = (
        parts_df.groupby(["category", "type", "location"])["SKU_cubic_ft"]
        .sum()
        .reset_index()
        .rename(columns={"category": "Category", "type": "Storage Type", "location": "Location", "SKU_cubic_ft": "Current Cubic Feet"})
    )

    cubic_alloc = cubic_alloc.merge(cat_cubic, on="Category", how="left")
    cubic_alloc["Current % Cubic Feet"] = cubic_alloc["Current Cubic Feet"] / cubic_alloc["Total Cubic Feet"]
    current_cubic_df = cubic_alloc

    cubic_SKU = current_cubic_df.merge(forward_cuft_rec_df, how="outer", on=["Category", "Storage Type", "Location"], suffixes=("_curr", "_rec")).fillna(0)
    cubic_SKU["Total Cubic Feet"] = cubic_SKU["Category"].map(dict(zip(cat_cubic["Category"], cat_cubic["Total Cubic Feet"]))).fillna(0)
    cubic_SKU["Recommended Cubic Feet"] = cubic_SKU["Recommended % Cubic Feet"] * cubic_SKU["Total Cubic Feet"]
    cubic_SKU = cubic_SKU.drop(columns=[c for c in ["Total Cubic Feet_curr", "Total Cubic Feet_rec"] if c in cubic_SKU.columns])

    # ------------ SKU Category Reserve Allocation in Cubic Feet ---------------
    res_cubic = res_cuft_req_df.drop(["Variable", "Index_4"], axis = 1)
    res_cubic = res_cubic.rename(columns = {"Index_1": "Category",
                                            "Index_2": "Storage Type",
                                            "Index_3": "Location",
                                            "Value": "Recommended % Reserve Cubic Feet"
                                            })
    res_cubic = res_cubic.merge(cat_cubic, on="Category", how="left")
    res_cubic["Total Cubic Feet"] = res_cubic["Category"].map(dict(zip(cat_cubic["Category"], cat_cubic["Total Cubic Feet"]))).fillna(0)
    res_cubic["Recommended Reserve Cubic Feet"] = res_cubic["Recommended % Reserve Cubic Feet"] * res_cubic["Total Cubic Feet"]

    return unique_SKU, cubic_SKU, res_unique_SKU, res_cubic
    
def build_map_tables():
    map_df = pd.DataFrame([
        {"Location": "LC02", "lat": 34.88929510572189, "lon": -82.1894677460074},
        {"Location": "WH01", "lat": 34.905184795550646, "lon": -82.19200859173132},
        {"Location": "BS30", "lat": 34.89113877335713, "lon": -82.18009074333585},
        {"Location": "BS32", "lat": 34.89450479012038, "lon": -82.19086249468852},
        {"Location": "WH50", "lat": 34.89337399987217, "lon": -82.18114216926472}
    ])

    parts_df = prep_parts()
    curr = (
        parts_df.groupby(["location", "category"])
        .agg({"line_down_orders": "sum", "material": "nunique", "SKU_cubic_ft": "sum"})
        .reset_index()
        .rename(
            columns={
                "category": "Category",
                "type": "Storage Type",
                "location": "Location",
                "material": "Material Count",
                "SKU_cubic_ft": "Total Cubic Feet",
                "line_down_orders": "Total Line Down Orders",
            }
        )
    )
    # Current storage units by location from location_storage_type
    storage_units_raw = fetch_location_storage_type()
    storage_units = (
        storage_units_raw.pivot_table(
            index="location",
            columns="type",
            values="type_current_units",
            aggfunc="sum",
            fill_value=0,
        )
        .rename(
            columns={
                "CABINET": "Current Cabinets",
                "SHELF": "Current Shelves",
                "REMSTAR": "Current Modulas",
                "RACK": "Current Racks",
                "MEZZANINE": "Current Mezzanine",
            }
        )
        .reset_index()
        .rename(columns={"location": "Location"})
    )
    totals = (
        storage_units_raw.groupby("location", as_index=False)["type_current_units"]
        .sum()
        .rename(columns={"location": "Location", "type_current_units": "Current Storage Units"})
    )
    storage_units = storage_units.merge(totals, how="left", on="Location")

    # Travel time per location
    loc_time = (
        fetch_loc()[["location", "travel_time"]]
        .rename(columns={"location": "Location", "travel_time": "Travel Time"})
    )
    # Aggregate to location-level totals for tooltip support
    loc_totals = (
        curr.groupby("Location")
        .agg({"Material Count": "sum", "Total Cubic Feet": "sum", "Total Line Down Orders": "sum"})
        .rename(
            columns={
                "Material Count": "Location Total Materials",
                "Total Cubic Feet": "Location Total Cubic Feet",
                "Total Line Down Orders": "Location Total Line Down Orders",
            }
        )
        .reset_index()
    )
    curr_map_df = map_df.copy()
    curr_map_df = curr_map_df.merge(curr, how="right", on=["Location"])
    curr_map_df = curr_map_df.merge(loc_totals, how="left", on="Location")
    curr_map_df = curr_map_df.merge(storage_units, how="left", on="Location")
    curr_map_df = curr_map_df.merge(loc_time, how="left", on="Location")

    totals = (
        parts_df.groupby(["category"])
        .agg({"line_down_orders": "sum", "material": "nunique", "SKU_cubic_ft": "sum"})
        .reset_index()
        .rename(
            columns={
                "category": "Category",
                "material": "Total Materials",
                "SKU_cubic_ft": "Total Cubic Feet",
            }
        )
    )
    line_down_tot = opti_df[opti_df["Variable"] == "LineDownOrders"].copy()
    line_down_tot = line_down_tot.drop(["Variable", "Index_4"], axis=1)
    line_down_tot = line_down_tot.rename(columns= {"Index_1": "Category", "Index_2": "User", "Index_3": "Location Priority", "Value": "Line Down Orders"})
    line_down_tot = line_down_tot.groupby(["Category"]).agg({"Line Down Orders": "first"}).reset_index()
    totals = totals.merge(line_down_tot, how="left", on=["Category"])

    forward_cubic = opti_df[opti_df["Variable"] == "SKUcubicPct"].copy()
    forward_cubic = forward_cubic.drop(["Variable", "Index_4"], axis = 1)
    forward_cubic = forward_cubic.rename(columns = {"Index_1": "Category",
                                            "Index_2": "Storage Type",
                                            "Index_3": "Location",
                                            "Value": "% Forward Cubic Feet"
                                            })
    forward_cubic = (
        forward_cubic.groupby(["Location", "Category"])
        .agg({"% Forward Cubic Feet": "sum"})
        .reset_index()
    )
    forward_cubic = forward_cubic.merge(totals[["Category", "Total Cubic Feet"]], how="left", on=["Category"])
    forward_cubic["Cubic Feet"] = forward_cubic["% Forward Cubic Feet"] * forward_cubic["Total Cubic Feet"]
    forward_cubic = forward_cubic.drop(["Total Cubic Feet"], axis=1)
    
    forward_unique = opti_df[opti_df["Variable"] == "SKUuniquePct"].copy()
    forward_unique = forward_unique.drop(["Variable", "Index_4"], axis = 1)
    forward_unique = forward_unique.rename(columns = {"Index_1": "Category",
                                            "Index_2": "Storage Type",
                                            "Index_3": "Location",
                                            "Value": "% Forward Materials"
                                            })
    forward_unique = (
        forward_unique.groupby(["Location", "Category"])
        .agg({"% Forward Materials": "sum"})
        .reset_index()
    )
    forward_unique = forward_unique.merge(totals[["Category", "Total Materials", "Line Down Orders"]], how="left", on=["Category"])
    forward_unique["Materials"] = forward_unique["% Forward Materials"] * forward_unique["Total Materials"]
    forward_unique["Approx Line Down Order"] = forward_unique["% Forward Materials"] * forward_unique["Line Down Orders"]
    forward_unique = forward_unique.drop(["Total Materials", "Line Down Orders"], axis=1)

    forward_storage_raw = opti_df[(opti_df["Variable"] == "StoreUnits")].copy()
    forward_storage_raw = forward_storage_raw.drop(["Variable", "Index_3", "Index_4"], axis=1)
    forward_storage_raw = forward_storage_raw.rename(columns = {"Index_1": "Storage Type",
                                                        "Index_2": "Location",
                                                        "Value": "Recommended Storage Units"
                                                        })
    forward_storage = (
        forward_storage_raw.pivot_table(
            index="Location",
            columns="Storage Type",
            values="Recommended Storage Units",
            aggfunc="sum",
            fill_value=0,
        )
        .rename(
            columns={
                "CABINET": "Current Cabinets",
                "SHELF": "Current Shelves",
                "REMSTAR": "Current Modulas",
                "RACK": "Current Racks",
                "MEZZANINE": "Current Mezzanine",
            }
        )
        .reset_index()
    )
    
    alloc_loc_totals = (
        forward_unique.groupby("Location")
        .agg({"Materials": "sum", "Approx Line Down Order": "sum"})
        .rename(columns={"Materials": "Alloc Total Materials", "Approx Line Down Order": "Alloc Total Line Down Orders"})
        .reset_index()
    )
    alloc_cuft_totals = (
        forward_cubic.groupby("Location")
        .agg({"Cubic Feet": "sum"})
        .rename(columns={"Cubic Feet": "Alloc Total Cubic Feet"})
        .reset_index()
    )
    alloc_loc_totals = alloc_loc_totals.merge(alloc_cuft_totals, how="left", on="Location")

    # Build allocation map with a base that includes Location + Category so merges keep Category
    alloc_base = forward_cubic[["Location", "Category"]].drop_duplicates()
    alloc_map_df = (
        alloc_base
        .merge(map_df, how="left", on="Location")  # bring lat/lon
        .merge(forward_cubic, how="left", on=["Location", "Category"])
        .merge(forward_unique, how="left", on=["Location", "Category"])
        .merge(forward_storage, how="left", on="Location")
        .merge(alloc_loc_totals, how="left", on="Location")
        .merge(loc_time, how="left", on="Location")
    )
    # Clean up duplicate category columns if they appear
    # if "Category_x" in alloc_map_df.columns and "Category" not in alloc_map_df.columns:
    #     alloc_map_df["Category"] = alloc_map_df["Category_x"].fillna(alloc_map_df.get("Category_y"))
    # for col in ("Category_x", "Category_y"):
    #     if col in alloc_map_df.columns:
    #         alloc_map_df = alloc_map_df.drop(columns=[col])


    return curr_map_df, alloc_map_df
    
def get_cost():
    obj = opti_df[opti_df["Variable"] == "objective value"]
    return obj[["Value"]]