from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict
import itertools

import pandas as pd

def _get_con():
    from pkg.db import get_con as _gc
    return _gc(read_only=True)


def _bool_cols_to_int(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Convert boolean-like columns to 0/1 integers, filling missing as 0."""
    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype("boolean").fillna(False).astype(int)
    return df

def get_i_loc() -> dict:
    con = _get_con()
    try:
        loc_df = con.execute(
            """
            SELECT location AS loc, floor_space AS FloorSpace, reserve_allowed AS ReserveAllowed, location_priority AS locpriority, current_storage_floor_space AS CurrentStorageFloorSpace
            FROM storage_location
            """
        ).df()
    except Exception as exc:
        raise RuntimeError(
            "Failed to read location from database. Ensure schema is initialized."
        ) from exc
    
    loc_df = _bool_cols_to_int(loc_df, ["ReserveAllowed"])
    loc_dict = loc_df.set_index(['loc']).to_dict(orient='dict')
    return loc_dict

def get_i_loc_user() -> dict:
    con = _get_con()
    try:
        loc_user_df = con.execute(
            """
            SELECT
              t.tech_name AS User,
              sl.location AS Loc,
              CASE WHEN tlu.tech_id IS NULL THEN 0 ELSE 1 END AS UserLocationCompat
            FROM tech t
            CROSS JOIN storage_location sl
            LEFT JOIN tech_location_usage tlu
              ON tlu.tech_id = t.tech_id AND tlu.location_id = sl.location_id
            """
        ).df()
    except Exception as exc:
        raise RuntimeError(
            "Failed to read tech_location_usage from database. Ensure schema is initialized."
        ) from exc
    
    # tech_map_df = con.execute("SELECT tech_name, value FROM tech_map").df()
    # tech_map_df.groupby('tech_name').agg({'value': 'first'})
    # user_map = {str(row.tech_name).upper(): str(row.value) for _, row in tech_map_df.iterrows()}
    # loc_user_df["User"] = loc_user_df["User"].apply(
    #     lambda u: user_map.get(str(u).upper(), u)
    # )
    
    loc_user_dict = loc_user_df.set_index(["Loc", "User"]).to_dict(orient="dict")
    return loc_user_dict


def get_i_locpriority() -> dict:
    con = _get_con()
    try:
        locprio_df = con.execute(
            """
            SELECT location_priority AS locpriority, travel_time AS traveltime
            FROM location_priority
            """
        ).df()
    except Exception as exc:
        raise RuntimeError(
            "Failed to read location priorities from database. Ensure schema is initialized."
        ) from exc
    
    locprio_dict = locprio_df.set_index(['locpriority']).to_dict(orient='dict')
    return locprio_dict

def get_i_sku() -> dict:
    con = _get_con()
    try:
        i_sku_df = con.execute(
            """
            SELECT category AS sku, numSKU AS NumSKU, total_stock AS "Total Stock", total_orders AS "TotalOrders", line_down_orders AS "LineDownOrders", min_vol AS "minVol", SKU_cubic_ft AS SKUcubicft
            FROM i_sku
            """
        ).df()
    except Exception as exc:
        raise RuntimeError(
            "Failed to read i_sku from database. Ensure schema is initialized."
        ) from exc
    
    i_sku_dict = i_sku_df.set_index(['sku']).to_dict(orient='dict')
    return i_sku_dict
    
def get_i_sku_type() -> dict:
    con = _get_con()
    try:
        i_sku_type_df = con.execute(
            """
            SELECT category AS sku, type, cubic_capacity_per_unit AS "cubiccapperunit", max_sku_per_unit AS "maxskusperunit", compatible, penalty
            FROM i_sku_type
            """
        ).df()
    except Exception as exc:
        raise RuntimeError(
            "Failed to read i_sku_type from database. Ensure schema is initialized."
        ) from exc
    
    i_sku_type_df = _bool_cols_to_int(i_sku_type_df, ["compatible", "penalty"])
    i_sku_type_dict = i_sku_type_df.set_index(['sku', 'type']).to_dict(orient='dict')
    return i_sku_type_dict

def get_i_sku_user() -> dict:
    con = _get_con()
    try:
        i_sku_user_df = con.execute(
            """
            SELECT category AS sku, tech_name AS user, numSKU AS "NumSKUUser", ld_orders_per_user AS "numLDOrdersPerUser", orders_per_user AS "numOrdersPerUser"
            FROM i_sku_user
            """
        ).df()
    except Exception as exc:
        raise RuntimeError(
            "Failed to read i_sku_user from database. Ensure schema is initialized."
        ) from exc
    
    # tech_map_df = con.execute("SELECT tech_name, value FROM tech_map").df()
    # tech_map_df.groupby('tech_name').agg({'value': 'first'})
    # user_map = {str(row.tech_name).upper(): str(row.value) for _, row in tech_map_df.iterrows()}
    # i_sku_user_df["user"] = i_sku_user_df["user"].apply(
    #     lambda u: user_map.get(str(u).upper(), u)
    # )
    
    i_sku_user_dict = i_sku_user_df.set_index(['sku', 'user']).to_dict(orient='dict')
    return i_sku_user_dict

def get_i_type() -> dict:
    con = _get_con()
    try:
        i_type_df = con.execute(
            """
            SELECT type, sqft_req AS SpaceReq, buy_cost AS BuyExpense, buy_invest AS BuyInvest, reloc_cost AS "RelocExpense", reloc_invest AS "RelocInvest"
            FROM storage_type
            """
        ).df()
    except Exception as exc:
        raise RuntimeError(
            "Failed to read storage_type from database. Ensure schema is initialized."
        ) from exc
    
    # type_map = {"CABINET": "Cab", "MEZZANINE": "Mez", "SHELF"}
    
    i_type_dict = i_type_df.set_index(['type']).to_dict(orient='dict')
    return i_type_dict

def get_i_type_loc() -> dict:
    con = _get_con()
    try:
        df = con.execute(
            """
            SELECT
              st.type,
              sl.location AS loc,
              COALESCE(lst.type_current_units, 0) AS cStoreUnits,
              1 AS VertSpace
            FROM storage_type st
            CROSS JOIN storage_location sl
            LEFT JOIN location_storage_type lst
              ON lst.type_id = st.type_id AND lst.location_id = sl.location_id
            """
        ).df()
    except Exception as exc:
        raise RuntimeError(
            "Failed to read storage_type/location data from database. Ensure schema is initialized."
        ) from exc

    return df.set_index(["type", "loc"]).to_dict(orient="dict")

def get_system() -> dict:
    con = _get_con()
    try:
        sys_df = con.execute(
            """
            SELECT line_down_cost AS Linedowncost, plan_budget AS Budget, space_factor AS SpaceFactor, reserve_cost AS reservecost
            FROM system
            """
        ).df()
    except Exception as exc:
        raise RuntimeError(
            "Failed to read system data from database. Ensure schema is initialized."
        ) from exc
    
    return sys_df
    


def build_sets_from_db() -> Dict[str, list]:
    """
    Build model sets from current database contents.
    Returns a dict with keys: priority, movement, sizes, SKUS, USERS, LOCS, TYPES,
    PRIORITIES, BREAKS, PRIORITYSET, secondary, not_reserve_users.
    """
    con = _get_con()

    # Priority/movement categories derived from parts
    parts_df = con.execute("SELECT priority, movement FROM parts").df()
    priority = sorted(parts_df["priority"].dropna().astype(str).unique().tolist()) or ["H", "M", "L"]
    movement = sorted(parts_df["movement"].dropna().astype(str).unique().tolist()) or ["H", "M", "L"]

    sizes_df = con.execute("SELECT size FROM size ORDER BY size_id").df()
    sizes = sizes_df["size"].dropna().astype(str).tolist() or ["S", "M", "L"]

    loc_df = con.execute(
        """
        SELECT location_id, location, location_priority, reserve_allowed
        FROM storage_location
        ORDER BY location_id
        """
    ).df()
    locs = loc_df["location"].dropna().astype(str).tolist()

    type_df = con.execute("SELECT type FROM storage_type ORDER BY type_id").df()
    types = type_df["type"].dropna().astype(str).tolist()

    priorities = sorted(
        loc_df["location_priority"].dropna().astype(int).unique().tolist()
    ) or [1]

    priorityset: Dict[int, list] = {}
    for p in priorities:
        locs_for_p = (
            loc_df.loc[loc_df["location_priority"] <= p, "location"]
            .dropna()
            .astype(str)
            .tolist()
        )
        priorityset[p] = locs_for_p

    skus = ["".join(x) for x in itertools.product(priority, movement, sizes) if "".join(x)]

    # tech_map_df = con.execute("SELECT value, tech_name FROM tech_map").df()
    # tech_map_df.groupby('tech_name').agg({'value': 'first'})
    # users = tech_map_df["value"].dropna().astype(str).tolist()
    # if not users:
    tech_df = con.execute("SELECT tech_name FROM tech").df()
    users = tech_df["tech_name"].dropna().astype(str).tolist()

    breaks = [0, 1, 2]
    secondary = (
        loc_df.loc[loc_df["reserve_allowed"] == True, "location"]
        .dropna()
        .astype(str)
        .tolist()
    )
    not_reserve_users = users  # adjust if different rule is needed

    return {
        "priority": priority,
        "movement": movement,
        "sizes": sizes,
        "SKUS": skus,
        "USERS": users,
        "LOCS": locs,
        "TYPES": types,
        "PRIORITIES": priorities,
        "BREAKS": breaks,
        "PRIORITYSET": priorityset,
        "secondary": secondary,
        "not_reserve_users": not_reserve_users
    }


