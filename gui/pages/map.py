import os
from pathlib import Path
from typing import Tuple

import pydeck as pdk
import streamlit as st
import pandas as pd

from optimization import alternate_dashboard

ROOT = Path(__file__).resolve().parents[2]


def _add_cat_dims(df):
    # Split the compact category code (e.g., "HMS") into individual attributes for filtering.
    out = df.copy()
    # Ensure a Series before accessing .str; fallback to empty string if column missing
    cat_series = out["Category"] if "Category" in out.columns else None
    if cat_series is None:
        cat_str = pd.Series([], dtype=str)
    else:
        cat_str = pd.Series(cat_series, dtype=str).fillna("")
    out["Priority"] = cat_str.str[0]
    out["Movement"] = cat_str.str[1]
    out["Size"] = cat_str.str[2]
    return out


def _apply_filters(df, pri_filter: str, mov_filter: str, size_filter: str):
    # Apply the three dropdown filters; "Any" leaves that dimension unfiltered.
    out = df
    if pri_filter != "Any":
        out = out[out["Priority"] == pri_filter]
    if mov_filter != "Any":
        out = out[out["Movement"] == mov_filter]
    if size_filter != "Any":
        out = out[out["Size"] == size_filter]
    return out


def _mapbox_token():
    # Prefer environment token; fall back to Streamlit secrets when available.
    env_token = os.getenv("MAPBOX_API_KEY")
    if env_token:
        return env_token
    try:
        return st.secrets.get("MAPBOX_API_KEY") if hasattr(st, "secrets") else None
    except Exception:
        return None


def _build_layer(df, color):
    # Common deck.gl Scatterplot layer configuration.
    return pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position=["lon", "lat"],
        get_radius="radius_m",
        radius_units="meters",
        radius_scale=1,
        radius_min_pixels=3,
        get_fill_color=color,
        pickable=True,
        auto_highlight=True,
        opacity=0.65,
    )


def _radius(df, value_col: str) -> Tuple[float, float]:
    # Scale point radii between 10–30 meters relative to the column range.
    if df.empty:
        return 0.0, 0.0
    mn = df[value_col].min()
    mx = df[value_col].max()
    if mn == mx:
        df["radius_m"] = 30.0
    else:
        df["radius_m"] = df[value_col].apply(lambda v: 10.0 + (v - mn) / (mx - mn) * (30.0 - 10.0))
    return mn, mx


def _ensure_alloc_columns(df):
    # Guarantee all expected allocation columns exist to prevent KeyErrors.
    for col in [
        "Category",
        "Materials",
        "Cubic Feet",
        "Approx Line Down Order",
        "Alloc Total Materials",
        "Alloc Total Line Down Orders",
        "Alloc Total Cubic Feet",
        "Current Cabinets",
        "Current Shelves",
        "Current Modulas",
        "Current Racks",
        "Travel Time",
    ]:
        if col not in df:
            df[col] = 0
    return df


def render():
    # Reduce page padding so the map can span edge-to-edge and keep tooltips legible.
    st.markdown(
        """
        <style>
        .block-container {
            padding-left: 0rem;
            padding-right: 0rem;
        }
        /* Force deck.gl tooltip legibility */
        .deck-tooltip {
            color: #ffffff !important;
            background: rgba(0, 0, 0, 0.85) !important;
            border: 1px solid rgba(255, 255, 255, 0.5) !important;
        }
        .deck-tooltip * {
            color: #ffffff !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Back to dashboard: swap session page and rerun the app to refresh.
    if st.button("Back to Optimization", type="secondary", key="map_back_to_opt"):
        st.session_state["page"] = "optimize"
        if hasattr(st, "experimental_rerun"):
            st.experimental_rerun()
        else:
            st.rerun()

    # Load current vs optimized allocation tables and derive priority/movement/size columns.
    current_df, alloc_df = alternate_dashboard.build_map_tables()
    current_df = _add_cat_dims(current_df)
    alloc_df = _add_cat_dims(_ensure_alloc_columns(alloc_df))

    # Build filter choices from data; use session defaults so selections persist on rerun.
    pri_options = sorted(current_df["Priority"].dropna().unique().tolist())
    mov_options = sorted(current_df["Movement"].dropna().unique().tolist())
    size_options = sorted(current_df["Size"].dropna().unique().tolist())
    st.session_state.setdefault("map_pri_filter", "Any")
    st.session_state.setdefault("map_mov_filter", "Any")
    st.session_state.setdefault("map_size_filter", "Any")

    # Render controls before applying filters so changes take effect immediately on rerun.
    with st.expander("Filters", expanded=True):
        col_filters = st.columns(3)
        col_filters[0].selectbox("Priority", ["Any"] + pri_options, key="map_pri_filter")
        col_filters[1].selectbox("Movement", ["Any"] + mov_options, key="map_mov_filter")
        col_filters[2].selectbox("Size", ["Any"] + size_options, key="map_size_filter")

    pri_filter = st.session_state.get("map_pri_filter", "Any")
    mov_filter = st.session_state.get("map_mov_filter", "Any")
    size_filter = st.session_state.get("map_size_filter", "Any")

    # Apply filters to both datasets using the helper above.
    filtered_df = _apply_filters(current_df, pri_filter, mov_filter, size_filter)
    filtered_alloc = _apply_filters(alloc_df, pri_filter, mov_filter, size_filter)

    # Aggregate totals per location for current inventory.
    current_totals = (
        filtered_df.groupby(["Location", "lat", "lon"], as_index=False)[
            [
                "Material Count",
                "Total Cubic Feet",
                "Total Line Down Orders",
                "Travel Time",
                "Current Cabinets",
                "Current Shelves",
                "Current Modulas",
                "Current Racks",
            ]
        ]
        .agg(
            {
                "Material Count": "sum",
                "Total Cubic Feet": "sum",
                "Total Line Down Orders": "sum",
                "Travel Time": "max",
                "Current Cabinets": "max",
                "Current Shelves": "max",
                "Current Modulas": "max",
                "Current Racks": "max",
            }
        )
        .rename(
            columns={
                "Material Count": "Total Material Count",
                "Total Cubic Feet": "Total Cubic Feet (All Categories)",
                "Total Line Down Orders": "Total Line Down Orders (All Categories)",
            }
        )
    )

    # Aggregate totals per location for optimized allocation.
    alloc_totals = (
        filtered_alloc.groupby(["Location", "lat", "lon"], as_index=False)[
            [
                "Materials",
                "Cubic Feet",
                "Approx Line Down Order",
                "Alloc Total Materials",
                "Alloc Total Line Down Orders",
                "Alloc Total Cubic Feet",
                "Travel Time",
                "Current Cabinets",
                "Current Shelves",
                "Current Modulas",
                "Current Racks",
            ]
        ]
        .agg(
            {
                "Materials": "sum",
                "Cubic Feet": "sum",
                "Approx Line Down Order": "sum",
                "Alloc Total Materials": "max",
                "Alloc Total Line Down Orders": "max",
                "Alloc Total Cubic Feet": "max",
                "Travel Time": "max",
                "Current Cabinets": "max",
                "Current Shelves": "max",
                "Current Modulas": "max",
                "Current Racks": "max",
            }
        )
        .rename(
            columns={
                "Materials": "Alloc Material Count",
                "Cubic Feet": "Alloc Cubic Feet",
                "Approx Line Down Order": "Alloc Line Down Orders",
                "Current Cabinets": "Recommended Cabinets",
                "Current Shelves": "Recommended Shelves",
                "Current Modulas": "Recommended Modulas",
                "Current Racks": "Recommended Racks",
            }
        )
    )

    current_totals["Travel Time Total"] = (
        current_totals["Travel Time"] * current_totals["Total Line Down Orders (All Categories)"]
    )
    alloc_totals["Travel Time Total"] = alloc_totals["Travel Time"] * alloc_totals["Alloc Total Line Down Orders"]

    # Overall travel time comparison: convert minutes to hours and compute savings/dollar value.
    current_travel_total = float(current_totals["Travel Time Total"].sum())  if not current_totals.empty else 0.0
    alloc_travel_total = float(alloc_totals["Travel Time Total"].sum())  if not alloc_totals.empty else 0.0
    travel_savings = (current_travel_total - alloc_travel_total) / 60
    dollar_savings = (current_travel_total - alloc_travel_total) * 882
    current_travel_total = current_travel_total / 60
    alloc_travel_total = alloc_travel_total / 60

    metric_cols = st.columns(4)
    metric_cols[0].metric("Travel time (current total)", f"{current_travel_total:,.2f} hours")
    metric_cols[1].metric("Travel time (recommended total)", f"{alloc_travel_total:,.2f} hours")
    metric_cols[2].metric("Overall travel time savings", f"{travel_savings:,.2f} hours")
    metric_cols[3].metric("Dollar value", f"${dollar_savings:,.2f}")

    _radius(current_totals, "Total Material Count")
    _radius(alloc_totals, "Alloc Material Count")

    # Configure map style/token and build layers for current vs recommended views.
    initial_view = pdk.ViewState(latitude=34.89285686441144, longitude=-82.17941293818576, zoom=13, pitch=0)
    mapbox_token = _mapbox_token()
    map_style = "mapbox://styles/mapbox/light-v9"
    map_kwargs = {"map_style": map_style, "initial_view_state": initial_view}
    if mapbox_token:
        map_kwargs["mapbox_key"] = mapbox_token
    else:
        map_kwargs["map_style"] = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"

    layer_current = _build_layer(current_totals, color=[0, 122, 255])
    layer_alloc = _build_layer(alloc_totals, color=[255, 99, 71])

    tooltip_style = {"style": {"color": "#ffffff", "backgroundColor": "rgba(0, 0, 0, 0.85)"}}

    tooltip_current = {
        "html": (
            "<b>Location:</b> {Location}<br/>"
            "<b>Total materials:</b> {Total Material Count}<br/>"
            "<b>Total cubic feet:</b> {Total Cubic Feet (All Categories)}<br/>"
            "<b>Total line down orders:</b> {Total Line Down Orders (All Categories)}<br/>"
            "<b>Cabinets:</b> {Current Cabinets}<br/>"
            "<b>Shelves:</b> {Current Shelves}<br/>"
            "<b>Modulas:</b> {Current Modulas}<br/>"
            "<b>Racks:</b> {Current Racks}<br/>"
            "<b>Travel time (per order):</b> {Travel Time}<br/>"
            "<b>Travel time total:</b> {Travel Time Total}"
        ),
        **tooltip_style,
    }

    tooltip_alloc = {
        "html": (
            "<b>Location:</b> {Location}<br/>"
            "<b>Total materials:</b> {Alloc Total Materials}<br/>"
            "<b>Total cubic feet:</b> {Alloc Total Cubic Feet}<br/>"
            "<b>Total line down orders:</b> {Alloc Total Line Down Orders}<br/>"
            "<b>Cabinets:</b> {Recommended Cabinets}<br/>"
            "<b>Shelves:</b> {Recommended Shelves}<br/>"
            "<b>Modulas:</b> {Recommended Modulas}<br/>"
            "<b>Racks:</b> {Recommended Racks}<br/>"
            "<b>Travel time (per order):</b> {Travel Time}<br/>"
            "<b>Travel time total:</b> {Travel Time Total}"
        ),
        **tooltip_style,
    }

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Current Allocation")
        if current_totals.empty:
            st.info("No data matches the selected filters.")
        else:
            st.pydeck_chart(pdk.Deck(**map_kwargs, layers=[layer_current], tooltip=tooltip_current))

    with col2:
        st.subheader("Model Allocation")
        if alloc_totals.empty:
            st.info("No data matches the selected filters.")
        else:
            st.pydeck_chart(pdk.Deck(**map_kwargs, layers=[layer_alloc], tooltip=tooltip_alloc))

    # Breakdown tables side-by-side using same selected location
    locations = sorted(set(filtered_df["Location"].dropna()) | set(filtered_alloc["Location"].dropna()))
    if not locations:
        st.info("No locations match the selected filters.")
        return

    # Pick a location to drive detailed breakdown below the maps.
    selected_loc = st.selectbox("Location breakdown", locations, index=0, key="map_loc_select")

    df_cols = st.columns(2)
    with df_cols[0]:
        travel_row = current_totals[current_totals["Location"] == selected_loc]
        if not travel_row.empty:
            travel_time = travel_row.iloc[0].get("Travel Time", None)
            travel_total = travel_row.iloc[0].get("Travel Time Total", None) / 60
            if travel_time is not None and travel_total is not None:
                st.markdown(f"**Travel time (per line down order):** {travel_time:.2f} mins")
                st.markdown(f"**Travel time total (line down orders × travel time):** {travel_total:.2f} hours")
        breakdown = (
            filtered_df[filtered_df["Location"] == selected_loc]
            .groupby("Category", as_index=False)[["Material Count", "Total Cubic Feet", "Total Line Down Orders"]]
            .sum()
            .rename(
                columns={
                    "Material Count": "Materials",
                    "Total Cubic Feet": "Cubic Feet",
                    "Total Line Down Orders": "Line Down Orders",
                }
            )
            .sort_values("Materials", ascending=False)
        )
        st.markdown(f"**Current category breakdown for {selected_loc}**")
        st.dataframe(breakdown.reset_index(drop=True), width="stretch")

        type_cols = ["Current Cabinets", "Current Shelves", "Current Modulas", "Current Racks"]
        available_type_cols = [c for c in type_cols if c in current_totals.columns]
        if available_type_cols:
            type_row = current_totals[current_totals["Location"] == selected_loc][available_type_cols]
            if not type_row.empty:
                type_data = (
                    type_row.T.reset_index()
                    .rename(columns={"index": "Storage Type", type_row.index[0]: "Units"})
                    .sort_values("Storage Type")
                )
                st.markdown(f"**Current storage types at {selected_loc}**")
                st.dataframe(type_data.reset_index(drop=True), width="stretch")

    with df_cols[1]:
        travel_row_a = alloc_totals[alloc_totals["Location"] == selected_loc]
        if not travel_row_a.empty:
            travel_time_a = travel_row_a.iloc[0].get("Travel Time", None)
            travel_total_a = travel_row_a.iloc[0].get("Travel Time Total", None) / 60
            if travel_time_a is not None and travel_total_a is not None:
                st.markdown(f"**Travel time (per line-down order):** {travel_time_a:.2f} mins")
                st.markdown(f"**Travel time total (line down orders × travel time):** {travel_total_a:.2f} hours")
        alloc_breakdown = (
            filtered_alloc[filtered_alloc["Location"] == selected_loc]
            .groupby("Category", as_index=False)[["Materials", "Cubic Feet", "Approx Line Down Order"]]
            .sum()
            .rename(
                columns={
                    "Materials": "Materials",
                    "Cubic Feet": "Cubic Feet",
                    "Approx Line Down Order": "Line Down Orders",
                }
            )
            .sort_values("Materials", ascending=False)
        )
        st.markdown(f"**Model category breakdown for {selected_loc}**")
        st.dataframe(alloc_breakdown.reset_index(drop=True), width="stretch")

        type_cols_a = ["Recommended Cabinets", "Recommended Shelves", "Recommended Modulas", "Recommended Racks"]
        available_type_cols_a = [c for c in type_cols_a if c in alloc_totals.columns]
        if available_type_cols_a:
            type_row_a = alloc_totals[alloc_totals["Location"] == selected_loc][available_type_cols_a]
            if not type_row_a.empty:
                type_data_a = (
                    type_row_a.T.reset_index()
                    .rename(columns={"index": "Storage Type", type_row_a.index[0]: "Units"})
                    .sort_values("Storage Type")
                )
                st.markdown(f"**Model storage types at {selected_loc}**")
                st.dataframe(type_data_a.reset_index(drop=True), width="stretch")


if __name__ == "__main__":
    render()
