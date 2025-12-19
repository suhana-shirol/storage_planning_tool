import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import altair as alt
import pandas as pd
import streamlit as st

from optimization import alternate_dashboard

ROOT = Path(__file__).resolve().parents[2]
DASHBOARD_XLSX = ROOT / "optimization" / "Dashboard.xlsx"
PARAMS_JSON_PATH = ROOT / "optimization" / "params.json"


def _dashboard_path() -> Path:
    # Use session-provided dashboard path when available; otherwise default location.
    path_str = st.session_state.get("dashboard_path")
    if path_str:
        return Path(path_str)
    return DASHBOARD_XLSX


def _load_params_json() -> Optional[Dict]:
    # Load params.json if it exists; show warning instead of raising on failures.
    if not PARAMS_JSON_PATH.exists():
        return None
    try:
        return json.loads(PARAMS_JSON_PATH.read_text())
    except Exception as exc:
        st.warning(f"Unable to load params.json: {exc}")
        return None


def _build_store_tables(decision_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Use alternate_dashboard to build the three store-related tables."""
    # Ensure alternate_dashboard uses the currently loaded decision variables
    alternate_dashboard.opti_df = decision_df
    # Build outputs from alternate_dashboard helper; handle older return shapes gracefully.
    results = alternate_dashboard.build_store_units()
    if len(results) == 3:
        units_change_df, buy_units_df, reloc_units_df = results
    else:
        units_change_df, reloc_units_df = results
        buy_units_df = pd.DataFrame()

    return units_change_df, buy_units_df, reloc_units_df


def _build_sku_alloc(decision_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build comparison of current vs optimized SKU allocation (forward counts/cubic and reserve counts/cubic)."""
    alternate_dashboard.opti_df = decision_df
    return alternate_dashboard.build_SKUalloc()


def _sku_unique_chart(df: pd.DataFrame, title: str) -> alt.Chart:
    tidy = df.melt(
        id_vars=["Storage Type", "Location"],
        value_vars=["Current Count", "Recommended Count"],
        var_name="Metric",
        value_name="Count",
    )
    storage_types = sorted(tidy["Storage Type"].dropna().unique())
    # BMW-inspired palette: lighter tints for current, darker shades for recommended
    current_palette = ["#66b2ff", "#59a8f7", "#4b9eee", "#3e94e6", "#318add", "#2480d5"]  # brighter forward blues
    recommended_palette = ["#1c69d4", "#0f5bb5", "#0a4a90", "#063b75", "#022c59", "#011f40"]
    color_domain: list[str] = []
    color_range: list[str] = []
    tidy["Segment"] = tidy.apply(
        lambda row: f"{row['Metric'].split()[0]} - {row['Storage Type']}", axis=1
    )
    for idx, stype in enumerate(storage_types):
        color_domain.append(f"Current - {stype}")
        color_range.append(current_palette[idx % len(current_palette)])
    for idx, stype in enumerate(storage_types):
        color_domain.append(f"Recommended - {stype}")
        color_range.append(recommended_palette[idx % len(recommended_palette)])

    return (
        alt.Chart(tidy)
        .mark_bar()
        .encode(
            x=alt.X("Location:N", title="Location"),
            xOffset=alt.XOffset("Metric:N", sort=["Current Count", "Recommended Count"]),
            y=alt.Y("Count:Q", title="Count"),
            color=alt.Color(
                "Segment:N",
                scale=alt.Scale(domain=color_domain, range=color_range),
                legend=alt.Legend(title="Storage Type / Metric"),
            ),
            order=alt.Order("Storage Type:N", sort="ascending"),
            tooltip=[
                "Storage Type",
                "Location",
                "Metric",
                alt.Tooltip("Count:Q", format=".2f"),
            ],
        )
        .properties(title=title)
    )


def _sku_cubic_chart(df: pd.DataFrame, title: str) -> alt.Chart:
    tidy = df.melt(
        id_vars=["Storage Type", "Location"],
        value_vars=["Current Cubic Feet", "Recommended Cubic Feet"],
        var_name="Metric",
        value_name="Cubic Feet",
    )
    storage_types = sorted(tidy["Storage Type"].dropna().unique())
    current_palette = ["#c7d9f1", "#a8c4e6", "#8bafdb", "#6c9ad0", "#4d85c5", "#2f70ba"]
    recommended_palette = ["#1c69d4", "#0f5bb5", "#0a4a90", "#063b75", "#022c59", "#011f40"]
    color_domain: list[str] = []
    color_range: list[str] = []
    tidy["Segment"] = tidy.apply(
        lambda row: f"{row['Metric'].split()[0]} - {row['Storage Type']}", axis=1
    )
    for idx, stype in enumerate(storage_types):
        color_domain.append(f"Current - {stype}")
        color_range.append(current_palette[idx % len(current_palette)])
    for idx, stype in enumerate(storage_types):
        color_domain.append(f"Recommended - {stype}")
        color_range.append(recommended_palette[idx % len(recommended_palette)])

    return (
        alt.Chart(tidy)
        .mark_bar()
        .encode(
            x=alt.X("Location:N", title="Location"),
            xOffset=alt.XOffset("Metric:N", sort=["Current Cubic Feet", "Recommended Cubic Feet"]),
            y=alt.Y("Cubic Feet:Q", title="Cubic Feet"),
            color=alt.Color(
                "Segment:N",
                scale=alt.Scale(domain=color_domain, range=color_range),
                legend=alt.Legend(title="Storage Type / Metric"),
            ),
            order=alt.Order("Storage Type:N", sort="ascending"),
            tooltip=[
                "Storage Type",
                "Location",
                "Metric",
                alt.Tooltip("Cubic Feet:Q", format=".2f"),
            ],
        )
        .properties(title=title)
    )


def _sku_forward_reserve_chart(df: pd.DataFrame, title: str, value_fields: Tuple[str, str], value_name: str, y_title: str) -> alt.Chart:
    tidy = df.melt(
        id_vars=["Storage Type", "Location"],
        value_vars=list(value_fields),
        var_name="Phase",
        value_name=value_name,
    )
    storage_types = sorted(tidy["Storage Type"].dropna().unique())
    forward_palette = ["#66b2ff", "#59a8f7", "#4b9eee", "#3e94e6", "#318add", "#2480d5"]
    reserve_palette = ["#c4c9ce", "#b7bdc3", "#aab1b8", "#9da5ad", "#9099a2", "#848d97"]
    tidy["Segment"] = tidy.apply(lambda r: f"{r['Phase']} - {r['Storage Type']}", axis=1)
    color_domain, color_range = [], []
    for i, st in enumerate(storage_types):
        color_domain.append(f"{value_fields[0]} - {st}")
        color_range.append(forward_palette[i % len(forward_palette)])
    for i, st in enumerate(storage_types):
        color_domain.append(f"{value_fields[1]} - {st}")
        color_range.append(reserve_palette[i % len(reserve_palette)])

    return (
        alt.Chart(tidy)
        .mark_bar()
        .encode(
            x=alt.X("Location:N", title="Location"),
            xOffset=alt.XOffset("Phase:N", sort=list(value_fields)),
            y=alt.Y(f"{value_name}:Q", title=y_title, stack="zero"),
            color=alt.Color("Segment:N", scale=alt.Scale(domain=color_domain, range=color_range),
                            legend=alt.Legend(title="Phase / Storage Type")),
            order=alt.Order("Storage Type:N", sort="ascending"),
            tooltip=["Location", "Storage Type", "Phase", alt.Tooltip(f"{value_name}:Q", format=".2f")],
        )
        .properties(title=title)
    )


def _sku_forward_reserve_percent_chart(df: pd.DataFrame, title: str) -> alt.Chart:
    rows = []
    for _, row in df.iterrows():
        loc = row.get("Location")
        stype = row.get("Storage Type")
        cur = row.get("Current % Cubic Feet", 0) or 0
        fwd = row.get("Recommended % Cubic Feet Forward", row.get("% Forward Cubic Feet", 0)) or 0
        res = row.get("Recommended % Reserve Cubic Feet", row.get("% Reserve Cubic Feet", 0)) or 0
        rows += [
            {"Location": loc, "Storage Type": stype, "Phase": "Current", "Percent": cur, "Offset": "Current"},
            {"Location": loc, "Storage Type": stype, "Phase": "Forward", "Percent": fwd, "Offset": "Recommended"},
            {"Location": loc, "Storage Type": stype, "Phase": "Reserve", "Percent": res, "Offset": "Recommended"},
        ]
    tidy = pd.DataFrame(rows)
    storage_types = sorted(tidy["Storage Type"].dropna().unique())
    forward_palette = ["#66b2ff", "#59a8f7", "#4b9eee", "#3e94e6", "#318add", "#2480d5"]
    reserve_palette = ["#c4c9ce", "#b7bdc3", "#aab1b8", "#9da5ad", "#9099a2", "#848d97"]
    current_palette = ["#acc9f2", "#9abcf2", "#88aff2", "#76a2f2", "#6495f2", "#5288f2"]
    color_domain, color_range = [], []
    for i, st in enumerate(storage_types):
        color_domain.append(f"Current - {st}"); color_range.append(current_palette[i % len(current_palette)])
    for i, st in enumerate(storage_types):
        color_domain.append(f"Forward - {st}"); color_range.append(forward_palette[i % len(forward_palette)])
    for i, st in enumerate(storage_types):
        color_domain.append(f"Reserve - {st}"); color_range.append(reserve_palette[i % len(reserve_palette)])
    tidy["Segment"] = tidy.apply(lambda r: f"{r['Phase']} - {r['Storage Type']}", axis=1)
    return (
        alt.Chart(tidy)
        .mark_bar()
        .encode(
            x=alt.X("Location:N", title="Location"),
            xOffset=alt.XOffset("Offset:N", sort=["Current", "Recommended"]),
            y=alt.Y("Percent:Q", title="Percent", stack="zero"),
            color=alt.Color("Segment:N", scale=alt.Scale(domain=color_domain, range=color_range),
                            legend=alt.Legend(title="Phase / Storage Type")),
            order=alt.Order("Phase:N"),
            tooltip=["Location", "Storage Type", "Phase", alt.Tooltip("Percent:Q", format=".2f")],
        )
        .properties(title=title)
    )


def _units_change_chart(df: pd.DataFrame, title: str) -> alt.Chart:
    tidy = df.melt(
        id_vars="Storage Type",
        value_vars=["Current # of Units", "Recommended # of Units"],
        var_name="Metric",
        value_name="Units",
    )
    colors = ["#c7d9f1", "#1c69d4"]  # light = current, dark = recommended
    metric_order = ["Current # of Units", "Recommended # of Units"]
    return (
        alt.Chart(tidy)
        .mark_bar()
        .encode(
            x=alt.X("Storage Type:N", title="Storage Type"),
            xOffset=alt.XOffset("Metric:N", sort=metric_order),
            y=alt.Y("Units:Q", title="Units"),
            color=alt.Color(
                "Metric:N",
                sort=metric_order,
                scale=alt.Scale(domain=metric_order, range=colors),
                legend=alt.Legend(title=None),
            ),
            tooltip=["Storage Type", "Metric", alt.Tooltip("Units:Q", format=".2f")],
        )
        .properties(title=title)
    )


def _buy_units_chart(df: pd.DataFrame, title: str) -> alt.Chart:
    col = "Number of Units"
    color = "#1c69d4"
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("Storage Type:N", title="Storage Type"),
            y=alt.Y(f"{col}:Q", title="Units"),
            color=alt.value(color),
            tooltip=["Storage Type", alt.Tooltip(f"{col}:Q", format=".2f")],
        )
        .properties(title=title)
    )


def _reloc_units_chart(df: pd.DataFrame, title: str) -> alt.Chart:
    tidy = df.rename(columns={"Number of Units": "Units"})
    return (
        alt.Chart(tidy)
        .mark_bar()
        .encode(
            x=alt.X("Recommended Location:N", title="Destination"),
            y=alt.Y("Units:Q", title="Units"),
            color=alt.Color("Storage Type:N"),
            tooltip=[
                "Storage Type",
                "Current Location",
                "Recommended Location",
                alt.Tooltip("Units:Q", format=".2f"),
            ],
        )
        .properties(title=title)
    )


def render():
    st.header("Optimization Dashboard")

    # Quick link to the map view for visualizing allocations
    if st.button("View Map", type="primary"):
        st.session_state["page"] = "map"
        st.rerun()

    decision_path = _dashboard_path()
    decision_df: Optional[pd.DataFrame] = None
    dashboard_tables: Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = None
    sku_alloc_counts: Optional[pd.DataFrame] = None
    sku_alloc_cubic: Optional[pd.DataFrame] = None
    res_alloc_counts: Optional[pd.DataFrame] = None
    res_alloc_cubic: Optional[pd.DataFrame] = None

    if decision_path.exists():
        try:
            # Load the generated dashboard workbook and offer a download mirror.
            decision_df = pd.read_excel(decision_path)
            st.caption(f"Download this dashboard as an XLSX")
            st.download_button(
                "Download dashboard",
                data=decision_path.read_bytes(),
                file_name=decision_path.name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="secondary",
            )
        except Exception as exc:
            st.error(f"Failed to load dashboard: {exc}")
    else:
        st.info("Run the model to generate dashboard.")

    if decision_df is not None:
        try:
            # Build all downstream tables once we have decision variables.
            dashboard_tables = _build_store_tables(decision_df)
            sku_alloc_counts, sku_alloc_cubic, res_alloc_counts, res_alloc_cubic = _build_sku_alloc(decision_df)
        except Exception as exc:
            st.error(f"Failed to build dashboard tables: {exc}")

    if dashboard_tables:
        units_change_df, buy_units_df, reloc_units_df = dashboard_tables
        st.markdown("---")
        st.subheader("Dashboard")

        section = st.selectbox("Section", ["Storage Type Configuration", "Forward SKU Allocation", "Forward + Reserve SKU Allocation"], key="dashboard_section")

        if section == "Storage Type Configuration":
            tabs = st.tabs(["Total Units Change", "Buy Units", "Relocations"])

            with tabs[0]:
                if units_change_df.empty:
                    st.info("No storage unit changes were generated.")
                else:
                    st.info("This display shows the total number of each storage type needed for optimal storage.")
                    locations = sorted(units_change_df["Location"].dropna().unique())
                    show_all = st.checkbox("Show all locations", key="units_change_all")
                    if show_all:
                        cols = st.columns(3)
                        for idx, loc in enumerate(locations):
                            filtered = units_change_df[units_change_df["Location"] == loc]
                            target_col = cols[idx % len(cols)]
                            with target_col:
                                st.altair_chart(
                                    _units_change_chart(filtered, f"{loc}"),
                                    width="stretch",
                                )
                    else:
                        selected_loc = st.selectbox("Location", locations, key="units_change_location")
                        filtered = units_change_df[units_change_df["Location"] == selected_loc]
                        st.altair_chart(
                            _units_change_chart(filtered, f"Current vs Recommended Units - {selected_loc}"),
                            width="stretch",
                        )
                        display_cols_order_units = [
                            "Location",
                            "Storage Type",
                            "Current Number of Units",
                            "Recommended Number of Units",
                        ]
                        display_cols_units = [c for c in display_cols_order_units if c in filtered.columns]
                        display_cols_units += [c for c in filtered.columns if c not in display_cols_units]
                        st.dataframe(filtered[display_cols_units].reset_index(drop=True), width="stretch")

            with tabs[1]:
                if buy_units_df.empty:
                    st.info("No buy units were recommended.")
                else:
                    st.info("The number of each storage type to buy to have optimal storage.")
                    locations = sorted(buy_units_df["Location"].dropna().unique())
                    selected_loc = st.selectbox("Location", locations, key="buy_units_location")
                    filtered = buy_units_df[buy_units_df["Location"] == selected_loc]
                    st.altair_chart(
                        _buy_units_chart(filtered, f"Buy Units - {selected_loc}"),
                        width="stretch",
                    )
                    st.dataframe(filtered.reset_index(drop=True), width="stretch")

            with tabs[2]:
                if reloc_units_df.empty:
                    st.info("No relocations were recommended.")
                else:
                    st.info("The number of each storage type to move between locations to have optimal storage.")
                    origins = sorted(reloc_units_df["Current Location"].dropna().unique())
                    selected_origin = st.selectbox("Current Location", origins, key="reloc_origin")
                    filtered = reloc_units_df[reloc_units_df["Current Location"] == selected_origin]
                    st.altair_chart(
                        _reloc_units_chart(filtered, f"Relocations from {selected_origin}"),
                        width="stretch",
                    )
                    st.dataframe(filtered.reset_index(drop=True), width="stretch")

        elif section == "Forward SKU Allocation":
            if (sku_alloc_counts is None or sku_alloc_counts.empty) and (sku_alloc_cubic is None or sku_alloc_cubic.empty):
                st.info("No SKU allocation data available.")
            else:
                def _add_cat_dims(df: pd.DataFrame) -> pd.DataFrame:
                    if df is None or df.empty:
                        return df
                    out = df.copy()
                    cat_str = out["Category"].astype(str)
                    out["Priority"] = cat_str.str[0]
                    out["Movement"] = cat_str.str[1]
                    out["Size"] = cat_str.str[2]
                    return out

                counts_df = _add_cat_dims(sku_alloc_counts)
                cubic_df = _add_cat_dims(sku_alloc_cubic)

                pri_options = sorted(set(counts_df["Priority"].dropna()) | set(cubic_df["Priority"].dropna()))
                mov_options = sorted(set(counts_df["Movement"].dropna()) | set(cubic_df["Movement"].dropna()))
                size_options = sorted(set(counts_df["Size"].dropna()) | set(cubic_df["Size"].dropna()))

                col_filters = st.columns(3)
                pri_filter = col_filters[0].selectbox("Priority", ["Any"] + pri_options, key="cat_pri")
                mov_filter = col_filters[1].selectbox("Movement", ["Any"] + mov_options, key="cat_mov")
                size_filter = col_filters[2].selectbox("Size", ["Any"] + size_options, key="cat_size")

                def _apply_filters(df: pd.DataFrame) -> pd.DataFrame:
                    if df is None or df.empty:
                        return df
                    out = df
                    if pri_filter != "Any":
                        out = out[out["Priority"] == pri_filter]
                    if mov_filter != "Any":
                        out = out[out["Movement"] == mov_filter]
                    if size_filter != "Any":
                        out = out[out["Size"] == size_filter]
                    return out

                counts_filtered = _apply_filters(counts_df)
                cubic_filtered = _apply_filters(cubic_df)

                categories = sorted(
                    pd.concat(
                        [
                            counts_filtered["Category"] if counts_filtered is not None else pd.Series(dtype=str),
                            cubic_filtered["Category"] if cubic_filtered is not None else pd.Series(dtype=str),
                        ],
                        ignore_index=True,
                    )
                    .dropna()
                    .unique()
                    .tolist()
                )

                if not categories:
                    st.info("No SKU allocation data matches the selected filters.")
                else:
                    show_all_cats = st.checkbox("Show all matching categories", key="alloc_show_all")
                    selected_cat = categories[0] if show_all_cats else st.selectbox("SKU Category", categories, key="sku_category")
                    tabs_alloc = st.tabs(["Count Allocation", "Cubic Allocation"])

                    with tabs_alloc[0]:
                        if counts_filtered is None or counts_filtered.empty:
                            st.info("No count allocation data available.")
                        else:
                            st.info("The number and % of unique materials allocated to each forward storage location/type in an optimal storage layout.")
                            if show_all_cats:
                                cols = st.columns(3)
                                for idx, cat in enumerate(categories):
                                    subset = counts_filtered[counts_filtered["Category"] == cat]
                                    target = cols[idx % len(cols)]
                                    with target:
                                        st.altair_chart(
                                            _sku_unique_chart(subset, f"{cat} (Count)"),
                                            width="stretch",
                                        )
                            else:
                                filtered = counts_filtered[counts_filtered["Category"] == selected_cat]
                                st.altair_chart(
                                    _sku_unique_chart(filtered, f"SKU Allocation (Count) - {selected_cat}"),
                                    width="stretch",
                                )
                                display_cols_order = [
                                    "Category",
                                    "Priority",
                                    "Movement",
                                    "Size",
                                    "Location",
                                    "Storage Type",
                                    "Current Count",
                                    "Current % of Category",
                                    "Recommended Count",
                                    "Recommended % Forward",
                                    "Total Category Count",
                                ]
                                display_cols = [c for c in display_cols_order if c in filtered.columns]
                                display_cols += [c for c in filtered.columns if c not in display_cols]
                                st.dataframe(filtered[display_cols].reset_index(drop=True), width="stretch")

                    with tabs_alloc[1]:
                        if cubic_filtered is None or cubic_filtered.empty:
                            st.info("No cubic allocation data available.")
                        else:
                            st.info("The volume (in cubic feet) of materials allocated to each forward storage location/type in an optimal storage layout.")
                            if show_all_cats:
                                cols_c = st.columns(3)
                                for idx, cat in enumerate(categories):
                                    subset_c = cubic_filtered[cubic_filtered["Category"] == cat]
                                    target_c = cols_c[idx % len(cols_c)]
                                    with target_c:
                                        st.altair_chart(
                                            _sku_cubic_chart(subset_c, f"{cat} (Cubic Ft)"),
                                            width="stretch",
                                        )
                            else:
                                filtered_cubic = cubic_filtered[cubic_filtered["Category"] == selected_cat]
                                st.altair_chart(
                                    _sku_cubic_chart(filtered_cubic, f"SKU Allocation (Cubic Ft) - {selected_cat}"),
                                    width="stretch",
                                )
                                display_cols_order_cubic = [
                                    "Category",
                                    "Priority",
                                    "Movement",
                                    "Size",
                                    "Location",
                                    "Storage Type",
                                    "Current Cubic Feet",
                                    "Current % Cubic Feet",
                                    "Recommended Cubic Feet",
                                    "Recommended % Cubic Feet",
                                    "Total Cubic Feet",
                                ]
                                display_cols_cubic = [c for c in display_cols_order_cubic if c in filtered_cubic.columns]
                                display_cols_cubic += [c for c in filtered_cubic.columns if c not in display_cols_cubic]
                                st.dataframe(filtered_cubic[display_cols_cubic].reset_index(drop=True), width="stretch")

        elif section == "Forward + Reserve SKU Allocation":
            if ((sku_alloc_counts is None or sku_alloc_counts.empty) and (sku_alloc_cubic is None or sku_alloc_cubic.empty)
                and (res_alloc_counts is None or res_alloc_counts.empty) and (res_alloc_cubic is None or res_alloc_cubic.empty)):
                st.info("No SKU allocation data available.")
            else:
                def _add_cat_dims(df: pd.DataFrame) -> pd.DataFrame:
                    if df is None or df.empty:
                        return df
                    out = df.copy()
                    cat_str = out["Category"].astype(str)
                    out["Priority"] = cat_str.str[0]
                    out["Movement"] = cat_str.str[1]
                    out["Size"] = cat_str.str[2]
                    return out

                f_counts = _add_cat_dims(sku_alloc_counts)
                f_cubic = _add_cat_dims(sku_alloc_cubic)
                r_counts = _add_cat_dims(res_alloc_counts)
                r_cubic = _add_cat_dims(res_alloc_cubic)

                pri_options = sorted(set(f_counts["Priority"].dropna()) | set(f_cubic["Priority"].dropna()) | set(r_counts["Priority"].dropna()) | set(r_cubic["Priority"].dropna()))
                mov_options = sorted(set(f_counts["Movement"].dropna()) | set(f_cubic["Movement"].dropna()) | set(r_counts["Movement"].dropna()) | set(r_cubic["Movement"].dropna()))
                size_options = sorted(set(f_counts["Size"].dropna()) | set(f_cubic["Size"].dropna()) | set(r_counts["Size"].dropna()) | set(r_cubic["Size"].dropna()))

                col_filters = st.columns(3)
                pri_filter = col_filters[0].selectbox("Priority", ["Any"] + pri_options, key="fr_pri")
                mov_filter = col_filters[1].selectbox("Movement", ["Any"] + mov_options, key="fr_mov")
                size_filter = col_filters[2].selectbox("Size", ["Any"] + size_options, key="fr_size")

                def _apply_filters(df: pd.DataFrame) -> pd.DataFrame:
                    if df is None or df.empty:
                        return df
                    out = df
                    if pri_filter != "Any":
                        out = out[out["Priority"] == pri_filter]
                    if mov_filter != "Any":
                        out = out[out["Movement"] == mov_filter]
                    if size_filter != "Any":
                        out = out[out["Size"] == size_filter]
                    return out

                f_counts_f = _apply_filters(f_counts)
                f_cubic_f = _apply_filters(f_cubic)
                r_counts_f = _apply_filters(r_counts)
                r_cubic_f = _apply_filters(r_cubic)

                categories = sorted(
                    pd.concat(
                        [
                            f_counts_f["Category"] if f_counts_f is not None else pd.Series(dtype=str),
                            f_cubic_f["Category"] if f_cubic_f is not None else pd.Series(dtype=str),
                            r_counts_f["Category"] if r_counts_f is not None else pd.Series(dtype=str),
                            r_cubic_f["Category"] if r_cubic_f is not None else pd.Series(dtype=str),
                        ],
                        ignore_index=True,
                    )
                    .dropna()
                    .unique()
                    .tolist()
                )

                if not categories:
                    st.info("No SKU allocation data matches the selected filters.")
                else:
                    show_all_cats = st.checkbox("Show all matching categories", key="alloc_fr_show_all")
                    selected_cat = categories[0] if show_all_cats else st.selectbox("SKU Category", categories, key="sku_category_fr")
                    tabs_alloc = st.tabs(["Count Allocation", "Cubic Allocation"])

                    with tabs_alloc[0]:
                        st.info("The number and % of unique materials allocated to each forward + reserve storage location/type in an optimal storage layout.")
                        if (f_counts_f is None or f_counts_f.empty) and (r_counts_f is None or r_counts_f.empty):
                            st.info("No count allocation data available.")
                        else:
                            if show_all_cats:
                                cols = st.columns(3)
                                for idx, cat in enumerate(categories):
                                    f_sub = f_counts_f[f_counts_f["Category"] == cat]
                                    r_sub = r_counts_f[r_counts_f["Category"] == cat]
                                    combined = f_sub.rename(columns={"Recommended Count": "Forward Count"})
                                    if not r_sub.empty:
                                        combined = combined.merge(
                                            r_sub[["Category","Storage Type","Location","Recommended Reserve Count","Recommended % Reserve"]],
                                                    how="outer",
                                                    on=["Category","Storage Type","Location"],
                                        )
                                    combined = combined.rename(columns={"Recommended Reserve Count": "Reserve Count"})
                                    if "Reserve Count" not in combined:
                                        combined["Reserve Count"] = 0.0
                                    else:
                                        combined["Reserve Count"] = combined["Reserve Count"].fillna(0)
                                    if "Forward Count" not in combined:
                                        combined["Forward Count"] = 0.0
                                    else:
                                        combined["Forward Count"] = combined["Forward Count"].fillna(0)
                                    if "Recommended % Reserve" not in combined:
                                        combined["Recommended % Reserve"] = 0.0
                                    else:
                                        combined["Recommended % Reserve"] = combined["Recommended % Reserve"].fillna(0)
                                    tgt = cols[idx % len(cols)]
                                    with tgt:
                                        st.altair_chart(
                                            _sku_forward_reserve_chart(
                                                combined,
                                                f"{cat} (Count)",
                                                ("Forward Count", "Reserve Count"),
                                                "Count",
                                                "Count",
                                            ),
                                            width="stretch",
                                        )
                            else:
                                f_sub = f_counts_f[f_counts_f["Category"] == selected_cat]
                                r_sub = r_counts_f[r_counts_f["Category"] == selected_cat]
                                combined = f_sub.rename(columns={"Recommended Count": "Forward Count"})
                                if not r_sub.empty:
                                    combined = combined.merge(
                                            r_sub[["Category","Storage Type","Location","Recommended Reserve Count","Recommended % Reserve"]],
                                                    how="outer",
                                                    on=["Category","Storage Type","Location"],
                                        )
                                    combined = combined.rename(columns={"Recommended % of Category Forward": "Recommended % Forward",
                                                                        "Recommended Reserve Count": "Reserve Count"})
                                if "Reserve Count" not in combined:
                                    combined["Reserve Count"] = 0.0
                                else:
                                    combined["Reserve Count"] = combined["Reserve Count"].fillna(0)
                                if "Forward Count" not in combined:
                                    combined["Forward Count"] = 0.0
                                else:
                                    combined["Forward Count"] = combined["Forward Count"].fillna(0)
                                if "Recommended % Reserve" not in combined:
                                    combined["Recommended % Reserve"] = 0.0
                                else:
                                    combined["Recommended % Reserve"] = combined["Recommended % Reserve"].fillna(0)
                                st.altair_chart(
                                    _sku_forward_reserve_chart(
                                        combined,
                                        f"SKU Allocation (Count) - {selected_cat}",
                                        ("Forward Count", "Reserve Count"),
                                        "Count",
                                        "Count",
                                    ),
                                    width="stretch",
                                )
                                display_cols_order = [
                                    "Category",
                                    "Priority",
                                    "Movement",
                                    "Size",
                                    "Location",
                                    "Storage Type",
                                    "Current Count",
                                    "Current % of Category",
                                    "Forward Count",
                                    "Recommended % Forward",
                                    "Reserve Count",
                                    "Recommended % Reserve",
                                    "Total Category Count",
                                ]
                                display_cols = [c for c in display_cols_order if c in combined.columns]
                                display_cols += [c for c in combined.columns if c not in display_cols]
                                st.dataframe(combined[display_cols].reset_index(drop=True), width="stretch")

                    with tabs_alloc[1]:
                        if (f_cubic_f is None or f_cubic_f.empty) and (r_cubic_f is None or r_cubic_f.empty):
                            st.info("No cubic allocation data available.")
                        else:
                            st.info("The volume (in cubic feet) of materials allocated to each forward + reserve storage location/type in an optimal storage layout.")
                            if show_all_cats:
                                cols_c = st.columns(3)
                                for idx, cat in enumerate(categories):
                                    f_sub = f_cubic_f[f_cubic_f["Category"] == cat]
                                    r_sub = r_cubic_f[r_cubic_f["Category"] == cat]
                                    combined_c = f_sub.rename(columns={"Recommended Cubic Feet": "Forward Cubic Feet", "Recommended % Cubic Feet": "% Forward Cubic Feet"})
                                    if not r_sub.empty:
                                        combined_c = combined_c.merge(
                                            r_sub[["Category","Storage Type","Location","Recommended Reserve Cubic Feet","Recommended % Reserve Cubic Feet"]],
                                            how="outer",
                                            on=["Category", "Storage Type", "Location"],
                                        )
                                    combined_c = combined_c.rename(columns={"Recommended Reserve Cubic Feet": "Reserve Cubic Feet",
                                                                            "Recommended % Reserve Cubic Feet": "% Reserve Cubic Feet"})
                                    if "Reserve Cubic Feet" not in combined_c:
                                        combined_c["Reserve Cubic Feet"] = 0.0
                                    else:
                                        combined_c["Reserve Cubic Feet"] = combined_c["Reserve Cubic Feet"].fillna(0)
                                    if "Forward Cubic Feet" not in combined_c:
                                        combined_c["Forward Cubic Feet"] = 0.0
                                    else:
                                        combined_c["Forward Cubic Feet"] = combined_c["Forward Cubic Feet"].fillna(0)
                                    if "% Reserve Cubic Feet" not in combined_c:
                                        combined_c["% Reserve Cubic Feet"] = 0.0
                                    else:
                                        combined_c["% Reserve Cubic Feet"] = combined_c["% Reserve Cubic Feet"].fillna(0)
                                    if "% Forward Cubic Feet" not in combined_c:
                                        combined_c["% Forward Cubic Feet"] = 0.0
                                    else:
                                        combined_c["% Forward Cubic Feet"] = combined_c["% Forward Cubic Feet"].fillna(0)
                                    tgt_c = cols_c[idx % len(cols_c)]
                                    with tgt_c:
                                        st.altair_chart(
                                            _sku_forward_reserve_percent_chart(
                                                combined_c,
                                                f"{cat} (Cubic Ft)"
                                            ),
                                            width="stretch",
                                        )
                            else:
                                f_sub = f_cubic_f[f_cubic_f["Category"] == selected_cat]
                                r_sub = r_cubic_f[r_cubic_f["Category"] == selected_cat]
                                combined_c = f_sub.rename(columns={"Recommended Cubic Feet": "Forward Cubic Feet", "Recommended % Cubic Feet": "% Forward Cubic Feet"})
                                if not r_sub.empty:
                                    combined_c = combined_c.merge(
                                        r_sub[["Category","Storage Type","Location","Recommended Reserve Cubic Feet","Recommended % Reserve Cubic Feet"]],
                                        how="outer",
                                        on=["Category", "Storage Type", "Location"],
                                    )
                                combined_c = combined_c.rename(columns={"Recommended Reserve Cubic Feet": "Reserve Cubic Feet",
                                                                        "Recommended % Reserve Cubic Feet": "% Reserve Cubic Feet"})
                                if "Reserve Cubic Feet" not in combined_c:
                                    combined_c["Reserve Cubic Feet"] = 0.0
                                else:
                                    combined_c["Reserve Cubic Feet"] = combined_c["Reserve Cubic Feet"].fillna(0)
                                if "Forward Cubic Feet" not in combined_c:
                                    combined_c["Forward Cubic Feet"] = 0.0
                                else:
                                    combined_c["Forward Cubic Feet"] = combined_c["Forward Cubic Feet"].fillna(0)
                                if "% Reserve Cubic Feet" not in combined_c:
                                    combined_c["% Reserve Cubic Feet"] = 0.0
                                else:
                                    combined_c["% Reserve Cubic Feet"] = combined_c["% Reserve Cubic Feet"].fillna(0)
                                if "% Forward Cubic Feet" not in combined_c:
                                    combined_c["% Forward Cubic Feet"] = 0.0
                                else:
                                    combined_c["% Forward Cubic Feet"] = combined_c["% Forward Cubic Feet"].fillna(0)
                                st.altair_chart(
                                    _sku_forward_reserve_percent_chart(
                                        combined_c,
                                        f"SKU Allocation (Cubic Ft) - {selected_cat}",
                                    ),
                                    width="stretch",
                                )
                                display_cols_order_cubic = [
                                    "Category",
                                    "Priority",
                                    "Movement",
                                    "Size",
                                    "Location",
                                    "Storage Type",
                                    "Current Cubic Feet",
                                    "Current % Cubic Feet",
                                    "Forward Cubic Feet",
                                    "% Forward Cubic Feet",
                                    "Reserve Cubic Feet",
                                    "% Reserve Cubic Feet",
                                    "Total Cubic Feet",
                                ]
                                display_cols_cubic = [c for c in display_cols_order_cubic if c in combined_c.columns]
                                display_cols_cubic += [c for c in combined_c.columns if c not in display_cols_cubic]
                                st.dataframe(combined_c[display_cols_cubic].reset_index(drop=True), width="stretch")
