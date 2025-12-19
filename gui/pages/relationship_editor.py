import streamlit as st
from pkg import db


def _render_table(df, column_config):
    # Helper to present tables with friendly column names and without indexes.
    if df.empty:
        st.info("No records yet.")
        return
    cols = [name for name, _ in column_config]
    renamed = dict(column_config)
    display_df = df[cols].rename(columns=renamed).reset_index(drop=True)
    st.dataframe(display_df, width='stretch')


def fetch_location_storage(con):
    # Current mapping of locations -> storage types (with codes/units).
    return con.execute(
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
    ).fetchdf()


def fetch_tech_usage(con):
    # Existing tech-to-location compatibility rows.
    return con.execute(
        """
        SELECT tlu.tech_id,
               tech.tech_name,
               tlu.location_id,
               sl.location
        FROM tech_location_usage tlu
        LEFT JOIN tech ON tlu.tech_id = tech.tech_id
        LEFT JOIN storage_location sl ON tlu.location_id = sl.location_id
        ORDER BY tlu.tech_id, tlu.location_id
        """
    ).fetchdf()


def fetch_tech_options(con):
    # Lookup list of techs for multiselect controls.
    return con.execute(
        """
        SELECT tech_id, tech_name
        FROM tech
        ORDER BY tech_name
        """
    ).fetchdf()


def fetch_location_tech_map(con, location_id):
    return con.execute(
        """
        SELECT tech_id
        FROM tech_location_usage
        WHERE location_id = ?
        """,
        [location_id],
    ).fetchdf()


def save_location_tech(con, location_id, tech_ids):
    con.execute("BEGIN")
    try:
        con.execute("DELETE FROM tech_location_usage WHERE location_id = ?", [location_id])
        if tech_ids:
            con.executemany(
                "INSERT INTO tech_location_usage (tech_id, location_id) VALUES (?, ?)",
                [(tech_id, location_id) for tech_id in tech_ids],
            )
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise


def fetch_location_options(con):
    # Lookup list of locations for multiselect controls.
    return con.execute(
        """
        SELECT location_id, location
        FROM storage_location
        ORDER BY location
        """
    ).fetchdf()


def fetch_storage_type_options(con):
    # Lookup list of storage types for multiselect controls.
    return con.execute(
        """
        SELECT type_id, type
        FROM storage_type
        ORDER BY type
        """
    ).fetchdf()


def fetch_size_options(con):
    # Size dimension lookup for type-size compatibility editor.
    return con.execute(
        """
        SELECT s.size_id, 
               s.size
        FROM size s
        ORDER BY s.size_id
        """
    ).fetchdf()


def fetch_location_storage_map(con, location_id):
    # Storage types currently mapped to a specific location.
    return con.execute(
        """
        SELECT type_id, code, type_current_units
        FROM location_storage_type
        WHERE location_id = ?
        """,
        [location_id],
    ).fetchdf()


def save_location_storage(con, location_id, rows):
    # Replace storage types for a location atomically.
    con.execute("BEGIN")
    try:
        con.execute("DELETE FROM location_storage_type WHERE location_id = ?", [location_id])
        if rows:
            con.executemany(
                """
                INSERT INTO location_storage_type (location_id, type_id, code, type_current_units)
                VALUES (?, ?, ?, ?)
                """,
                rows,
            )
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise

def fetch_type_size_compat(con):
    # Full compatibility matrix (type -> size).
    return con.execute(
        """
        SELECT tsc.type_id,
               st.type,
               tsc.size_id,
               s.size,
               tsc.max_sku_per_unit
        FROM type_size_compat tsc
        LEFT JOIN storage_type st ON tsc.type_id = st.type_id
        LEFT JOIN size s ON tsc.size_id = s.size_id
        ORDER BY tsc.type_id, tsc.size_id
        """
    ).fetchdf()

def fetch_type_size_map(con, type_id):
    # Size compatibility for a specific storage type.
    return con.execute(
        """
        SELECT tsc.size_id,
               s.size,
               tsc.max_sku_per_unit
        FROM type_size_compat tsc
        LEFT JOIN size s ON tsc.size_id = s.size_id
        WHERE tsc.type_id = ?
        ORDER BY tsc.size_id
        """,
        [type_id],
    ).fetchdf()


def save_type_size_map(con, type_id, rows):
    # Replace compatibility rows for one storage type.
    con.execute("BEGIN")
    try:
        con.execute("DELETE FROM type_size_compat WHERE type_id = ?", [type_id])
        if rows:
            con.executemany(
                """
                INSERT INTO type_size_compat (type_id, size_id, max_sku_per_unit)
                VALUES (?, ?, ?)
                """,
                rows,
            )
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise


def render() -> None:
    st.title("Location Usage")
    st.caption(
        "Review and adjust the relationships between locations, storage types, and techs. "
        "These tables are generated when you confirm the base parameters."
    )

    con = db.get_con()
    # summary = st.session_state.get("relationship_summary")
    # if summary:
    #     with st.expander("Most recent build", expanded=False):
    #         for table_name, result in summary:
    #             st.write(f"{table_name}: {result}")

    tab_loc_types, tab_tech_usage, tab_type_sizes = st.tabs(
        ["Location -> Storage Types", "Technology -> Location", "Type -> Size"]
    )

    with tab_loc_types:
        loc_df = fetch_location_storage(con)
        # Overview of current location -> storage type mappings.
        _render_table(
            loc_df,
            [
                ("location", "Location"),
                ("type", "Storage Type"),
                ("code", "Code"),
                ("type_current_units", "Current Units"),
            ],
        )

        location_options_df = fetch_location_options(con)
        type_options_df = fetch_storage_type_options(con)

        if location_options_df.empty or type_options_df.empty:
            st.info("Define locations and storage types before editing this table.")
        else:
            location_label_to_id = {row.location: int(row.location_id) for _, row in location_options_df.iterrows()}
            location_labels = list(location_label_to_id.keys())
            default_loc = st.session_state.get("relationship_location_choice", location_labels[0])
            if default_loc not in location_labels and isinstance(default_loc, str) and " - " in default_loc:
                legacy_label = default_loc.split(" - ", 1)[1]
                if legacy_label in location_labels:
                    default_loc = legacy_label
            if default_loc not in location_labels:
                default_loc = location_labels[0]
            loc_index = location_labels.index(default_loc)
            location_choice = st.selectbox("Select a location", location_labels, index=loc_index)
            st.session_state["relationship_location_choice"] = location_choice
            location_id = location_label_to_id[location_choice]
            location_name = location_choice

            existing_df = fetch_location_storage_map(con, location_id)
            existing_map = {
                int(row.type_id): {"code": row.code or "", "units": row.type_current_units or 0}
                for _, row in existing_df.iterrows()
            }

            type_label_to_id = {row.type: int(row.type_id) for _, row in type_options_df.iterrows()}
            type_labels = list(type_label_to_id.keys())
            type_lookup = {int(row.type_id): row.type for _, row in type_options_df.iterrows()}
            default_selected_labels = [
                label for label in type_labels if type_label_to_id[label] in existing_map
            ]

            st.markdown(f"#### Storage types in {location_name}")
            with st.form(f"location_storage_form_{location_id}"):
                selected_labels = st.multiselect(
                    "Select storage types",
                    type_labels,
                    default=default_selected_labels,
                    key=f"location_storage_select_{location_id}",
                )

                selected_type_ids = [type_label_to_id[label] for label in selected_labels]

                code_inputs = {}
                unit_inputs = {}
                for type_id in selected_type_ids:
                    cols = st.columns([2, 1])
                    code_inputs[type_id] = cols[0].text_input(
                        f"Code for {type_lookup[type_id]}",
                        value=existing_map.get(type_id, {}).get("code", ""),
                        key=f"code_{location_id}_{type_id}",
                    )
                    unit_inputs[type_id] = cols[1].number_input(
                        f"Current units for {type_lookup[type_id]}",
                        min_value=0,
                        value=int(existing_map.get(type_id, {}).get("units", 0)),
                        step=1,
                        key=f"units_{location_id}_{type_id}",
                    )

                submitted = st.form_submit_button("Save storage types for location")

            if submitted:
                rows = []
                has_error = False
                for type_id in selected_type_ids:
                    code = code_inputs[type_id].strip()
                    units = unit_inputs[type_id]
                    if not code:
                        st.error(f"Code is required for {type_lookup[type_id]}.")
                        has_error = True
                    rows.append((location_id, type_id, code, int(units)))

                if not has_error:
                    try:
                        save_location_storage(con, location_id, rows)
                        st.success(f"Updated storage types for {location_name}.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Failed to update location storage types: {exc}")

    with tab_tech_usage:
        tech_df = fetch_tech_usage(con)
        # Overview of tech -> location compatibility rows.
        _render_table(
            tech_df,
            [
                ("tech_name", "Tech"),
                ("location", "Location"),
            ],
        )

        tech_options_df = fetch_tech_options(con)

        if location_options_df.empty or tech_options_df.empty:
            st.info("Define locations and techs before editing this table.")
        else:
            location_label_to_id = {row.location: int(row.location_id) for _, row in location_options_df.iterrows()}
            location_labels = list(location_label_to_id.keys())
            default_loc = st.session_state.get("relationship_location_choice_tech", location_labels[0])
            if default_loc not in location_labels and isinstance(default_loc, str) and " - " in default_loc:
                legacy_label = default_loc.split(" - ", 1)[1]
                if legacy_label in location_labels:
                    default_loc = legacy_label
            if default_loc not in location_labels:
                default_loc = location_labels[0]
            loc_index = location_labels.index(default_loc)
            location_choice = st.selectbox("Select a location", location_labels, index=loc_index, key="tech_location_select")
            st.session_state["relationship_location_choice_tech"] = location_choice
            location_id = location_label_to_id[location_choice]
            location_name = location_choice

            existing_tech_df = fetch_location_tech_map(con, location_id)
            existing_tech_ids = set(int(row.tech_id) for _, row in existing_tech_df.iterrows())

            tech_label_to_id = {row.tech_name: int(row.tech_id) for _, row in tech_options_df.iterrows()}
            tech_labels = list(tech_label_to_id.keys())
            default_selected = [label for label in tech_labels if tech_label_to_id[label] in existing_tech_ids]

            st.markdown(f"#### Tech compatibility for {location_name}")
            with st.form(f"location_tech_form_{location_id}"):
                selected_labels = st.multiselect(
                    "Select techs allowed in this location",
                    tech_labels,
                    default=default_selected,
                    key=f"tech_multiselect_{location_id}",
                )
                submitted = st.form_submit_button("Save tech/location mapping")

            if submitted:
                tech_ids = [tech_label_to_id[label] for label in selected_labels]
                try:
                    save_location_tech(con, location_id, tech_ids)
                    st.success(f"Updated tech compatibility for {location_name}.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Failed to update tech/location mapping: {exc}")

    with tab_type_sizes:
        type_size_df = fetch_type_size_compat(con)
        # Overview of storage type -> size compatibility.
        _render_table(
            type_size_df,
            [
                ("type", "Storage Type"),
                ("size", "Size"),
                ("max_sku_per_unit", "Maximum number of SKUs"),
            ],
        )

        type_options = fetch_storage_type_options(con)
        size_options = fetch_size_options(con)

        if type_options.empty or size_options.empty:
            st.info("Define storage types and sizes before editing compatibility.")
        else:
            type_label_to_id = {row.type: int(row.type_id) for _, row in type_options.iterrows()}
            type_labels = list(type_label_to_id.keys())
            default_type = st.session_state.get("relationship_type_choice", type_labels[0])
            if default_type not in type_labels and isinstance(default_type, str) and " - " in default_type:
                legacy_label = default_type.split(" - ", 1)[1]
                if legacy_label in type_labels:
                    default_type = legacy_label
            if default_type not in type_labels:
                default_type = type_labels[0]
            type_index = type_labels.index(default_type)
            type_choice = st.selectbox("Select storage type", type_labels, index=type_index)
            st.session_state["relationship_type_choice"] = type_choice
            type_id = type_label_to_id[type_choice]
            type_name = type_choice

            compat_df = fetch_type_size_map(con, type_id)
            existing_map = {
                int(row.size_id): {
                    "size": row["size"] or "",
                    "max": row.max_sku_per_unit or 0.0,
                }
                for _, row in compat_df.iterrows()
            }

            size_label_to_id = {row["size"]: int(row.size_id) for _, row in size_options.iterrows()}
            size_labels = list(size_label_to_id.keys())
            default_selected = [
                label for label in size_labels if size_label_to_id[label] in existing_map
            ]

            st.markdown(f"#### Size compatibility for {type_name}")
            with st.form(f"type_size_form_{type_id}"):
                selected_labels = st.multiselect(
                    f"Compatible sizes for {type_name}",
                    size_labels,
                    default=default_selected,
                    key=f"type_size_multiselect_{type_id}",
                )

                size_inputs = {}
                for label in selected_labels:
                    size_id = size_label_to_id[label]
                    cols = st.columns([3])
                    max_key = f"max_{type_id}_{size_id}"
                    max_default = float(existing_map.get(size_id, {}).get("max", 0.0))
                    max_val = cols[0].number_input(
                        f"Maximum SKUs ({label})",
                        min_value=0.0,
                        value=max_default,
                        key=max_key,
                    )
                    size_inputs[size_id] = max_val

                submitted = st.form_submit_button("Save compatible SKU sizes for storage type")

            if submitted:
                rows = []
                has_error = False
                for size_id, max_val in size_inputs.items():
                    rows.append((type_id, size_id, float(max_val)))
                if not rows:
                    st.error("Select at least one size.")
                elif not has_error:
                    try:
                        save_type_size_map(con, type_id, rows)
                        st.success(f"Updated compatible sizes for {type_name}.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Failed to update compatible storage type sizes: {exc}")


    st.markdown(
        "Once you select the below button and go to Step 3, you cannot return to edit plan inputs or mappings. The next step is to upload files"
    )
    confirm = st.button("Step 3: File Uploads", key="confirm_upload")
    if confirm:
        st.session_state["page"] = "upload"
        st.rerun()

    if st.button("Back"):
        st.session_state["page"] = "parameters"
        st.rerun()
