import streamlit as st
from pkg import db


def _render_table(df, column_config):
    """
    Helper to display a dataframe without index/ID columns.
    `column_config` is a list of (column_name, "Pretty Title").
    """
    if df.empty:
        st.info("No records yet.")
        return
    cols = [name for name, _ in column_config]
    renamed = dict(column_config)
    display_df = df[cols].rename(columns=renamed).reset_index(drop=True)
    st.dataframe(display_df, width='stretch')


# ---------- Storage types ----------
def fetch_storage_types(con):
    return con.execute(
        """
        SELECT type_id, type, sqft_req, buy_cost, buy_invest, reloc_cost,
               reloc_invest, cubic_capacity_per_unit
        FROM storage_type
        ORDER BY type_id
        """
    ).fetchdf()


def upsert_storage_type(con, payload):
    con.execute(
        """
        INSERT INTO storage_type (type_id, type, sqft_req, buy_cost, buy_invest, reloc_cost,
                                  reloc_invest, cubic_capacity_per_unit)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (type_id) DO UPDATE SET
            type = excluded.type,
            sqft_req = excluded.sqft_req,
            buy_cost = excluded.buy_cost,
            buy_invest = excluded.buy_invest,
            reloc_cost = excluded.reloc_cost,
            reloc_invest = excluded.reloc_invest,
            cubic_capacity_per_unit = excluded.cubic_capacity_per_unit
        """,
        [
            payload["type_id"],
            payload["type"],
            payload["sqft_req"],
            payload["buy_cost"],
            payload["buy_invest"],
            payload["reloc_cost"],
            payload["reloc_invest"],
            payload["cubic_capacity_per_unit"],
        ],
    )

def render_storage_type_editor(con) -> None:
    # Display existing storage types and allow add/edit in-place.
    df = fetch_storage_types(con)

    st.subheader("Storage Types")
    storage_type_columns = [
        ("type", "Type"),
        ("sqft_req", "Square Feet Required (per unit of storage typ)"),
        ("buy_cost", "Buy Cost"),
        ("buy_invest", "Buy Investment"),
        ("reloc_cost", "Relocation Cost"),
        ("reloc_invest", "Relocation Investment"),
        ("cubic_capacity_per_unit", "Cubic Capacity/Unit"),
    ]
    _render_table(df, storage_type_columns)

    st.markdown("### Add or modify a storage type")
    options = ["New storage type"] + [f"{row.type_id} - {row.type}" for _, row in df.iterrows()]
    choice = st.selectbox("Select entry", options, key="storage_type_select")

    # Pre-fill if editing
    selected = None
    next_id = int(df.type_id.max()) + 1 if not df.empty else 1
    if choice != "New storage type":
        selected_id = int(choice.split(" - ")[0])
        selected = df.loc[df["type_id"] == selected_id].iloc[0]

    with st.form("storage_type_form"):
        type_id = selected.type_id if selected is not None else next_id
        #st.markdown(f"**Type ID:** `{type_id}` (auto-assigned)" if selected is None else f"**Type ID:** `{type_id}`")

        type_name = st.text_input("Type name", value=selected.type if selected is not None else "")

        sqft_req = st.number_input("Sqft required", min_value=0.0, value=float(selected.sqft_req) if selected is not None and selected.sqft_req is not None else 0.0)

        buy_cost = st.number_input("Buy cost", min_value=0.0, value=float(selected.buy_cost) if selected is not None and selected.buy_cost is not None else 0.0)
        
        buy_invest = st.number_input("Buy invest", min_value=0.0, value=float(selected.buy_invest) if selected is not None and selected.buy_invest is not None else 0.0)
        
        reloc_cost = st.number_input("Relocation cost", min_value=0.0, value=float(selected.reloc_cost) if selected is not None and selected.reloc_cost is not None else 0.0)
        
        reloc_invest = st.number_input("Relocation invest", min_value=0.0, value=float(selected.reloc_invest) if selected is not None and selected.reloc_invest is not None else 0.0)
        
        cubic_capacity_per_unit = st.number_input(
            "Cubic capacity per unit",
            min_value=0.0,
            value=float(selected.cubic_capacity_per_unit) if selected is not None and selected.cubic_capacity_per_unit is not None else 0.0,
        )
        submitted = st.form_submit_button("Save")

    if submitted:
        if not type_name.strip():
            st.error("Type name is required.")
            return
        payload = {
            "type_id": int(type_id),
            "type": type_name.strip(),
            "sqft_req": float(sqft_req),
            "buy_cost": float(buy_cost),
            "buy_invest": float(buy_invest),
            "reloc_cost": float(reloc_cost),
            "reloc_invest": float(reloc_invest),
            "cubic_capacity_per_unit": float(cubic_capacity_per_unit),
        }
        db.clear_relationship_tables()
        st.session_state.pop("relationship_summary", None)
        try:
            upsert_storage_type(con, payload)
            st.success("Saved storage type. Relationship tables were cleared; rebuild them when you're ready.")
            st.rerun()
        except Exception as exc:
            st.error(f"Failed to save: {exc}")

# ---------- Techs ----------

def fetch_techs(con):
    return con.execute(
        """
        SELECT tech_id, tech_name
        FROM tech
        ORDER BY tech_id
        """
    ).fetchdf()


def upsert_tech(con, payload):
    con.execute(
        """
        INSERT INTO tech (tech_id, tech_name)
        VALUES (?, ?)
        ON CONFLICT (tech_id) DO UPDATE SET
            tech_name = excluded.tech_name
        """,
        [
            payload["tech_id"],
            payload["tech_name"],
        ],
    )


def render_tech_editor(con) -> None:
    # Manage tech list and ensure indicator columns exist when saving new techs.
    df = fetch_techs(con)

    st.subheader("Technologies")
    tech_columns = [
        ("tech_name", "Technology"),
    ]
    _render_table(df, tech_columns)

    st.markdown("### Add or modify a technology")
    options = ["New tech"] + [f"{row.tech_id} - {row.tech_name}" for _, row in df.iterrows()]
    choice = st.selectbox("Select entry", options, key="tech_select")

    selected = None
    next_id = int(df.tech_id.max()) + 1 if not df.empty else 1
    if choice != "New tech":
        selected_id = int(choice.split(" - ")[0])
        selected = df.loc[df["tech_id"] == selected_id].iloc[0]

    with st.form("tech_form"):
        tech_id = int(selected.tech_id) if selected is not None else next_id
        # st.number_input(
        #     "Tech ID (auto-assigned)",
        #     min_value=1,
        #     value=tech_id,
        #     step=1,
        #     disabled=True,
        # )

        tech_name = st.text_input(
            "Tech name (auto-capitalized on save)",
            value=selected.tech_name if selected is not None else "",
        )

        submitted = st.form_submit_button("Save technology")

    if submitted:
        if not tech_name.strip():
            st.error("Technology name is required.")
            return
        payload = {
            "tech_id": tech_id,
            "tech_name": tech_name.strip().upper(),
        }
        db.clear_relationship_tables()
        st.session_state.pop("relationship_summary", None)
        try:
            upsert_tech(con, payload)
            db.ensure_indicator_columns_for_tech(payload["tech_name"])
            st.success("Saved tech. Relationship tables were cleared; rebuild them when you're ready.")
            st.rerun()
        except Exception as exc:
            st.error(f"Failed to save technology: {exc}")


# ---------- System params ----------

def fetch_system_config(con):
    return con.execute(
        """
        SELECT team, line_down_cost, plan_budget, space_factor, reserve_cost
        FROM system
        ORDER BY team
        """
    ).fetchdf()


def upsert_system(con, payload):
    con.execute(
        """
        INSERT INTO system (team, line_down_cost, plan_budget, space_factor, reserve_cost)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT (team) DO UPDATE SET
            line_down_cost = excluded.line_down_cost,
            plan_budget = excluded.plan_budget,
            space_factor = excluded.space_factor,
            reserve_cost = excluded.reserve_cost
        """,
        [
            payload["team"],
            payload["line_down_cost"],
            payload["plan_budget"],
            payload["space_factor"],
            payload["reserve_cost"],
        ],
    )


def render_system_editor(con) -> None:
    # Single-row system parameters; editable via form.
    df = fetch_system_config(con)

    st.subheader("Overall plan parameters")
    system_columns = [
        ("line_down_cost", "Line-down Cost"),
        ("plan_budget", "Plan Budget"),
        ("space_factor", "Space Factor"),
        ("reserve_cost", "Reserve Cost"),
    ]
    _render_table(df, system_columns)

    if df.empty:
        st.info("No system configuration found. Please seed the database.")
        return

    st.markdown("### Update plan parameters")
    current = df.iloc[0]

    with st.form("system_form"):
        line_down_cost = st.number_input(
            "Line-down cost",
            min_value=0.0,
            value=float(current.line_down_cost) if current.line_down_cost is not None else 0.0,
        )
        plan_budget = st.number_input(
            "Plan budget",
            min_value=0.0,
            value=float(current.plan_budget) if current.plan_budget is not None else 0.0,
        )
        space_factor = st.number_input(
            "Space factor",
            min_value=0.0,
            value=float(current.space_factor) if current.space_factor is not None else 0.0,
        )
        reserve_cost = st.number_input(
            "Reserve cost",
            min_value=0.0,
            value=float(current.reserve_cost) if current.reserve_cost is not None else 0.0,
        )
        submitted = st.form_submit_button("Save plan parameters")

    if submitted:
        payload = {
            "team": current.team,
            "line_down_cost": float(line_down_cost),
            "plan_budget": float(plan_budget),
            "space_factor": float(space_factor),
            "reserve_cost": float(reserve_cost),
        }
        try:
            upsert_system(con, payload)
            st.success("Saved plan parameters.")
            st.rerun()
        except Exception as exc:
            st.error(f"Failed to save plan parameters: {exc}")


# ---------- Locations ----------

def fetch_locations(con):
    return con.execute(
        """
        SELECT location_id, location, floor_space, current_storage_floor_space, location_priority,
               travel_time, reserve_allowed
        FROM storage_location
        ORDER BY location_id
        """
    ).fetchdf()

def fetch_location_priorities(con):
    return con.execute(
        """
        SELECT location_priority, travel_time, reserve_allowed
        FROM location_priority
        ORDER BY location_priority
        """
    ).fetchdf()


def upsert_location(con, payload):
    # Check if this location_id already exists
    exists = con.execute(
        "SELECT COUNT(*) FROM storage_location WHERE location_id = ?",
        [payload["location_id"]],
    ).fetchone()[0]

    if exists:
        # Update non-key columns only (location_id stays the same)
        con.execute(
            """
            UPDATE storage_location
            SET location = ?,
                floor_space = ?,
                current_storage_floor_space = ?,
                location_priority = ?,
                travel_time = ?,
                reserve_allowed = ?
            WHERE location_id = ?
            """,
            [
                payload["location"],
                payload["floor_space"],
                payload["current_storage_floor_space"],
                payload["location_priority"],
                payload["travel_time"],
                payload["reserve_allowed"],
                payload["location_id"],
            ],
        )
    else:
        # Insert new location
        con.execute(
            """
            INSERT INTO storage_location (
                location_id, location, floor_space, current_storage_floor_space, location_priority,
                travel_time, reserve_allowed
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                payload["location_id"],
                payload["location"],
                payload["floor_space"],
                payload["current_storage_floor_space"],
                payload["location_priority"],
                payload["travel_time"],
                payload["reserve_allowed"],
            ],
        )



def render_location_editor(con) -> None:
    # Manage storage locations and their base attributes (priority drives travel/reserve flags).
    df = fetch_locations(con)
    df_prio = fetch_location_priorities(con)

    st.subheader("Locations")
    location_columns = [
        ("location", "Location"),
        ("floor_space", "Floor Space"),
        ("current_storage_floor_space", "Current Storage Floor Space"),
        ("location_priority", "Location Priority"),
    ]
    _render_table(df, location_columns)

    st.markdown("### Add or modify a location")
    options = ["New location"] + [f"{row.location_id} - {row.location}" for _, row in df.iterrows()]
    choice = st.selectbox("Select entry", options, key="location_select")

    selected = None
    next_id = int(df.location_id.max()) + 1 if not df.empty else 1
    if choice != "New location":
        selected_id = int(choice.split(" - ")[0])
        selected = df.loc[df["location_id"] == selected_id].iloc[0]

    # priority dropdown options
    if not df_prio.empty:
        prio_values = df_prio["location_priority"].tolist()
        default_prio = (
            int(selected.location_priority)
            if selected is not None and selected.location_priority is not None
            else prio_values[0]
        )
        prio_index = prio_values.index(default_prio) if default_prio in prio_values else 0
    else:
        prio_values = [1]
        prio_index = 0

    with st.form("location_form"):
        location_id = int(selected.location_id) if selected is not None else next_id
        # st.number_input(
        #     "Location ID (auto-assigned)",
        #     min_value=1,
        #     value=location_id,
        #     step=1,
        #     disabled=True,
        # )

        location_name = st.text_input(
            "Location name",
            value=selected.location if selected is not None else "",
        )

        floor_space = st.number_input(
            "Floor space (sqft)",
            min_value=0,
            value=int(selected.floor_space) if selected is not None and selected.floor_space is not None else 0,
            step=100,
        )

        current_storage_floor_space = st.number_input(
            "Current storage floor space (sqft)",
            min_value=0,
            value=int(selected.current_storage_floor_space)
            if selected is not None and selected.current_storage_floor_space is not None
            else 0,
            step=100,
        )

        location_priority = st.selectbox(
            "Location priority",
            options=prio_values,
            index=prio_index,
            help="Priority bucket; travel time and reserve_allowed are derived from this.",
        )

        # Show derived values read-only for user awareness
        prio_row = df_prio.loc[df_prio["location_priority"] == int(location_priority)].iloc[0]
        st.caption(
            f"For priority {int(location_priority)}: "
            f"travel_time={int(prio_row.travel_time)}, "
            f"reserve_allowed={bool(prio_row.reserve_allowed)}"
        )

        submitted = st.form_submit_button("Save location")

    if submitted:
        if not location_name.strip():
            st.error("Location name is required.")
            return

        # Auto-populate from location_priority table
        prio_row = df_prio.loc[df_prio["location_priority"] == int(location_priority)].iloc[0]
        travel_time = int(prio_row.travel_time)
        reserve_allowed = bool(prio_row.reserve_allowed)

        payload = {
            "location_id": location_id,
            "location": location_name.strip().upper(),
            "floor_space": int(floor_space),
            "current_storage_floor_space": int(current_storage_floor_space),
            "location_priority": int(location_priority),
            "travel_time": travel_time,
            "reserve_allowed": reserve_allowed,
        }
        db.clear_relationship_tables()
        st.session_state.pop("relationship_summary", None)
        try:
            upsert_location(con, payload)
            st.success("Saved location. Relationship tables were cleared; rebuild them when you're ready.")
            st.rerun()
        except Exception as exc:
            st.error(f"Failed to save location: {exc}")


# ---------- Page entry ----------

def render() -> None:
    st.title("Plan Inputs")

    db_path = st.session_state.get("db_path")
    if not db_path:
        st.warning("No plan database is active. Start a new plan first.")
        if st.button("Back to landing"):
            st.session_state["page"] = "landing"
            st.rerun()
        return

    # st.success(f"Plan database: `{db_path}`")

    con = db.get_con()

    tab_types, tab_locations, tab_techs, tab_system = st.tabs(
        ["Storage types", "Locations", "Techs", "Plan parameters"]
    )

    with tab_types:
        render_storage_type_editor(con)

    with tab_locations:
        render_location_editor(con)

    with tab_techs:
        render_tech_editor(con)

    with tab_system:
        render_system_editor(con)

    st.divider()
    st.subheader("Confirm parameters")
    st.markdown(
        "Once storage types, locations, and techs look good, build the downstream relationship tables. "
        "You can re-run this step whenever you make changes; it truncates and rebuilds the tables each time."
    )

    if st.button("Step 2: Storage Type, Location, & Technology Mapping", type="primary"):
        with st.spinner("Building relationship tables..."):
            summary = db.build_relationship_tables()
        st.session_state["relationship_summary"] = summary
        st.success("Relationship tables rebuilt from the latest parameters.")
        st.session_state["page"] = "relationships"
        st.rerun()

    # rel_summary = st.session_state.get("relationship_summary")
    # if rel_summary:
    #     with st.expander("Latest relationship build results", expanded=False):
    #         for table_name, result in rel_summary:
    #             st.write(f"{table_name}: {result}")

    if st.button("Back"):
        st.session_state["page"] = "landing"
        st.rerun()
