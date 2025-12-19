import io
from pathlib import Path
from typing import List, Dict, Any
import streamlit as st
import pandas as pd

from pkg.validate import load_schema, required_cols
from pkg.db import _exec_schema, get_con, fetch_tech_map, upsert_tech_map_entry, delete_tech_map_entry
from pkg import etl_loaders

# your real preprocessors
from pkg.preprocess_mc46 import preprocess_mc46
from pkg.preprocess_cost_center import preprocess_cost_center
from pkg.preprocess_lx03 import preprocess_lx03
from pkg.preprocess_consumption import preprocess_consumption

ALLOWED_TYPES = ["csv", "xlsx", "xls"]

def _duckdb_tables_exist(names: list[str], require_rows: bool = False) -> bool:
    """
    Return True if ALL named tables exist in DuckDB (and optionally have >0 rows).
    This makes the UI resilient across reruns and new sessions.
    """
    con = get_con()
    # Check existence
    placeholders = ",".join(["?"] * len(names))
    exists = set(
        r[0].lower()
        for r in con.execute(
            f"""
            SELECT table_name
            FROM information_schema.tables
            WHERE lower(table_schema) IN ('main', 'temp') 
              AND lower(table_name) IN ({placeholders})
            """,
            names,
        ).fetchall()
    )
    if not all(n.lower() in exists for n in names):
        return False

    if not require_rows:
        return True

    # Optionally require at least one row in each table
    for n in names:
        cnt = con.execute(f"SELECT COUNT(*) FROM {n}").fetchone()[0]
        if int(cnt) == 0:
            return False
    return True

# ---------- session helpers ----------
def _ensure_state():
    # Initialize session_state keys used across uploads/validation/builds.
    if "uploads" not in st.session_state:
        st.session_state.uploads = {}
    if "validation" not in st.session_state:
        st.session_state.validation = {}
    if "built" not in st.session_state:
        # tracks which datasets have been loaded into DuckDB this session
        st.session_state.built = {"lx03": False, "mc46": False, "cost_center": False}
    if "schema_inited" not in st.session_state:
        st.session_state.schema_inited = False
    if "tech_map_confirmed" not in st.session_state:
        st.session_state.tech_map_confirmed = False

def _store_file(slot: str, file):
    # Persist uploaded file bytes in session_state; clear state when file is removed.
    if file is None:
        st.session_state.uploads.pop(slot, None)
        st.session_state.validation.pop(slot, None)
        st.session_state.built[slot] = False
        return
    data = file.getvalue() if hasattr(file, "getvalue") else file.read()
    st.session_state.uploads[slot] = {
        "name": file.name,
        "bytes": data,
        "ext": Path(file.name).suffix.lower().lstrip("."),
    }
    # new upload invalidates previous built flag
    st.session_state.built[slot] = False

def _read_df(slot: str, dtypes: Dict[str, str] | None = None) -> pd.DataFrame:
    # Read uploaded file for slot into a DataFrame, honoring optional dtype hints.
    info = st.session_state.uploads[slot]
    ext = info["ext"]
    buf = io.BytesIO(info["bytes"])

    dtype_map = None
    if dtypes:
        dtype_map = {}
        for col, dtype in dtypes.items():
            norm = dtype.lower()
            if norm.startswith(("int", "float")):
                dtype_map[col] = "string"
            else:
                dtype_map[col] = dtype

    if ext == "csv":
        return pd.read_csv(buf, dtype=dtype_map)
    elif ext in {"xlsx", "xls"}:
        return pd.read_excel(buf, dtype=dtype_map)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

# ---------- header validation ----------
def _peek_columns(file_bytes: bytes, ext: str) -> List[str]:
    # Read only headers from CSV/XLSX to validate required columns quickly.
    buf = io.BytesIO(file_bytes)
    if ext == "csv":
        df = pd.read_csv(buf, nrows=0)
    elif ext in {"xlsx", "xls"}:
        try:
            df = pd.read_excel(buf, nrows=0)
        except TypeError:
            df = pd.read_excel(buf); df = df.iloc[0:0]
    else:
        return []
    return list(df.columns)

def _normalize(col: str) -> str:
    # Normalize column names for case/spacing-insensitive comparison.
    return " ".join(str(col).strip().split()).lower()

def _validate_headers(slot: str, required: List[str]) -> Dict:
    # Compare uploaded columns to required set and report missing/present for UI.
    info = st.session_state.uploads.get(slot)
    if not info:
        return {"ok": False, "missing": required, "present": [], "original_cols": []}
    cols_original = _peek_columns(info["bytes"], info["ext"])
    cols_norm = {_normalize(c): c for c in cols_original}
    req_norm = [_normalize(c) for c in required]
    missing, present = [], []
    for r_raw, r in zip(required, req_norm):
        if r in cols_norm: present.append(cols_norm[r])
        else:              missing.append(r_raw)
    ok = len(missing) == 0
    return {"ok": ok, "missing": missing, "present": present, "original_cols": cols_original}

def _render_required_cols_box(title: str, required_cols_list: List[str]):
    # Helper to display the required column list for a dataset.
    with st.container(border=True):
        st.caption(f"Required Columns for {title}")
        st.markdown("\n".join([f"- `{c}`" for c in required_cols_list]) or "_(none listed)_")

def _render_validation_result(title: str, result: Dict):
    if result["ok"]:
        st.success(f"{title}: ✅ Headers look good.")
        if result["original_cols"]:
            with st.expander("Show detected columns"):
                st.code("\n".join(result["original_cols"]))
    else:
        st.error(f"{title}: ❌ Missing required columns.")
        st.markdown("**Missing:** " + ", ".join(f"`{c}`" for c in result["missing"]))
        if result["original_cols"]:
            with st.expander("Show detected columns"):
                st.code("\n".join(result["original_cols"]))

# ---------- build helpers ----------
def _ensure_schema():
    # Lazily create schema once per session using models/schema.sql.
    if not st.session_state.schema_inited:
        con = get_con()
        _exec_schema(con)  # creates tables per models/schema.sql
        st.session_state.schema_inited = True

def _preview_table(sql: str, title: str, limit: int = 50):
    # Convenience preview for a small slice of a table.
    con = get_con()
    df = con.execute(f"{sql} LIMIT {limit}").df()
    st.markdown(f"**{title} (top {min(limit, len(df))})**")
    st.dataframe(df, width='stretch')

def _fetch_tech_choices() -> List[str]:
    # Read tech list to populate mapping dropdown; fallback defaults exist higher up.
    con = get_con()
    rows = con.execute("SELECT tech_name FROM tech ORDER BY tech_name").fetchall()
    base = [row[0].upper() for row in rows]
    return base or ["ASSEMBLY", "BODY", "PAINT", "UNK"]

def _render_tech_map_section():
    # UI for confirming/editing MC46 technology mapping prior to processing MC46 uploads.
    _ensure_schema()
    st.markdown("#### Confirm MC46 technology mapping")
    st.caption(
        "We use this map to translate the characters present in column `Ind. Std Desc.` "
        "into production-line technology groups for each material."
    )
    df = fetch_tech_map()
    mapping_lookup = {row.value.upper(): row.tech_name.upper() for _, row in df.iterrows()}
    if df.empty:
        st.warning("No mapping rows found. Add at least one entry before uploading MC46.")
    else:
        st.dataframe(df.sort_values("value"), width='stretch')

    with st.expander("Modify mapping"):
        selected = None
        options = ["New mapping"] + [row.value for _, row in df.iterrows()]
        choice = st.selectbox("Select entry", options, key="tech_map_select")
        if choice != "New mapping":
            selected = df.loc[df["value"] == choice].iloc[0]

        with st.form("tech_map_upload_form", border=False):
            value_raw = st.text_input(
                "Raw value",
                value=selected.value if (selected is not None) else "",
                help="Example: A, BODY, LC",
            )
            value = value_raw.strip()
            tech_choices = _fetch_tech_choices()
            if "UNK" not in tech_choices:
                tech_choices.append("UNK")
            current = None
            if selected is not None and isinstance(selected.tech_name, str):
                current = selected.tech_name.upper()
            elif value:
                current = mapping_lookup.get(value.upper())
            default_idx = tech_choices.index(current) if current and current in tech_choices else 0
            tech_name = st.selectbox(
                "Maps to technology",
                options=tech_choices,
                index=default_idx,
            )
            cols = st.columns([1, 1])
            save_clicked = cols[0].form_submit_button("Save mapping", width='stretch', type="secondary")
            delete_clicked = cols[1].form_submit_button(
                "Delete mapping", width='stretch', disabled=(value == ""), type="secondary"
            )
        if save_clicked:
            if not value:
                st.error("Value is required.")
            else:
                upsert_tech_map_entry(value.upper(), tech_name.upper())
                st.session_state.tech_map_confirmed = False
                st.success(f"Saved mapping for {value.upper()}.")
                st.rerun()
        if delete_clicked and value:
            delete_tech_map_entry(value.upper())
            st.session_state.tech_map_confirmed = False
            st.success(f"Deleted mapping for {value.upper()}.")
            st.rerun()

    confirmed = st.checkbox(
        "I confirm this mapping before uploading MC46",
        value=st.session_state.get("tech_map_confirmed", False),
        key="tech_map_confirm_checkbox",
    )
    st.session_state.tech_map_confirmed = confirmed

def _upload_build_LX03(lx03_req_out: List[str], schema: Dict[str, Any]):
    # Read LX03 upload, preprocess, validate required columns, then load into DuckDB.
    with st.status("Building LX03 → lx03 …", expanded=True) as status:
        st.write("Reading file …")

        if "dtypes" in schema:
            dtypes = schema["dtypes"]
            df_raw = _read_df("lx03", dtypes)
        else:
            df_raw = _read_df("lx03")
        
        st.write("Preprocessing …")
        df_out = preprocess_lx03(df_raw.copy())
        missing = [c for c in lx03_req_out if c not in df_out.columns]
        if missing:
            raise ValueError(f"LX03 preprocess output missing required columns: {missing}")

        # st.write("Initializing schema (if needed) …")
        _ensure_schema()

        st.write("Loading …")
        upserted = etl_loaders.load_lx03_seed(df_out)
        # st.write(f"Upserted {upserted} parts rows.")

        status.update(label="LX03 build complete ✅", state="complete")
    # show cost_center preview
    # _preview_table("SELECT * FROM parts", "parts")

def _upload_build_mc46(mc46_req_out: List[str], schema: Dict[str, Any]):
    # Read MC46 upload, preprocess, and update existing parts seeded by LX03.
    with st.status("Building MC46 → parts …", expanded=True) as status:
        st.write("Reading file …")
        if "dtypes" in schema:
            dtypes = schema["dtypes"]
            df_raw = _read_df("mc46", dtypes)
        else:
            df_raw = _read_df("mc46")
        
        st.write("Preprocessing …")
        df_out = preprocess_mc46(df_raw.copy())

        # (Optional) ensure required output cols
        missing = [c for c in mc46_req_out if c not in df_out.columns]
        if missing:
            raise ValueError(f"MC46 preprocess output missing required columns: {missing}")

        # st.write("Initializing schema (if needed) …")
        _ensure_schema()

        st.write("Loading …")
        upserted = etl_loaders.load_mc46_seed(df_out)
        # st.write(f"Upserted {upserted} parts rows.")

        status.update(label="MC46 build complete ✅", state="complete")

    # Show previews
    # _preview_table("SELECT * FROM parts", "parts")

    # NEW: preview part_users if it exists
    # try:
    #     if _duckdb_tables_exist(["part_users"]):
    #         _preview_table("SELECT * FROM part_users", "part_users")
    # except Exception:
    #     pass

    # Mark built and refresh UI if you prefer
    st.session_state.built["mc46"] = True

def _upload_build_cost_center(cc_req_out: List[str], schema: Dict[str, Any]):
    # Read cost center upload, preprocess, and replace cost_center table rows.
    with st.status("Building Cost Center → cost_center …", expanded=True) as status:
        st.write("Reading file …")

        if "dtypes" in schema:
            dtypes = schema["dtypes"]
            df_raw = _read_df("cost_center", dtypes)
        else:
            df_raw = _read_df("cost_center")
       
        st.write("Preprocessing …")
        df_out = preprocess_cost_center(df_raw.copy())
        missing = [c for c in cc_req_out if c not in df_out.columns]
        if missing:
            raise ValueError(f"Cost Center preprocess output missing required columns: {missing}")

        # st.write("Initializing schema (if needed) …")
        _ensure_schema()

        st.write("Loading …")
        upserted = etl_loaders.load_cost_center(df_out)
        # st.write(f"Upserted {upserted} cost center rows.")

        status.update(label="Cost Center build complete ✅", state="complete")
    # show cost_center preview
    # _preview_table("SELECT * FROM cost_center", "cost_center")



# ---------- page ----------
def render():
    # Step-by-step upload/builder flow for LX03, MC46, Cost Center, and Consumption.
    st.header("Upload Files")
    _ensure_state()

    st.caption("Upload each file, then use **Build** to preprocess and load into app database.")

    # Input schemas (for header checks)
    mc46_in  = load_schema("mc46")
    cc_in = load_schema("cost_center")
    lx03_in = load_schema("lx03")
    cons_in = load_schema("consumption")
    mc46_req_in, cc_req_in, lx03_req_in, cons_req_in = required_cols(mc46_in), required_cols(cc_in), required_cols(lx03_in), required_cols(cons_in)

    # Output schemas (to ensure preprocessors produced expected shape)
    mc46_out = load_schema("mc46_output_schema.json")
    cc_out   = load_schema("cost_center_output_schema.json")
    lx03_out = load_schema("lx03_output_schema.json")
    material_to_priority_out = load_schema("material_to_priority")
    material_to_movement_out = load_schema("material_to_movement")
    material_to_tech_out = load_schema("material_to_tech")
    mc46_req_out, cc_req_out, lx03_req_out, mat_prio_req_out, mat_move_req_out, mat_tech_req_out = required_cols(mc46_out), required_cols(cc_out), required_cols(lx03_out), required_cols(material_to_priority_out), required_cols(material_to_movement_out), required_cols(material_to_tech_out)

    col1, col2 = st.columns(2)
    # ----- LX03 -----
    with col1:
        with st.container(border=True):
            st.subheader("1) LX03")
            f_lx03 = st.file_uploader("Upload LX03 file", type=ALLOWED_TYPES, key="f_lx03", label_visibility="collapsed")
            _store_file("lx03", f_lx03)
            if "lx03" in st.session_state.uploads:
                st.success(f"Loaded: {st.session_state.uploads['lx03']['name']}")
            _render_required_cols_box("LX03", lx03_req_in)
            if "lx03" in st.session_state.uploads:
                res = _validate_headers("lx03", lx03_req_in)
                st.session_state.validation["lx03"] = res
                _render_validation_result("LX03", res)

                can_build = res.get("ok", False)
                
                if st.button("Build LX03", width="stretch", disabled=not can_build, type="secondary"):
                    try:
                        _upload_build_LX03(lx03_req_out, lx03_in)
                        st.session_state.built["lx03"] = True
                    except Exception as e:
                        st.error(f"LX03 build failed: {e}")

            if st.session_state.built.get("lx03"):
                st.info("LX03 already built this session.")
                # with st.expander("Preview lx03"):
                #     _preview_table("SELECT * FROM parts", "parts")

    # ----- MC46 -----
    with col2:
        with st.container(border=True):
            st.subheader("2) MC46")

            parts_ready = _duckdb_tables_exist(["parts"], require_rows=True)

            if not parts_ready:
                st.info(
                    "MC46 upload depends on LX03 upload. "
                    "Please **Build LX03** above first."
                )
            else: 
                _render_tech_map_section()
                if not st.session_state.get("tech_map_confirmed"):
                    st.warning("Please confirm the technology mapping before uploading MC46.")
                f_mc46 = st.file_uploader("Upload MC46 file", type=ALLOWED_TYPES, key="f_mc46", label_visibility="collapsed")
                _store_file("mc46", f_mc46)
                if "mc46" in st.session_state.uploads:
                    st.success(f"Loaded: {st.session_state.uploads['mc46']['name']}")
                _render_required_cols_box("MC46", mc46_req_in)
                if "mc46" in st.session_state.uploads:
                    res = _validate_headers("mc46", mc46_req_in)
                    st.session_state.validation["mc46"] = res
                    _render_validation_result("MC46", res)

                    can_build = res.get("ok", False) and st.session_state.get("tech_map_confirmed", False)
                    
                    if st.button("Build MC46", width="stretch", disabled=not can_build, type="secondary"):
                        try:
                            _upload_build_mc46(mc46_req_out, mc46_in)
                            st.session_state.built["mc46"] = True
                        except Exception as e:
                            st.error(f"MC46 build failed: {e}")

                # If already built this session, show a quick peek again
                if st.session_state.built.get("mc46"):
                    st.info("MC46 already built this session.")
                    # with st.expander("Preview parts"):
                    #     _preview_table("SELECT * FROM parts", "parts")

    col3, col4 = st.columns(2)
    # ----- Cost Center -----
    with col3:
        with st.container(border=True):
            st.subheader("3) Cost Center")

            parts_ready = _duckdb_tables_exist(["parts"], require_rows=True)

            if not parts_ready:
                st.info(
                    "MC46 upload depends on LX03 upload. "
                    "Please run **Build LX03** above first."
                )
            else: 
                f_cc = st.file_uploader("Upload Cost Center file", type=ALLOWED_TYPES, key="f_cc", label_visibility="collapsed")
                _store_file("cost_center", f_cc)
                if "cost_center" in st.session_state.uploads:
                    st.success(f"Loaded: {st.session_state.uploads['cost_center']['name']}")
                _render_required_cols_box("Cost Center", cc_req_in)
                if "cost_center" in st.session_state.uploads:
                    res = _validate_headers("cost_center", cc_req_in)
                    st.session_state.validation["cost_center"] = res
                    _render_validation_result("Cost Center", res)

                    can_build = res.get("ok", False)
                    
                    if st.button("Build Cost Center", width="stretch", disabled=not can_build, type="secondary"):
                        try:
                            _upload_build_cost_center(cc_req_out, cc_in)
                            st.session_state.built["cost_center"] = True
                        except Exception as e:
                            st.error(f"Cost Center build failed: {e}")

                if st.session_state.built.get("cost_center"):
                    st.info("Cost Center already built this session.")
                    # with st.expander("Preview cost_center"):
                    #     _preview_table("SELECT * FROM cost_center", "cost_center")

    # ----- Consumption -----
    with col4:
        with st.container(border=True):
            st.subheader("4) Consumption")

            # Require BOTH MC46 & Cost Center to be built first
            # Set require_rows=True if you want them to have at least 1 row
            deps_ready = _duckdb_tables_exist(["parts", "cost_center"], require_rows=True)

            if not deps_ready:
                st.info(
                    "Consumption depends on **both** MC46 and Cost Center being built. "
                    "Please run **Build MC46** and **Build Cost Center** above first."
                )
            else:
                # Input + Output schema lists assumed defined earlier as cons_req_in / cons_req_out
                f_cons = st.file_uploader(
                    "Upload Consumption file",
                    type=ALLOWED_TYPES,
                    key="f_cons",
                    label_visibility="collapsed",
                    help="Enabled only after MC46 and Cost Center are built."
                )
                _store_file("consumption", f_cons)

                if "consumption" in st.session_state.uploads:
                    st.success(f"Loaded: {st.session_state.uploads['consumption']['name']}")

                _render_required_cols_box("Consumption", cons_req_in)

                # Header check only when uploaded
                if "consumption" in st.session_state.uploads:
                    res = _validate_headers("consumption", cons_req_in)
                    st.session_state.validation["consumption"] = res
                    _render_validation_result("Consumption", res)

                # Button is enabled ONLY when deps_ready AND headers pass
                can_build_consumption = (
                    deps_ready and
                    st.session_state.validation.get("consumption", {}).get("ok", False)
                )

                
                run = st.button(
                    "Build Consumption",
                    type="secondary",
                    width="stretch",
                    disabled=not can_build_consumption
                )

                if run:
                    try:
                        with st.status("Building Consumption-derived updates …", expanded=True) as status:
                            # 1) Read
                            st.write("Reading file …")
                            df_raw = _read_df("consumption")
                            # 2) Preprocess (now safe to hit parts/cost_center in DuckDB)
                            st.write("Preprocessing Consumption …")
                            mat_to_prio, mat_to_move, mat_to_tech = preprocess_consumption(df_raw.copy())
                            # 3) (optional) Output schema check
                            for update_type in ["prio", "move", "tech"]:
                                if update_type == "prio":
                                    df = mat_to_prio
                                    req_out = mat_prio_req_out
                                elif update_type == "move":
                                    df = mat_to_move
                                    req_out = mat_move_req_out
                                elif update_type == "tech":
                                    df = mat_to_tech
                                    req_out = mat_tech_req_out
                                missing = [c for c in req_out if c not in df.columns]
                                if missing:
                                    raise ValueError(f"Derived {update_type} output missing required columns: {missing}")
                                
                            # 4) Apply ETL updates
                            st.write("Loading derived fields …")
                            etl_loaders.load_priority_updates(mat_to_prio)
                            #st.write("Applying derived Movement → parts …")
                            etl_loaders.load_movement_updates(mat_to_move)
                            #st.write("Applying derived Users → parts …")
                            etl_loaders.load_tech_updates(mat_to_tech)
                            
                            status.update(label="Consumption build complete ✅", state="complete")

                        # with st.expander("Preview parts"):
                        #     _preview_table("SELECT * FROM parts", "parts")

                        st.session_state.built["consumption"] = True

                    except Exception as e:
                        st.error(f"Consumption build failed: {e}")

    # Final step: move to materials review once required tables exist
    ready_for_review = _duckdb_tables_exist(["parts", "cost_center"], require_rows=True)
    confirm = st.button(
        "Step 4: Review materials table",
        type="primary",
        disabled=not ready_for_review,
        width="stretch"
    )
    if confirm:
        try:
            etl_loaders.normalize_part_priorities()
            # Build and cache i_sku for the review page
            st.session_state["i_sku"], st.session_state["i_sku_user"], st.session_state["i_sku_type"] = (
                etl_loaders.save_i_sku_tables()
            )
        except Exception as exc:
            st.error(f"Failed to prepare priorities for review: {exc}")
        else:
            st.session_state["page"] = "review"
            st.rerun()
