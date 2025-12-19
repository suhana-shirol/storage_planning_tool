import streamlit as st
from pathlib import Path
from pkg import db


def render() -> None:
    with st.container():
        # Hero/header content for the landing page.
        st.title("Non-Series Materials Storage Planning Tool")
        # st.info("This application was made in collaboration with Georgia Tech Senior Design.")

        # Kick off a fresh plan: reset DB, seed lookup tables, then send user to parameters page.
        if st.button("Start new plan", type="primary", width='stretch'):
            with st.spinner("Getting things ready..."):
                try:
                    summary = db.reset_and_seed_db()
                except Exception as exc:  # pragma: no cover - UI surface
                    st.error(f"Initialization failed: {exc}")
                    return
            st.session_state["db_path"] = str(db.DB_PATH)
            st.session_state["seed_summary"] = summary
            st.session_state["page"] = "parameters"
            st.rerun()
