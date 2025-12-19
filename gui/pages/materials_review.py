import streamlit as st
import pandas as pd
from io import BytesIO
import subprocess
import sys
from pathlib import Path

from pkg.db import get_con
from pkg.db import close_con
from pkg.aggregate_parts import build_i_sku, build_i_sku_user, build_i_sku_type
from optimization.calculate_piecewise_params import generate_piecewise_params

ROOT = Path(__file__).resolve().parents[2]
PIECEWISE_PARAMS_PATH = ROOT / "optimization" / "piecewiseparams.json"
DECISION_VARS_PATH = ROOT / "optimization" / "non_zero_decision_variables.xlsx"


def _fetch_parts_df():
    # Pull the full parts table for review/export.
    con = get_con()
    return con.execute(
        """
        SELECT *
        FROM parts
        ORDER BY material
        """
    ).fetchdf()


def _download_buttons(label_csv: str, label_xlsx: str, df: pd.DataFrame, base_name: str) -> None:
    # Render paired CSV/XLSX download buttons for a provided DataFrame.
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label_csv,
        data=csv_bytes,
        file_name=f"{base_name}.csv",
        mime="text/csv",
        width="stretch",
    )

    xlsx_buf = BytesIO()
    df.to_excel(xlsx_buf, index=False)
    xlsx_buf.seek(0)
    st.download_button(
        label_xlsx,
        data=xlsx_buf,
        file_name=f"{base_name}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width="stretch",
    )


def render() -> None:
    st.header("Materials Review")
    st.caption("View and export the full `parts` table.")

    # Parts table: should be populated after LX03/MC46/consumption steps.
    try:
        parts_df = _fetch_parts_df()
    except Exception as exc:
        st.error(f"Failed to load parts: {exc}")
        return

    if parts_df.empty:
        st.info("No parts data available. Upload LX03/MC46 to populate this table.")
        return

    st.caption(f"Parts: {len(parts_df):,} rows")
    st.dataframe(parts_df, width="stretch")
    _download_buttons("Download parts as CSV", "Download parts as XLSX", parts_df, "parts")

    st.markdown("---")

    # i_sku table (built in Step 4)
    i_sku_df = st.session_state.get("i_sku")
    if i_sku_df is None:
        try:
            i_sku_df = build_i_sku()
        except Exception as exc:
            st.error(f"Failed to build i_sku: {exc}")
            return
    if i_sku_df.empty:
        st.info("i_sku is empty.")
        return

    st.caption(f"i_sku: {len(i_sku_df):,} rows")
    st.dataframe(i_sku_df, width="stretch")
    _download_buttons("Download i_sku as CSV", "Download i_sku as XLSX", i_sku_df, "i_sku")

    st.markdown("---")

    # i_sku_user table (built in Step 4)
    i_sku_user_df = st.session_state.get("i_sku_user")
    if i_sku_user_df is None:
        try:
            i_sku_user_df = build_i_sku_user()
        except Exception as exc:
            st.error(f"Failed to build i_sku_user: {exc}")
            return
    if i_sku_user_df.empty:
        st.info("i_sku_user is empty.")
        return

    st.caption(f"i_sku_user: {len(i_sku_user_df):,} rows")
    st.dataframe(i_sku_user_df, width="stretch")
    _download_buttons("Download i_sku_user as CSV", "Download i_sku_user as XLSX", i_sku_user_df, "i_sku_user")

    st.markdown("---")

    # i_sku_type table (built in Step 4)
    i_sku_type_df = st.session_state.get("i_sku_type")
    if i_sku_type_df is None:
        try:
            i_sku_type_df = build_i_sku_type()
        except Exception as exc:
            st.error(f"Failed to build i_sku_type: {exc}")
            return
    if i_sku_type_df.empty:
        st.info("i_sku_type is empty.")
        return

    st.caption(f"i_sku_type: {len(i_sku_type_df):,} rows")
    st.dataframe(i_sku_type_df, width="stretch")
    _download_buttons("Download i_sku_type as CSV", "Download i_sku_type as XLSX", i_sku_type_df, "i_sku_type")
    
    st.markdown("---")

    confirm = st.button("Run Model", type="primary", width="stretch")
    if confirm:
        with st.spinner("Running optimization model..."):
            try:
                # Build inputs for the optimizer, close DB to avoid locks, then run the algorithm.
                params = generate_piecewise_params(output_path=str(PIECEWISE_PARAMS_PATH))
                st.session_state["piecewise_params"] = params
                # Close shared DB connection to release file lock before subprocess runs
                close_con()

                subprocess.run(
                    [sys.executable, "-m", "optimization.algo"],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    check=True,
                )

                if not DECISION_VARS_PATH.exists():
                    raise FileNotFoundError(
                        f"Expected decision variables at {DECISION_VARS_PATH} but did not find the file."
                    )

                st.session_state["decision_variables_path"] = str(DECISION_VARS_PATH)
                st.session_state["page"] = "optimize"
                st.rerun()
            except subprocess.CalledProcessError as exc:
                log = exc.stderr or exc.stdout or str(exc)
                st.error(f"Failed to run optimization model: {log}")
            except Exception as exc:
                st.error(f"Failed to build model dictionaries: {exc}")
