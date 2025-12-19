import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # -> project/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from gui.pages import landing, plan_inputs, relationship_editor, upload_page, materials_review, optimization, map
from gui import style

ICON = Path(__file__).resolve().parents[1] / "assets" / "bmw_logo.png"
PROGRESS_STEPS = [
    ("landing", "Plan Setup"),
    ("parameters", "Inputs"),
    ("relationships", "Location & Tech Mapping"),
    ("upload", "Upload Files"),
    ("review", "Materials Review"),
    ("optimize", "Optimizing...")
]


def render_step_progress(current_page: str) -> None:
    total = len(PROGRESS_STEPS)
    idx = next((i for i, (key, _) in enumerate(PROGRESS_STEPS) if key == current_page), 0)
    st.progress((idx + 1) / total)
    labels = []
    for i, (_, label) in enumerate(PROGRESS_STEPS):
        if i < idx:
            labels.append(f"✅ {label}")
        elif i == idx:
            labels.append(f"**{label}**")
        else:
            labels.append(label)
    st.markdown(" → ".join(labels))


def main() -> None:
    st.set_page_config(
        page_title="TX-155 Storage Planning Tool",
        layout="wide",
        page_icon=ICON if ICON.exists() else None,
    )
    style.apply_brand_styles()
    st.session_state.setdefault("page", "landing")
    page = st.session_state["page"]

    # Map page should display progress as optimize step
    progress_page = "optimize" if page == "map" else page

    render_step_progress(progress_page)

    if page == "parameters":
        plan_inputs.render()
    elif page == "relationships":
        relationship_editor.render()
    elif page == "upload":
        upload_page.render()
    elif page == "review":
        materials_review.render()
    elif page == "optimize":
        optimization.render()
    elif page == "map":
        map.render()
    else:
        landing.render()


if __name__ == "__main__":
    main()
