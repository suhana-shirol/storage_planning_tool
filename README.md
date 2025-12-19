# Storage Planning Tool

Wizard-style Streamlit app that loads SAP exports into DuckDB, builds derived aggregates, and runs an optimization model to recommend storage unit changes, buys, and relocations. Everything runs locally: Streamlit UI, embedded DuckDB, and a Pyomo + GLPK solver.

---

## Quick start
1) Install Python deps: `pip install -r storage_planning_requirements.txt`
2) Ensure GLPK solver is available on PATH (`glpsol` binary). The `glpk` Python package is listed, but you still need the native solver (conda or Chocolatey on Windows are easy options).
3) Run the app: `streamlit run gui/storage_planning_app.py`
4) Follow the wizard tabs: Start new plan → Inputs → Relationships → Uploads → Review → Optimize.

---

## Technical stack
- **Languages**: Python 3, SQL (DDL/DML in `models/schema.sql`), Markdown for docs.
- **UI**: Streamlit (`gui/`), Altair charts for the optimization dashboard.
- **Data/ETL**: pandas, numpy, schema-driven preprocessors (`pkg/preprocess_*.py`), loaders in `pkg/etl_loaders.py`, validation in `pkg/validate.py`.
- **Database**: DuckDB file at `db/storage_planning.duckdb`; seeds in `models/seeds/*.csv`.
- **Optimization**: Pyomo model solved with GLPK (`optimization/algo.py`), piecewise fitting via `pwlf`, reshaping dashboards via `optimization/alternate_dashboard.py`.
- **Artifacts**: Piecewise params and decision variables written to `optimization/` (e.g., `piecewiseparams.json`, `non_zero_decision_variables.xlsx/csv`, `debug_params.json`).

---

## Project layout
```
project/
  README.md
  requirements.txt
  end_to_end_flow.md
  project_layout.md

  .streamlit/
    config.toml

  assets/
    bmw_logo.png

  db/
    (generated DuckDB files live here)

  gui/
    storage_planning_app.py
    style.py
    pages/
      landing.py
      plan_inputs.py
      relationship_editor.py
      upload_page.py
      materials_review.py
      optimization.py

  models/
    schema.sql                     # DDL for all tables
    seeds/                         # Default lookup data loaded on reset
      location_priority.csv
      location_storage_type.csv
      part_category.csv
      size.csv
      storage_location.csv
      storage_type.csv
      system.csv
      tech.csv
      tech_location_usage.csv
      tech_map.csv
      type_size_compat.csv

  optimization/
    __init__.py
    algo.py                        # Pyomo model (GLPK solver) writes decision variables
    alternate_dashboard.py         # Shapes solver outputs for dashboard tables/charts
    dashboard.py                   # Legacy/offline dashboard helper
    Dashboard.xlsx
    non_zero_decision_variables.csv
    non_zero_decision_variables.xlsx
    params.json
    piecewiseparams.json
    retrieve.py                    # Pulls model inputs from DuckDB
    calculate_piecewise_params.py  # Builds piecewiseparams.json from parts data

  pkg/
    __init__.py
    aggregate_parts.py             # Builds i_sku, i_sku_user, i_sku_type
    db.py                          # DuckDB connection, schema init/seed/reset
    etl_loaders.py                 # Loaders for LX03/MC46/Cost Center/Consumption; saves i_* tables
    preprocess_cost_center.py      # Normalize cost center upload
    preprocess_consumption.py      # Derive priority/movement/tech from consumption
    preprocess_lx03.py             # Normalize LX03 upload
    preprocess_mc46.py             # Normalize MC46 upload
    validate.py                    # Schema loading/validation helpers

  schemas/                         # Input/output schema contracts
    consumption_schema.json
    cost_center_output_schema.json
    cost_center_schema.json
    i_sku_type.json
    i_sku_user.json
    i_sku.json
    lx03_output_schema.json
    lx03_schema.json
    material_to_movement_schema.json
    material_to_priority_schema.json
    material_to_tech_schema.json
    mc46_output_schema.json
    mc46_schema.json
```

---

## Flow (high level)
1) **Plan setup**: Reset DB and seed lookups (`Start new plan`).
2) **Inputs**: Edit storage types/locations/tech/system; rebuild relationships.
3) **Uploads**: LX03 → `parts`; MC46 enriches `parts`; Cost Center → `cost_center`; Consumption fills priority/movement/tech users.
4) **Derive aggregates**: Build/persist `i_sku`, `i_sku_user`, `i_sku_type`; normalize priorities.
5) **Optimize**: Generate piecewise params, run Pyomo+GLPK, write decision variables; dashboard renders store/buy/relocation charts with downloads.

See `end_to_end_flow.md` for detailed, step-by-step behavior and contracts.

---

## Notes
- Everything is local: no external DB or cloud services required.
- Keep `requirements.txt` synced if you add dependencies (UI, ETL, or solver).
- If the solver cannot be found, ensure `glpsol` is installed and on PATH before running the Optimize step.
