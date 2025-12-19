# Storage Planning Tool - End-to-End Flow

Authoritative description of what the current code does. Keep it in sync with the repo by updating sections when features change.

---

## Component Reference

- **GUI entry point** - `gui/storage_planning_app.py` routes between Wizard pages (Landing + Plan Inputs + Location Usage + Uploads + Materials Review + Optimize) and applies shared styling from `gui/style.py`.
- **Wizard pages**
  - `gui/pages/landing.py` - start/reset a plan.
  - `gui/pages/plan_inputs.py` - edit base parameters (storage types, techs, locations, system config) before deriving relationship tables.
  - `gui/pages/relationship_editor.py` - review/override generated relationship tables and lock parameters.
  - `gui/pages/upload_page.py` - upload and build each dataset (LX03 + MC46 + Cost Center + Consumption), then hand off to the review/model stages.
  - `gui/pages/materials_review.py` - review/download `parts` and derived i_sku tables and trigger the optimization model run.
  - `gui/pages/optimization.py` - render the optimization dashboard from generated decision variables.
- **Database helpers** - `pkg/db.py` manages the DuckDB connection at `db/storage_planning.duckdb`, executes `models/schema.sql`, seeds CSVs from `models/seeds/`, and exposes utilities such as `reset_and_seed_db()`, `build_relationship_tables()`, and `ensure_indicator_columns_for_tech()`.
- **ETL loaders** - `pkg/etl_loaders.py` contains the load/update routines that each page calls (`load_lx03_seed`, `load_mc46_seed`, `load_cost_center`, `load_priority_updates`, `load_movement_updates`, `load_tech_updates`, `save_i_sku_tables()`, `normalize_part_priorities()`).
- **Preprocessors** - `pkg/preprocess_lx03.py`, `pkg/preprocess_mc46.py`, `pkg/preprocess_cost_center.py`, `pkg/preprocess_consumption.py` normalize uploads using schemas from `schemas/*.json` and helper functions in `pkg/validate.py`.
- **Aggregation/model prep** - `pkg/aggregate_parts.py` builds `i_sku`, `i_sku_user`, and `i_sku_type` from `parts` plus lookup tables (used by the optimization model and persisted via `save_i_sku_tables()`).
- **Optimization pipeline** - `optimization/calculate_piecewise_params.py` derives `piecewiseparams.json`; `optimization/algo.py` (Pyomo + GLPK) solves the model and emits `non_zero_decision_variables.{xlsx,csv}` plus debug params; `optimization/alternate_dashboard.py` reshapes decision variables for charts consumed by `gui/pages/optimization.py` (and `optimization/dashboard.py` if used offline).
- **Schemas & contracts** - JSON files in `schemas/` define `required_cols`, optional dtype hints, and defaults for both inputs and outputs (e.g., `mc46_schema.json`, `lx03_output_schema.json`, `material_to_priority_schema.json`, `i_sku_schema.json`). The upload page uses these to gate buttons and to assert that preprocessors return the documented shape.

---

## Flow Overview

1. **Start a plan (Landing page)** - resets DuckDB, applies `schema.sql`, and seeds base tables from `models/seeds/*.csv` so editors and uploads have data to work with.
2. **Define plan parameters (Plan Inputs page)** - CRUD UI for storage types, technologies, system-level values, and storage locations. Any edit clears downstream relationship tables.
3. **Build/adjust relationship tables (Relationship Editor page)** - rebuilds FK-dependent tables (`location_storage_type`, `tech_location_usage`), then allows manual adjustments before locking parameters and moving to uploads.
4. **Ingestion wizard (Upload page)** - sequentially uploads LX03, MC46, cost center, and consumption files. Each step validates headers, calls its preprocessor, ensures schema setup, and runs the relevant loader.
5. **Review derived tables (Materials Review entry point)** - **Step 4: Review materials table** normalizes part priorities, builds/persists `i_sku`, `i_sku_user`, and `i_sku_type` via `save_i_sku_tables()`, and caches them for display/download.
6. **Run optimization model (Materials Review page)** - **Run Model** generates `optimization/piecewiseparams.json`, closes the DuckDB connection, and runs `python -m optimization.algo` to solve the Pyomo model and write `non_zero_decision_variables.{xlsx,csv}` (plus `debug_params.json`/`params.json` when present).
7. **Optimization dashboard (Optimize page)** - reads decision variables, rebuilds store/buy/relocation summaries with `alternate_dashboard`, and shows location-filtered charts/tables with a download of the decision variables.

Artifacts from each stage stay in DuckDB (`parts`, `cost_center`, `i_*`) or on disk under `optimization/` (piecewise params + decision variables) and feed the dashboard.

---

## Step-by-Step Detail

### 0. Start / Reset Plan (Landing page)

- **User** clicks **Start new plan**.
- **Backend** calls `pkg.db.reset_and_seed_db()`:
  - deletes any existing `db/storage_planning.duckdb` file.
  - executes `models/schema.sql` (idempotent DDL for system, tech, storage_type, storage_location, location_priority, location_storage_type, tech_location_usage, type_size_compat, parts, cost_center, tech_map, i_sku, i_sku_user, i_sku_type).
  - loads base CSVs via `pkg.etl_loaders.base_table_specs()` (system, tech, location_priority, storage_location, storage_type, tech_map, size).
- **Outputs** - session stores `db_path` and a summary of inserted rows; wizard advances to the Plan Inputs tab.
- **Edge cases** - failures bubble up as `st.error`. Because this step recreates the DB, it also clears prior uploads.

### 1. Enter Plan Inputs (`gui/pages/plan_inputs.py`)

The page opens a DuckDB connection via `db.get_con()` and renders editors inside tabs. All editors share `_render_table()` for read-only previews.

1. **Storage types**
   - Table: `storage_type` columns (`type`, `sqft_req`, `buy_cost`, `buy_invest`, `reloc_cost`, `reloc_invest`, `min_sku_per_unit`, `max_sku_per_unit`, `cubic_capacity_per_unit`).
   - Form writes through `upsert_storage_type()`. Saving auto-clears relationship tables via `db.clear_relationship_tables()` because downstream allocations depend on these records.
2. **Locations**
   - Table: `storage_location` with derived values (`reserve_cost`, `travel_time`) coming from `location_priority` table.
   - `render_location_editor()` enforces uppercase names and auto-populates derived metrics when saving.
3. **Technologies**
   - Table: `tech`. On save, `db.ensure_indicator_columns_for_tech()` adds boolean columns to `parts` and `cost_center` for any new tech to keep schema in sync (uppercased, sanitized).
4. **Plan parameters**
   - Table/form for `system` entries (line_down_cost, plan_budget, space_factor).

At the bottom, **Confirm** triggers `db.build_relationship_tables()` which loads `models/seeds/location_storage_type.csv`, `tech_location_usage.csv`, and `type_size_compat.csv` into their tables (overwriting rows). Upon success, wizard navigates to the Relationship Editor page. Pressing **Back to landing** returns to step 0 without touching data.

### 2. Review Relationship Tables (`gui/pages/relationship_editor.py`)

This page provides read/write controls for the tables populated in Step 1 confirmation.

- **Location + Storage Types tab**
  - Displays `location_storage_type` joined with `storage_location` and `storage_type` for readability.
  - Editor lets users choose a location, multi-select storage types, and provide a code + current unit count per type. Submissions replace all rows for the selected location inside a DuckDB transaction (`save_location_storage`). Empty selections delete rows, effectively zeroing allocations.
- **Technology + Location tab**
  - Shows `tech_location_usage` joins.
  - Users multi-select tech IDs per location; `save_location_tech()` wipes and inserts rows atomically.
- **Navigation** - Back button returns to Plan Inputs; **Confirm All Parameters** locks the session by navigating to the Upload page (no backend mutation; edits already persisted).

### 3. Upload and Build Data Sets (`gui/pages/upload_page.py`)

The Upload page orchestrates four sequential ingestion steps. Shared helpers:

- `_ensure_state()` initializes `st.session_state` for file buffers, validation metadata, and which datasets are built this session.
- `_store_file()` caches uploaded file bytes and resets the built flag.
- `_validate_headers()` + `_render_validation_result()` perform schema-based header checks before enabling buttons.
- `_ensure_schema()` runs `_exec_schema()` lazily to guard against missing tables.
- `_duckdb_tables_exist()` verifies prerequisites (e.g., MC46 waits for LX03-loaded `parts`). Setting `require_rows=True` ensures the dependency has data.
- `_preview_table()` renders the top rows after each build.

#### 3.1 LX03 + `parts` bootstrap

- **User** uploads LX03 CSV/XLS/XLSX, sees required columns from `schemas/lx03_schema.json`, and presses **Build LX03** once validation is green.
- **Backend**
  1. Reads file with pandas and optional dtype hints in the schema (cleaning numerics).
  2. Calls `preprocess_lx03()` which validates columns, joins storage type codes from `location_storage_type` to derive `type_id`, computes `Size` buckets based on type + `Total Stock` thresholds, deduplicates by Material, and enforces `lx03_output` schema.
  3. Runs `_ensure_schema()` to create tables if missing.
  4. Invokes `load_lx03_seed()` which truncates `parts`, determines the boolean indicator columns currently defined, and inserts Material/Size/Total Stock plus indicator defaults (0).
- **Data** - replaces the entire `parts` table; this stage establishes the row spine for later updates.
- **Error handling** - ensures a Material column exists; raises on missing required output columns; UI wraps execution inside `st.status` and surfaces exceptions.

#### 3.2 MC46 + `parts` enrichment

- **Prereqs** - `parts` must exist with rows (LX03 done). User must confirm the technology lookup table (`tech_map`) through the UI checkbox.
- **User** optionally edits mapping rows via `_render_tech_map_section()` (calls `pkg.db.fetch_tech_map`, `upsert_tech_map_entry`, `delete_tech_map_entry`). Once confirmed, they upload MC46 and click **Upload & Build MC46**.
- **Backend**
  1. Reads MC46 file with dtype hints from `schemas/mc46_schema.json`.
  2. `preprocess_mc46()` validates headers, splits `Ind. Std Desc.` tokens (`[,;/]` delimiters), maps tokens using DuckDB `tech_map` (with defaults), removes unknown placeholders, and produces boolean columns for every tech (including UNK). It forces Movement to `L` when `MRP Controller == 899`, passes through `Priority`, and enforces the `mc46_output` schema.
  3. `load_mc46_seed()` updates existing `parts` rows only (Material join). It writes priority, movement, and indicator columns (base `ASSEMBLY/BODY/PAINT/UNK` plus any custom tech indicators) while preserving LX03 columns like `Size`.
- **Outputs** - `parts` now contains priority/movement classifications and tech booleans suitable for downstream logic.
- **Edge cases** - MC46 rows with materials not present in LX03 are ignored; missing tech_map confirmation disables the build button; indicator columns are auto-added by Plan Inputs when techs change.

#### 3.3 Cost Center + `cost_center`

- **Prereqs** - LX03 (parts) must exist to keep the flow linear, although the loader writes to `cost_center` independently.
- **Backend flow**
  1. Reads file using `schemas/cost_center_schema.json`.
  2. `preprocess_cost_center()` filters departments to TX-2/3/4 prefixes, maps them to BODY/PAINT/ASSEMBLY, pivots to indicator columns (`ASSEMBLY`, `BODY`, `PAINT`, `UNK`), enforces `cost_center_output` schema.
  3. `load_cost_center()` converts `Cost Center` to integers, dedupes, truncates the DuckDB table, and inserts normalized indicators (adding missing boolean columns if they exist in schema).
- **UI** previews `cost_center` after load. Re-uploading overwrites the table.

#### 3.4 Consumption + Derived updates

- **Prereqs** - both `parts` and `cost_center` tables must exist with rows (`_duckdb_tables_exist(["parts","cost_center"], require_rows=True)`).
- **Backend flow**
  1. Reads the consumption report using `schemas/consumption_schema.json`.
  2. `preprocess_consumption()` produces three data frames:
     - `material_to_priority` - counts OR09 (line down) orders per Material, tags `Priority` H/L, and records `line_down_orders`.
     - `material_to_movement` - calculates movement tiers via Pareto on unique orders per Material, includes the order counts.
     - `material_to_tech` - maps Cost Centers to tech indicators by joining the DuckDB `cost_center` table and collapsing duplicates; unknown cost centers flag UNK.
     - Each is validated against its output schema.
  3. Loaders apply incremental updates:
     - `load_priority_updates()` fills missing `priority`/`line_down_orders` only and defaults any remaining NULLs to `L`.
     - `load_movement_updates()` fills missing `movement`/`orders`; enforces `L` fallback and zeroes order metrics.
     - `load_tech_updates()` ORs in new tech indicators, clears `UNK` whenever a concrete user exists, recomputes `num_users`.
- **UI** wraps the process in a `st.status` block and previews `parts` afterward.
- **Edge cases** - consumption rows whose materials are not yet in `parts` are ignored; missing cost centers become UNK via the preprocessor; schema mismatches stop the run with explicit error messages.

### 4. Review Materials and Derived Tables (`gui/pages/materials_review.py`)

- **Entry point** - Upload page button **Step 4: Review materials table** first calls `etl_loaders.normalize_part_priorities()` and `etl_loaders.save_i_sku_tables()` (persisting `i_sku`, `i_sku_user`, `i_sku_type` and caching them in `st.session_state`), then navigates to this page.
- **Parts table** - loads `parts`, shows row counts, and provides CSV/XLSX downloads.
- **Aggregated tables** - uses cached `i_sku`/`i_sku_user`/`i_sku_type` if present; otherwise rebuilds via `build_i_sku*` and offers downloads. Empty tables surface info banners.
- **Run Model button**
  - Generates piecewise parameters with `generate_piecewise_params()` and writes `optimization/piecewiseparams.json` (also stored in session state).
  - Closes the shared DuckDB connection (`close_con`) to release file locks before launching the solver.
  - Runs `[sys.executable, "-m", "optimization.algo"]` from project root, capturing stdout/stderr. The solver writes `optimization/non_zero_decision_variables.xlsx` (and `.csv`) plus `optimization/debug_params.json`.
  - On success, stores the decision variable path in session state, sets `page` to `optimize`, and reruns; failures surface via `st.error`.

### 5. Optimization Dashboard (`gui/pages/optimization.py`)

- Loads decision variables from `st.session_state["decision_variables_path"]` or the default `optimization/non_zero_decision_variables.xlsx`; exposes a download button.
- Optionally reads `optimization/params.json` (warns if unreadable).
- Uses `alternate_dashboard` helpers to build:
  - Units change (current vs recommended store units per storage type/location).
  - Buy units per location.
  - Relocations (storage type, current location -> recommended location).
- UI tabs: **Total Units Change**, **Buy Units**, **Relocations**. Each filters by location (or origin), renders an Altair bar chart, and shows the filtered dataframe; empty tables show info banners.

---

## Logging, Validation, and Error Surfacing

- **Schema checks** - `pkg/validate.validate()` throws `ValidationError` when required columns are missing or typed incorrectly. Upload UI catches these via `_validate_headers()` before data ever hits pandas.
- **Tech indicators** - `ensure_indicator_columns_for_tech()` keeps DuckDB tables aligned with dynamic tech additions; uploaders rely on these columns existing before load.
- **Transactions** - every loader uses DuckDB `BEGIN ... COMMIT` blocks; failures roll back so partially written tables do not occur.
- **Status widgets** - `st.status` on each upload surfaces step-by-step progress. Exceptions bubble up to Streamlit error banners; model run errors show the captured subprocess output.
- **Plan resets** - rerunning **Start new plan** is the supported way to clear all data. There is no separate "Clear database" button in this version; resetting also re-seeds tech_map/system defaults.

---

## Data Contracts Snapshot

- **parts** (`models/schema.sql`)
  - Keys: `material TEXT PRIMARY KEY`.
  - Columns populated by LX03: `size`, `total_stock`, `size_id`, `storage_type`; MC46/consumption fill `priority`, `movement`, `orders`, `line_down_orders`, `num_users`, and tech booleans like `ASSEMBLY/BODY/PAINT/UNK` (plus any dynamically added ones).
  - Loaders never drop rows; LX03 truncates/rebuilds, while subsequent steps update existing materials.
- **cost_center**
  - Columns: `cost_center INTEGER`, `ASSEMBLY/BODY/PAINT/UNK BOOLEAN`, `updated_at TIMESTAMP`. Always truncated before insert.
- **i_sku / i_sku_user / i_sku_type**
  - `i_sku`: key `category_id`; category string (priority+movement+size), totals (`numSKU`, `total_stock`, `total_orders`, `line_down_orders`), `size_id`, `min_vol`, `SKU_cubic_ft`.
  - `i_sku_user`: composite key (`category_id`, `tech_id`); columns include `category`, `tech_name`, `numSKU`, `ld_orders_per_user`, `orders_per_user`.
  - `i_sku_type`: composite key (`category_id`, `type_id`, `size_id`); columns include storage type metadata, `cubic_capacity_per_unit`, `max_sku_per_unit`, `compatible`, `penalty`.
- **Support tables** - `storage_type`, `storage_location`, `location_priority`, `location_storage_type`, `tech`, `tech_location_usage`, `tech_map`, `system`, `type_size_compat`, `size`. Editors modify these directly; seeds act as defaults.
- **Optimization artifacts (filesystem)** - `optimization/piecewiseparams.json`, `optimization/non_zero_decision_variables.{xlsx,csv}`, `optimization/debug_params.json`, and optional `optimization/params.json` consumed by the dashboard.

---

## Recommended Test Scenarios

1. **Plan initialization** - click **Start new plan** and verify the seed summary lists every base table (system/tech/storage_location/storage_type/tech_map/size) with expected row counts.
2. **Editing parameters** - add a new tech, confirm indicator columns appear in DuckDB (`PRAGMA table_info('parts')`) and that relationship tables clear/rebuild as expected.
3. **LX03 ingestion** - upload a minimal LX03 file, confirm `parts` row count equals unique materials and that `Size` derived values match thresholds.
4. **MC46 + Cost Center ingestion** - confirm tech_map must be acknowledged before MC46 builds, booleans flip as expected, and cost_center truncates/rewrites on reupload.
5. **Consumption ingestion** - ensure buttons stay disabled until parts + cost_center exist, then verify derived priority/movement/tech updates fill NULLs and clear UNK when concrete techs appear.
6. **Review and optimization** - use **Step 4: Review materials table** to build `i_*` tables, download them, run the model to produce `piecewiseparams.json` and `non_zero_decision_variables.xlsx`, and confirm the Optimization dashboard renders charts/tables per location.

---

## Improvements

1. Make cost center mappings dynamic with new technologies: currently `preproccess_consumption.py` maps cost center codes to based on a hard-coded mapping from `Department` to technology group. If a user enters a new technology, they should be able to add a department mapping. This would require an additional table or a column in the `tech` table and modifications to the relationships page and the `preproccess_consumption.py`.
