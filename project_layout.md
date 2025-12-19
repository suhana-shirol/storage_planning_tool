```
project/
  README.md
  requirements.txt
  GitExample.md
  end_to_end_flow.md
  project_layout.md

  .streamlit/
    config.toml

  assets/
    bmw_logo.png

  db/
    (empty placeholder for generated DuckDB files)

  gui/
    __init__.py
    style.py
    storage_planning_app.py
    pages/
      __init__.py
      landing.py
      plan_inputs.py
      relationship_editor.py
      upload_page.py

  models/
    schema.sql
    seeds/
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
    algo.py
    alternate_dashboard.py
    dashboard.py
    Dashboard.xlsx
    non_zero_decision_variables.csv
    non_zero_decision_variables.xlsx
    params.json
    piecewiseparms.json
    retrieve.py


  pkg/
    __init__.py
    aggregate_parts.py
    db.py
    etl_loaders.py
    preprocess_cost_center.py
    preprocess_consumption.py
    preprocess_lx03.py
    preprocess_mc46.py
    validate.py

  schemas/
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
