# pkg/validate.py
import pandas as pd
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

SCHEMA_DIR = Path(__file__).resolve().parents[1] / "schemas"

_NAME_TO_FILE = {
    "mc46": "mc46_schema.json",
    "mc46_output": "mc46_output_schema.json",
    "cost_center": "cost_center_schema.json",
    "cost_center_output": "cost_center_output_schema.json",
    "lx03" : "lx03_schema.json",
    "lx03_output" : "lx03_output_schema.json",
    "consumption": "consumption_schema.json",
    "material_to_priority" : "material_to_priority_schema.json",
    "material_to_movement": "material_to_movement_schema.json",
    "material_to_tech" : "material_to_tech_schema.json",
    "i_sku" : "i_sku.json",
    "i_sku_user" : "i_sku_user.json",
    "i_sku_type" : "i_sku_type.json"
}

class ValidationError(Exception):
    """Custom error for schema mismatches."""
    pass

def resource_path(rel_path):
    """Handles bundled files for PyInstaller."""
    base = getattr(sys, '_MEIPASS', os.path.abspath("."))
    return os.path.join(base, rel_path)

def load_schema(name_or_filename: str) -> Dict[str, Any]:
    """
    Load a JSON schema by short name ('mc46', 'cost_center', 'consumption')
    or by explicit filename ('mc46_schema.json', etc.).
    """
    # Map friendly names to filenames when provided.
    filename = _NAME_TO_FILE.get(name_or_filename, name_or_filename)
    path = SCHEMA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Schema not found: {path}")
    # Read and parse the JSON schema file.
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def required_cols(schema: Dict[str, Any]) -> List[str]:
    """Return required column names from the schema (gracefully handles missing key)."""
    return list(schema.get("required_cols", []))
    
def validate(df: pd.DataFrame, schema_name: str):
    """Validate a DataFrame against a schema definition"""
    # Load the schema definition and expected column list.
    schema = load_schema(schema_name)
    required_cols = schema["required_cols"]

    # Identify missing and extra columns relative to the schema.
    missing = [c for c in required_cols if c not in df.columns]
    extra = [c for c in df.columns if c not in required_cols]

    if missing:
        raise ValidationError(f"Missing required columns: {missing}")
    
    # Optional dtype validation when schema supplies expectations.
    if "dtypes" in schema:
        for col, expected_type in schema["dtypes"].items():
            if col in df.columns and str(df[col].dtype) != expected_type:
                raise ValidationError(f"Column {col} expected type {expected_type}, got {df[col].dtype}")
    
    # Trim DataFrame to required columns, preserving order.
    validated_df = df[[c for c in required_cols if c in df.columns]]
    return validated_df, missing, extra

def enforce_output_schema(df: pd.DataFrame, schema):
    # Add any missing required columns, filling with defaults when provided.
    defaults = schema.get("defaults", {})
    required_cols = schema["required_cols"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = defaults.get(col, pd.NA)

    return df
