import itertools
import json
from typing import Dict

import numpy as np
import pandas as pd
from pwlf import PiecewiseLinFit

from pkg.db import get_con


def fetch_parts() -> pd.DataFrame:
    con = get_con()
    try:
        return con.execute("SELECT * FROM parts").df()
    except Exception as exc:
        raise RuntimeError("Failed to read parts from DuckDB. Ensure schema is initialized.") from exc


def prep_parts() -> pd.DataFrame:
    parts_df = fetch_parts()
    if "material" not in parts_df.columns:
        raise ValueError("parts table is missing required column 'material'.")

    # Avoid division by zero; keep NaN where num_users is zero/NaN
    denom = parts_df["num_users"].replace(0, np.nan)
    parts_df["orders_per_user"] = parts_df["orders"] / denom
    parts_df["ld_orders_per_user"] = np.ceil(parts_df["line_down_orders"] / denom * 0.5833)
    parts_df[["orders_per_user", "ld_orders_per_user"]] = parts_df[
        ["orders_per_user", "ld_orders_per_user"]
    ].fillna(0)

    parts_df["category"] = (
        parts_df.get("priority", "").fillna("")
        + parts_df.get("movement", "").fillna("")
        + parts_df.get("size", "").fillna("")
    )
    return parts_df


def fit_pwlf_two_segment(x, y):
    """Fit a 2-segment PWLF (1 breakpoint). Returns (bp_x, bp_y)."""
    pwlf_model = PiecewiseLinFit(x, y)
    breaks = pwlf_model.fit(2)
    bp_x = breaks[1]
    bp_y = float(pwlf_model.predict([bp_x])[0])
    return bp_x, bp_y


def is_concave(breaks, vals, tol=1e-9):
    """True if piecewise linear function is concave (slopes non-increasing)."""
    slopes = np.diff(vals) / np.diff(breaks)
    return np.all(slopes[1:] <= slopes[:-1] + tol), slopes


def tuples_to_strings(d: Dict) -> Dict:
    return {str(k): v for k, v in d.items()}


def make_json_safe(o):
    """Convert numpy types + Python scalars/lists for JSON."""
    if isinstance(o, dict):
        return {make_json_safe(k): make_json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [make_json_safe(v) for v in o]
    if hasattr(o, "item") and callable(o.item):
        return o.item()
    if hasattr(o, "tolist") and callable(o.tolist):
        return o.tolist()
    return o


def generate_piecewise_params(output_path: str = "piecewiseparams.json") -> dict:
    """
    Compute piecewise params and write JSON to `output_path`.
    Returns the JSON-able dict.
    """
    breakpoints: dict = {}
    pieceval: dict = {}
    resbreakpoints: dict = {}
    respieceval: dict = {}
    concave_flag: dict = {}
    resconcave_flag: dict = {}

    parts_df = prep_parts()
    priority = sorted(set(parts_df["priority"].fillna("")))
    movement = sorted(set(parts_df["movement"].fillna("")))
    size = sorted(set(parts_df["size"].fillna("")))
    USERS = [col for col in parts_df.columns if col.isupper() and col != "MATERIAL"]
    SKUS = ["".join(x) for x in itertools.product(priority, movement, size) if "".join(x)]

    for isku in SKUS:
        for iuser in USERS:
            df = parts_df[(parts_df["category"] == isku) & (parts_df[iuser] == 1)].sort_values(
                "ld_orders_per_user", ascending=False
            )

            total_skus = len(df)
            total_orders = df["ld_orders_per_user"].sum()

            breakpoints[isku, iuser, 0] = 0
            pieceval[isku, iuser, 0] = 0

            if total_skus <= 2 or df["ld_orders_per_user"].nunique() <= 1:
                mid = total_skus // 2
                breakpoints[isku, iuser, 1] = mid
                pieceval[isku, iuser, 1] = total_orders * 0.5
                breakpoints[isku, iuser, 2] = total_skus
                pieceval[isku, iuser, 2] = total_orders
                continue

            x = np.arange(1, total_skus + 1)
            y = df["ld_orders_per_user"].cumsum()
            bp_x, bp_y = fit_pwlf_two_segment(x, y)

            if bp_y >= total_orders:
                bp_y = total_orders * 0.80

            breakpoints[isku, iuser, 1] = bp_x
            pieceval[isku, iuser, 1] = bp_y
            breakpoints[isku, iuser, 2] = total_skus
            pieceval[isku, iuser, 2] = total_orders

            brks = [
                breakpoints[isku, iuser, 0],
                breakpoints[isku, iuser, 1],
                breakpoints[isku, iuser, 2],
            ]
            vals = [
                pieceval[isku, iuser, 0],
                pieceval[isku, iuser, 1],
                pieceval[isku, iuser, 2],
            ]
            ok, slopes = is_concave(brks, vals)
            concave_flag[(isku, iuser)] = ok
            if not ok:
                print(f"WARNING: Not concave for SKU={isku}, User={iuser}. Slopes = {slopes}")

    for isku in SKUS:
        df = parts_df[parts_df["category"] == isku].sort_values("orders", ascending=False)
        total_skus = df["material"].nunique()
        total_orders = df["orders"].sum()

        resbreakpoints[isku, 0] = 0
        respieceval[isku, 0] = 0

        if total_skus <= 2 or df["orders"].nunique() <= 1:
            mid = total_skus // 2
            resbreakpoints[isku, 1] = mid
            respieceval[isku, 1] = total_orders * 0.5
            resbreakpoints[isku, 2] = total_skus
            respieceval[isku, 2] = total_orders
            continue

        x = np.arange(1, total_skus + 1)
        y = df["orders"].cumsum()
        bp_x, bp_y = fit_pwlf_two_segment(x, y)

        if bp_y >= total_orders:
                bp_y = total_orders * 0.80

        resbreakpoints[isku, 1] = bp_x
        respieceval[isku, 1] = bp_y
        resbreakpoints[isku, 2] = total_skus
        respieceval[isku, 2] = total_orders

        brks = [resbreakpoints[isku, 0], resbreakpoints[isku, 1], resbreakpoints[isku, 2]]
        vals = [respieceval[isku, 0], respieceval[isku, 1], respieceval[isku, 2]]
        ok, slopes = is_concave(brks, vals)
        resconcave_flag[isku] = ok
        if not ok:
            print(f"WARNING: Not concave (RESIDUAL) for SKU={isku}. Slopes={slopes}")

    allpiecewise_json = {
        "breakpoints": tuples_to_strings(breakpoints),
        "pieceval": tuples_to_strings(pieceval),
        "resbreakpoints": tuples_to_strings(resbreakpoints),
        "respieceval": tuples_to_strings(respieceval),
    }
    allpiecewise_json = make_json_safe(allpiecewise_json)

    with open(output_path, "w") as f:
        json.dump(allpiecewise_json, f, indent=4, sort_keys=True)

    return allpiecewise_json


# if __name__ == "__main__":
#     generate_piecewise_params()
