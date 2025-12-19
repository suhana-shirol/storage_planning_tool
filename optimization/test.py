import pandas as pd
from optimization import alternate_dashboard
df = pd.read_excel("optimization/non_zero_decision_variables.xlsx")
alternate_dashboard.opti_df = df
print("store units:", alternate_dashboard.build_store_units())
print("sku alloc:", [len(x) for x in alternate_dashboard.build_SKUalloc()])
