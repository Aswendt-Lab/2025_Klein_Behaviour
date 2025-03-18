import os
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests  # Import for FDR correction
import numpy as np

# Set up the directory paths
code_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(code_dir)
excel_file = os.path.join(parent_dir, 'input', 'Assessments_JK_V1.2.xlsx')
output_dir = os.path.join(parent_dir, "output")

# Read the Excel file into a DataFrame
df = pd.read_excel(excel_file)

# The columns to merge (wide format columns) with new names:
score_columns = [
    'BI_T0', 'MRS_T0', 'NIHSS_T0',
    'BI_T1', 'MRS_T1', 'NIHSS_T1',
    'BI_T2', 'MRS_T2', 'NIHSS_T2'
]

# Melt the DataFrame so that the score columns become rows in one "score" column.
# We keep 'record_id' and 'redcap_event_name' as identifier variables.
df_long = df.melt(id_vars=['record_id', 'redcap_event_name'],
                  value_vars=score_columns,
                  var_name='original_col',
                  value_name='score')

# The 'original_col' contains strings like 'BI_T0'.
# We want to extract:
#  - The assessment (e.g. 'BI', 'MRS', 'NIHSS'): from the first element.
#  - The time point (tp): from the second element, remove the "T" and convert to an integer.
split_cols = df_long['original_col'].str.split('_', expand=True)
df_long['assessment'] = split_cols[0].str.upper()  # Ensuring consistent case (e.g. 'BI', 'MRS', 'NIHSS')
df_long['tp'] = split_cols[1].str.replace('T', '', regex=False).astype(int)

# Also convert redcap_event_name to lowercase for consistency
df_long['redcap_event_name'] = df_long['redcap_event_name'].str.lower()

# Drop the temporary 'original_col' column
df_long.drop(columns=['original_col'], inplace=True)

# Drop any row that has a NaN value
df_long.dropna(inplace=True)

# Reorder the columns so that 'score' is the last column.
df_long = df_long[['record_id', 'redcap_event_name', 'tp', 'assessment', 'score']]

# Save the cleaned DataFrame as a CSV file at the output path
output_file = os.path.join(output_dir, 'behavioral_data_cleaned_BI_MRS_NIHSS.csv')
df_long.to_csv(output_file, index=False)

print(f"Cleaned data saved to {output_file}")





















