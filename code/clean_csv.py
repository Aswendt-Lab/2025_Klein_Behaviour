import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests  # Import for FDR correction
import numpy as np

# Set up the directory paths
code_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(code_dir)
excel_file = os.path.join(parent_dir, 'input', 'FM_JK_V1.xlsx')
output_dir = os.path.join(parent_dir, "output")

# Read the Excel file into a DataFrame
df = pd.read_excel(excel_file)

# The columns to merge (wide format columns)
score_columns = [
    'T0_FM_UEx', 'T0_FM_LEx',
    'T1_FM_Uex', 'T1_FM_Lex',
    'T2_FM_UEx', 'T2_FM_LEx'
]

# Melt the DataFrame so that the score columns become rows in one "score" column.
# We keep 'record_id' and 'redcap_event_name' as identifier variables.
df_long = df.melt(id_vars=['record_id', 'redcap_event_name'],
                  value_vars=score_columns,
                  var_name='original_col',
                  value_name='score')

# The 'original_col' contains strings like 'T0_FM_UEx'.
# We want to extract:
#  - The time point (tp): take the first element, remove the "T", and convert to an integer.
#  - The position: the third element.
# The second element ("FM") is not used.
# We split the string by "_" and expand into separate columns.
split_cols = df_long['original_col'].str.split('_', expand=True)
# The first column is like 'T0' -> remove the 'T' and convert to int.
df_long['tp'] = split_cols[0].str.replace('T', '', regex=False).astype(int)
# The third element (position) is in the third column (index 2) and we convert it to lowercase.
df_long['position'] = split_cols[2].str.lower()

# Also convert redcap_event_name to lowercase for consistency
df_long['redcap_event_name'] = df_long['redcap_event_name'].str.lower()

# Drop the temporary 'original_col' column
df_long.drop(columns=['original_col'], inplace=True)

# Drop any row that has a NaN value
df_long.dropna(inplace=True)

# Reorder the columns so that 'score' is the last column.
df_long = df_long[['record_id', 'redcap_event_name', 'tp', 'position', 'score']]

# Save the cleaned DataFrame as a CSV file at the output path
output_file = os.path.join(output_dir, 'behavioral_data_cleaned.csv')
df_long.to_csv(output_file, index=False)

print(f"Cleaned data saved to {output_file}")
