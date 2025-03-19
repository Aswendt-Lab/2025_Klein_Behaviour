import os
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.graph_objects as go
import matplotlib.colors as mcolors

# Suppose df_all has already been loaded from your CSV file.
expected_cols = ['record_id', 'assessment', 'recovery_type']
assert all(col in df_all.columns for col in expected_cols), "df_all does not contain all expected columns."

# Define the ordering of recovery types.
recovery_types = sorted(df_all['recovery_type'].unique())

# Create the color palette using Seaborn.
colors = sns.color_palette("Set1", n_colors=len(recovery_types))
# Convert the colors to hex format using matplotlib.
colors_hex = [mcolors.to_hex(c) for c in colors]

# Map recovery types to colors.
recovery_color_map = dict(zip(recovery_types, colors_hex))
print("Recovery Color Map:", recovery_color_map)

# Map each recovery type to a numeric code for the heatmap.
numeric_map = {rt: i for i, rt in enumerate(recovery_types)}
df_all['recovery_numeric'] = df_all['recovery_type'].map(numeric_map)

# Drop duplicate rows and any rows with NaN values.
df_all = df_all.drop_duplicates()
df_all = df_all.dropna()

# Reset the index to create a unique identifier for each row.
df_all = df_all.reset_index(drop=True)
df_all['unique_id'] = df_all.index

# Reorder the assessment columns according to the specified order.
assessment_order = ["FM-lex", "FM-uex", "BI", "MRS", "NIHSS"]
# Pivot the DataFrame using the unique identifier as the index.
heatmap_data = df_all.pivot(index='unique_id', columns='assessment', values='recovery_numeric')
# Reindex the pivoted table to enforce the desired order of assessment columns.
heatmap_data = heatmap_data.reindex(columns=assessment_order)

# Drop rows with any NaN values in the assessment columns
heatmap_data = heatmap_data.dropna(subset=assessment_order)

# Create a custom discrete colorscale for Plotly.
n = len(recovery_types)
colorscale = []
for i, color in enumerate(colors_hex):
    lower_bound = (i - 0.5) / (n - 1) if i > 0 else 0.0
    upper_bound = (i + 0.5) / (n - 1) if i < n - 1 else 1.0
    colorscale.append([lower_bound, color])
    colorscale.append([upper_bound, color])

# Create the heatmap with xgap and ygap for separation.
# Remove the default colorbar by setting showscale=False.
fig = go.Figure(data=go.Heatmap(
    z=heatmap_data.values,
    x=heatmap_data.columns,
    y=heatmap_data.index,
    colorscale=colorscale,
    zmin=0,
    zmax=n-1,
    xgap=2,  # gap between columns (in pixels)
    ygap=2,  # gap between rows (in pixels)
    showscale=False  # drop the built-in colorbar
))

# Add custom dummy scatter traces for the legend.
for rt in recovery_types:
    fig.add_trace(go.Scatter(
         x=[None],
         y=[None],
         mode='markers',
         marker=dict(size=10, color=recovery_color_map[rt]),
         legendgroup=rt,
         showlegend=True,
         name=rt
    ))

# Create a mapping from the unique_id to the original record_id.
unique_id_to_record_id = df_all.set_index('unique_id')['record_id'].to_dict()

# Update the layout with axis titles, custom legend location, new dimensions,
# and update y-axis to show every tick with the original record_id.
fig.update_layout(
    title='Recovery Type Heatmap',
    xaxis_title='Assessment',
    yaxis_title='Record ID',
    yaxis=dict(
         autorange='reversed',
         tickmode='array',
         tickvals=list(heatmap_data.index),
         ticktext=[unique_id_to_record_id[i] for i in heatmap_data.index]
    ),
    legend=dict(
         orientation="v",
         x=1.05,  # places legend outside to the right
         y=1,
         title="Recovery Type"
    ),
    height=800,  # taller plot
    width=600    # narrower plot
)

# Display the figure.
fig.show(renderer='browser')

# Optionally, save the figure to your svg folder.
svg_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "figures", "heatmap")
os.makedirs(svg_folder, exist_ok=True)
fig.write_image(os.path.join(svg_folder, "recovery_type_heatmap.svg"))
