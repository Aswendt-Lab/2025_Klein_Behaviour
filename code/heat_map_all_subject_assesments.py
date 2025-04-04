import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go

###############################
# Setup directories
###############################
code_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(code_dir)
output_dir = os.path.join(parent_dir, "output")
figures_dir = os.path.join(output_dir, "figures")
os.makedirs(figures_dir, exist_ok=True)
input_dir = os.path.join(parent_dir, "input")

# Create a dedicated folder for individual assessment SVG files
svg_folder = figures_dir
os.makedirs(svg_folder, exist_ok=True)

###############################
# Load and filter data
###############################
stitched_csv_file = os.path.join(output_dir, 'behavioral_data_cleaned_FM_BI_MRS_NIHSS_all_asssessment_types.csv')
df_all = pd.read_csv(stitched_csv_file)
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Calibri'
mpl.rcParams['font.size'] = 12

# Further filter data to only include time point tp == 2
df_all = df_all[df_all["tp"] == 2]

expected_cols = ['record_id', 'assessment', 'recovery_type']

# Define a color mapping using colors from the Okabe-Ito palette:
recovery_color_map = {
    "Steady recovery": "#009E73",                  # Bluish green for growth
    "Steady decline": "#D55E00",                   # Vermilion red for decline
    "Early recovery with chronic decline": "#E69F00", # Orange for early improvement but caution
    "Late recovery with acute decline": "#0072B2"    # Deep blue for late recovery after an acute drop
}

print("Recovery Color Map:", recovery_color_map)

# Use the order provided by the fixed dictionary.
recovery_types = list(recovery_color_map.keys())
discrete_colors_hex = list(recovery_color_map.values())

# Map each recovery type to a numeric code for the heatmap.
numeric_map = {rt: i for i, rt in enumerate(recovery_types)}
df_all['recovery_numeric'] = df_all['recovery_type'].map(numeric_map)

# Drop duplicate rows and any rows with NaN values.
df_all = df_all.drop_duplicates().dropna()

###############################
# Add stroke category information
###############################
def categorize_stroke(stroke_str):
    # Handle missing or non-string values: default to INFARCT
    if not isinstance(stroke_str, str):
        return "INFARCT"
    
    stroke_lower = stroke_str.lower().strip()
    
    # Check for bleeding markers first
    if "blutung" in stroke_lower or stroke_lower in ["icb", "sab"]:
        return "BLEEDING"
    
    # Check for ischaemia markers - now categorized as INFARCT
    if "ischaem" in stroke_lower:
        return "INFARCT"
    
    # Check for infarct markers
    if "infrakt" in stroke_lower or "infarkt" in stroke_lower:
        return "INFARCT"
    
    # Default to INFARCT if nothing matches
    return "INFARCT"

# Create new column based on stroke_type mapping
df_all["stroke_category"] = df_all["stroke_type"].apply(categorize_stroke)

###############################
# Pivot data for heatmap
###############################
assessment_order = ["FM-lex", "FM-uex", "BI", "MRS", "NIHSS"]
assessment_order = assessment_order[::-1]

# Pivot the DataFrame with record_id as the index.
heatmap_data = df_all.pivot(index='record_id', columns='assessment', values='recovery_numeric')
heatmap_data = heatmap_data.reindex(columns=assessment_order)
heatmap_data = heatmap_data.dropna(subset=assessment_order)
heatmap_data.index = heatmap_data.index.astype(str)

###############################
# Update record ID labels based on stroke category
###############################
# Create a helper DataFrame for stroke category per record.
stroke_info = df_all[['record_id', 'stroke_category']].drop_duplicates().set_index('record_id')
stroke_info.index = stroke_info.index.astype(str)
# Filter stroke_info to only include record_ids present in heatmap_data.
stroke_info = stroke_info.loc[heatmap_data.index]
# Create new labels by prefixing with "B" for BLEEDING and "I" for INFARCT.
stroke_info['record_label'] = stroke_info.index.to_series().copy()
stroke_info.loc[stroke_info['stroke_category'] == "BLEEDING", 'record_label'] = "B" + stroke_info.loc[stroke_info['stroke_category'] == "BLEEDING", 'record_label']
stroke_info.loc[stroke_info['stroke_category'] == "INFARCT", 'record_label'] = "I" + stroke_info.loc[stroke_info['stroke_category'] == "INFARCT", 'record_label']

###############################
# Sort records based on stroke_category
###############################
# This sorts the record IDs (used as x-ticks in the flipped heatmap) based on stroke_category.
sorted_index = stroke_info.sort_values("stroke_category").index
heatmap_data = heatmap_data.reindex(sorted_index)
stroke_info = stroke_info.reindex(sorted_index)

###############################
# Create discrete colorscale for Plotly heatmap
###############################
N = len(recovery_types)
colorscale = []
for i, color in enumerate(discrete_colors_hex):
    # Calculate normalized boundaries for each discrete color.
    left = i / N
    right = (i + 1) / N
    colorscale.extend([[left, color], [right, color]])

###############################
# Create text annotations for each cell based on recovery type symbols
###############################
# Define the symbol mapping for each recovery type.
# Note: escape backslashes properly.
recovery_symbol_map = {
    'Steady recovery': '/',
    'Steady decline': '\\',
    'Early recovery with chronic decline': '/\\',
    'Late recovery with acute decline': '\\/'
}

# Create a reverse mapping from numeric code to symbol.
symbol_mapping = {numeric_map[rt]: recovery_symbol_map[rt] for rt in recovery_types}

# Create a matrix of text annotations with the same shape as heatmap_data.
text_matrix = heatmap_data.applymap(lambda x: symbol_mapping.get(x, ''))

###############################
# Transpose data to flip x and y axes
###############################
# Now x-axis will be record IDs and y-axis will be assessments.
heatmap_data_T = heatmap_data.T
text_matrix_T = text_matrix.T

###############################
# Create and display heatmap using Plotly (with flipped axes)
###############################
fig = go.Figure(
    data=go.Heatmap(
        z=heatmap_data_T.values,
        x=heatmap_data_T.columns,   # x-axis: record IDs
        y=heatmap_data_T.index,     # y-axis: assessments
        colorscale=colorscale,      # use the discrete colorscale
        xgap=1,
        ygap=1,
        showscale=False,
        text=text_matrix_T.values,  # text annotations for each cell
        texttemplate="%{text}",      # display the text annotations
        colorbar=dict(
            title="Recovery Type Code",
            tickmode='array',
            tickvals=list(range(N)),
            ticktext=recovery_types
        ),
        zmin=-0.5,
        zmax=N - 0.5
    )
)

fig.update_layout(
    title="",
    xaxis=dict(
        tickmode='array',
        tickvals=list(heatmap_data_T.columns),
        ticktext=stroke_info['record_label'].tolist(),
        side='top',          # Place x-axis ticks on the top
        tickangle=50,        # Rotate x-axis ticks by 50 degrees
        tickfont=dict(family="Calibri", size=12),
        title=dict(text="Record ID", font=dict(family="Calibri", size=16.2))
    ),
    yaxis=dict(
        tickmode='array',
        tickvals=assessment_order,
        ticktext=assessment_order,
        tickfont=dict(family="Calibri", size=12),
        title=dict(text="Assessment", font=dict(family="Calibri", size=16.2))
    ),
    template='plotly_white',
    width=750,   # Increase width for horizontal orientation
    height=300
)

# Display the figure.
fig.show(renderer='png')

# Optionally, save the figure to your svg folder.
svg_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "output", "figures", "heatmap")
os.makedirs(svg_folder, exist_ok=True)
fig.write_image(os.path.join(svg_folder, "recovery_type_heatmap.svg"))
