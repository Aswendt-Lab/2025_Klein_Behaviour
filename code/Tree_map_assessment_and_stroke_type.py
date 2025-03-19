import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.colors as mcolors

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
svg_folder = os.path.join(figures_dir, "assessment_svgs")
os.makedirs(svg_folder, exist_ok=True)

# Load the merged DataFrame
stitched_csv_file = os.path.join(output_dir, 'behavioral_data_cleaned_FM_BI_MRS_NIHSS_all_asssessment_types.csv')
df_all = pd.read_csv(stitched_csv_file)

###############################
# Define mapping function for stroke types
###############################
def categorize_stroke(stroke_str):
    # Handle missing or non-string values
    if not isinstance(stroke_str, str):
        return "OTHER"
    
    stroke_lower = stroke_str.lower().strip()
    
    # Check for bleeding markers first
    if "blutung" in stroke_lower or stroke_lower in ["icb", "sab"]:
        return "BLEEDING"
    
    # Check for ischaemia markers - all such cases become OTHER
    if "ischaem" in stroke_lower:
        return "OTHER"
    
    # Check for infarct markers
    if "infrakt" in stroke_lower or "infarkt" in stroke_lower:
        return "INFARCT"
    
    # Default to OTHER if nothing matches
    return "OTHER"

# Create new column based on mapping
df_all["stroke_category"] = df_all["stroke_type"].apply(categorize_stroke)

###############################
# Further filter data to only include time point tp == 2
###############################
df_all = df_all[df_all["tp"] == 2]

###############################
# Define nodes for the Sankey diagram
###############################
# Left-side nodes: recovery types
recovery_types = [
    "Steady recovery", 
    "Steady decline", 
    "Early recovery with chronic decline", 
    "Late recovery with acute decline"
]
# Right-side nodes: stroke categories
stroke_categories = ["INFARCT", "BLEEDING", "OTHER"]

# Combine node labels; left nodes come first, then right nodes.
node_labels = list(recovery_types) + list(stroke_categories)
label_to_idx = {lab: i for i, lab in enumerate(node_labels)}

###############################
# Create color palettes for nodes using Seaborn and Matplotlib
###############################
# For recovery types, use the "Set1" palette.
recovery_colors = sns.color_palette("Set1", n_colors=len(recovery_types))
recovery_colors_hex = [mcolors.to_hex(color) for color in recovery_colors]

# For stroke categories, use a different palette such as "Set2".
stroke_colors = sns.color_palette("Set2", n_colors=len(stroke_categories))
stroke_colors_hex = [mcolors.to_hex(color) for color in stroke_colors]

# Combine the color lists in the same order as node_labels.
node_colors = recovery_colors_hex + stroke_colors_hex

###############################
# Order assessments based on a predefined order
###############################
assess_order = ["FM-lex", "FM-uex", "BI", "MRS", "NIHSS"]
assessments = [assess for assess in assess_order if assess in df_all["assessment"].unique()]
n_assess = len(assessments)

###############################
# Create a subplot figure with one row and one column per assessment
###############################
fig = make_subplots(
    rows=1, cols=n_assess,
    specs=[[{"type": "sankey"} for _ in range(n_assess)]],
    horizontal_spacing=0.03
)

# Loop through each assessment and add a Sankey trace to the appropriate subplot
for i, assess in enumerate(assessments, start=1):
    df_assess = df_all[df_all["assessment"] == assess]
    
    # Aggregate counts by recovery_type and stroke_category
    agg_df = (
        df_assess.groupby(["recovery_type", "stroke_category"])
        .size()
        .reset_index(name="count")
    )
    
    source = []
    target = []
    values = []
    percentages = []  # To store percentage for each link
    
    # Compute totals per recovery type for percentage calculation
    rec_totals = agg_df.groupby("recovery_type")["count"].sum().to_dict()
    
    for _, row in agg_df.iterrows():
        rec_type = row["recovery_type"]
        stroke_cat = row["stroke_category"]
        flow_value = row["count"]
        
        # Only include flows for predefined nodes
        if rec_type in label_to_idx and stroke_cat in label_to_idx:
            source.append(label_to_idx[rec_type])
            target.append(label_to_idx[stroke_cat])
            values.append(flow_value)
            # Calculate percentage relative to the recovery type's total
            pct = (flow_value / rec_totals[rec_type]) * 100
            percentages.append(pct)
    
    # Create a Sankey trace with customdata (percentages) and a hovertemplate to display them
    sankey_trace = go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=node_labels,
            color=node_colors
        ),
        link=dict(
            source=source,
            target=target,
            value=values,
            customdata=percentages,
            hovertemplate="Flow: %{value}<br>Percentage: %{customdata:.1f}%<extra></extra>"
        )
    )
    
    # Add the trace to the subplot at row=1, col=i
    fig.add_trace(sankey_trace, row=1, col=i)
    
    # Add annotation (subtitle) above each subplot.
    # Compute x-coordinate as the center of each subplot (i-0.5)/n_assess in paper coordinates.
    x_center = (i - 0.5) / n_assess
    fig.add_annotation(
        x=x_center, y=1.1, xref="paper", yref="paper",
        text=assess,
        showarrow=False,
        font=dict(size=14, color="black"),
        align="center"
    )

# Update overall layout of the figure
fig.update_layout(
    title_text="Sankey Diagrams per Assessment (tp=2)",
    font=dict(size=10),
    width=300 * n_assess,  # Adjust width per subplot as needed
    height=600,
    showlegend=False,
    margin=dict(t=100)  # Increase top margin to accommodate subtitles
)

# Display the figure (this will not automatically open in a browser)
fig.show(renderer="iframe")  # or use "png" or another renderer that suits your environment

# Save the combined figure as an SVG file (if desired)
output_svg = os.path.join(svg_folder, "sankey_all_assessments.svg")
fig.write_image(output_svg, engine="kaleido")
print(f"Sankey diagrams saved to {output_svg}")
