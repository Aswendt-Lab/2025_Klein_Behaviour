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
svg_folder = figures_dir
os.makedirs(svg_folder, exist_ok=True)

# Load the merged DataFrame
stitched_csv_file = os.path.join(output_dir, 'behavioral_data_cleaned_FM_BI_MRS_NIHSS_all_asssessment_types.csv')
df_all = pd.read_csv(stitched_csv_file)

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
    "Late recovery with acute decline",
    "Unclassified"
]
# Right-side nodes: stroke categories
stroke_categories = ["INFARCT", "BLEEDING", "OTHER"]

# Combine node labels (this order is preserved for mapping indices)
node_order = list(recovery_types) + list(stroke_categories)
label_to_idx = {lab: i for i, lab in enumerate(node_order)}

###############################
# Create color palettes for nodes using custom color mappings
###############################
# Use the Okabe-Ito palette for recovery types:
recovery_color_map = {
    "Steady recovery": "#009E73",                    # Bluish green for growth
    "Steady decline": "#D55E00",                     # Vermilion red for decline
    "Early recovery with chronic decline": "#E69F00", # Orange for early improvement but caution
    "Late recovery with acute decline": "#0072B2",    # Deep blue for late recovery after an acute drop
    "Unclassified": "#D9D9D9"}
recovery_colors_hex = [recovery_color_map[rt] for rt in recovery_types]

# For stroke categories, use the specified colors:
stroke_color_map = {
    "INFARCT": "#4d4d4d",  # gray
    "BLEEDING": "#e41a1c",  # red
    "OTHER": "#CCCCCC"     # default color for OTHER
}
stroke_colors_hex = [stroke_color_map[sc] for sc in stroke_categories]

# Combine the color lists in the same order as node_order.
node_colors = recovery_colors_hex + stroke_colors_hex

###############################
# Order assessments based on a predefined order
###############################
assess_order = ["FM-lex", "FM-uex", "BI", "MRS", "NIHSS"]
assessments = [assess for assess in assess_order if assess in df_all["assessment"].unique()]
n_assess = len(assessments)

###############################
# Create a subplot figure with one column per assessment using subplot_titles
###############################
fig = make_subplots(
    rows=1, cols=n_assess,
    specs=[[{"type": "sankey"} for _ in range(n_assess)]],
    subplot_titles=assessments,
    horizontal_spacing=0.03
)
for annotation in fig.layout.annotations:
    annotation.font.size = 14  # Adjust the size as needed

# Loop through each assessment and add a Sankey trace to the appropriate subplot
for i, assess in enumerate(assessments, start=1):
    df_assess = df_all[df_all["assessment"] == assess]
    
    # Prepare data for Sankey flows: aggregate counts by recovery_type and stroke_category
    agg_df = (
        df_assess.groupby(["recovery_type", "stroke_category"])
        .size()
        .reset_index(name="count")
    )
    
    source = []
    target = []
    values = []      # Percentage values relative to total subjects
    percentages = [] # Store percentage for each flow link
    
    # Compute total number of subjects for the current assessment
    df_recovery = df_assess.drop_duplicates(subset='record_id')
    total_subjects = len(df_recovery)
    
    # Calculate flow percentages relative to the overall subjects
    for _, row in agg_df.iterrows():
        rec_type = row["recovery_type"]
        stroke_cat = row["stroke_category"]
        flow_count = row["count"]
        
        # Only include flows for predefined nodes (if both exist in the mapping)
        if rec_type in label_to_idx and stroke_cat in label_to_idx:
            pct = (flow_count / total_subjects) * 100
            source.append(label_to_idx[rec_type])
            target.append(label_to_idx[stroke_cat])
            values.append(pct)
            percentages.append(pct)
    
    # Compute percentages for recovery types (left nodes)
    left_labels = []
    for rt in recovery_types:
        count = (df_recovery['recovery_type'] == rt).sum()
        pct = (count / total_subjects * 100) if total_subjects > 0 else 0
        left_labels.append(f"{count}")
    
    # Compute percentages for stroke categories (right nodes)
    right_labels = []
    for sc in stroke_categories:
        count = (df_recovery['stroke_category'] == sc).sum()
        pct = (count / total_subjects * 100) if total_subjects > 0 else 0
        right_labels.append(f"{count}")
    
    # Combined node labels for this assessment
    node_labels_with_perc = left_labels + right_labels
    
    # Create a Sankey trace with customdata (percentages) and a hovertemplate to display them.
    sankey_trace = go.Sankey(
        node=dict(
            pad=5,
            thickness=15,
            line=dict(color="black", width=0.5),
            color=node_colors
        ),
        link=dict(
            source=source,
            target=target,
            value=values,
            customdata=percentages,
            hovertemplate="Percentage: %{value:.1f}%<extra></extra>"
        )
    )
    
    # Add the trace to the subplot at row=1, col=i
    fig.add_trace(sankey_trace, row=1, col=i)

# Remove all outer margins/padding around the plot
m = 20
fig.update_layout(
    font=dict(size=12),
    width=680,  # 18 cm
    height=227,  # 6 cm
    showlegend=False,
    margin=dict(l=m, r=m, t=m, b=m)
)

# Display the figure using a PNG renderer
fig.show(renderer="png")  # or use another renderer that suits your environment

# Save the combined figure as an SVG file (if desired)
output_svg = os.path.join(svg_folder, "sankey_all_assessments.svg")
fig.write_image(output_svg, engine="kaleido")
print(f"Sankey diagrams saved to {output_svg}")

###############################
# Compute percentages for BLEEDING in steady vs. non-steady recoverers and save as CSV
###############################
results = []
# Loop through each assessment to compute the percentages
for assess in assessments:
    # Filter the data for the current assessment and ensure one row per subject
    df_assess = df_all[df_all["assessment"] == assess]
    df_recovery = df_assess.drop_duplicates(subset="record_id")
    
    # For steady recoverers (recovery_type exactly "Steady recovery")
    df_steady = df_recovery[df_recovery["recovery_type"] == "Steady recovery"]
    count_steady = len(df_steady)
    count_bleeding_steady = len(df_steady[df_steady["stroke_category"] == "BLEEDING"])
    pct_steady = (count_bleeding_steady / count_steady * 100) if count_steady > 0 else 0
    
    # For non-steady recoverers (all others)
    df_non_steady = df_recovery[df_recovery["recovery_type"] != "Steady recovery"]
    count_non_steady = len(df_non_steady)
    count_bleeding_non_steady = len(df_non_steady[df_non_steady["stroke_category"] == "BLEEDING"])
    pct_non_steady = (count_bleeding_non_steady / count_non_steady * 100) if count_non_steady > 0 else 0
    
    # Prepare sentences
    sentence_steady = f"{pct_steady:.1f}% of the steady recoverers are Bleedings."
    sentence_non_steady = f"{pct_non_steady:.1f}% of the patients that are not steady recoverers are Bleedings."
    
    # Append the results for this assessment
    results.append({
        "assessment": assess,
        "steady_recoverers_bleeding_count": count_bleeding_steady,
        "steady_recoverers_sentence": sentence_steady,
        "non_steady_recoverers_bleeding_count": count_bleeding_non_steady,
        "non_steady_recoverers_sentence": sentence_non_steady
    })

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Save the DataFrame as a CSV file
output_csv = os.path.join(output_dir, "bleeding_percentages_by_recovery.csv")
results_df.to_csv(output_csv, index=False)
print(f"Bleeding percentages CSV saved to {output_csv}")
