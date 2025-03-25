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

###############################
# Compute adjusted score
###############################
def adjust_score(row):
    if row['assessment'] == 'MRS':
        return 5 - row['score']
    elif row['assessment'] == 'NIHSS':
        return 42 - row['score']
    else:
        return row['score']

df_all['adjusted_score'] = df_all.apply(adjust_score, axis=1)

###############################
# Data preparation: time point ordering and subject filtering
###############################
# Ensure that 'tp' is treated as a categorical variable (ordered by its natural order)
df_all['tp'] = pd.Categorical(df_all['tp'], categories=sorted(df_all['tp'].unique()), ordered=True)

# Keep only subjects that have exactly three time points
subject_counts = df_all.groupby('record_id')['tp'].nunique()
valid_ids = subject_counts[subject_counts == 3].index
df_all = df_all[df_all['record_id'].isin(valid_ids)]

###############################
# Define assessment order (kept as in original code)
###############################
assessment_order = ["FM-lex", "FM-uex", "BI", "MRS", "NIHSS"]
df_all['assessment'] = pd.Categorical(df_all['assessment'], categories=assessment_order, ordered=True)
df_all = df_all.sort_values('assessment')
assessments = [a for a in assessment_order if a in df_all['assessment'].unique()]
n_assess = len(assessments)

###############################
# Define stroke category line style and color mappings
###############################
# Assume two possible stroke categories: e.g., "INFARCT" and "BLEEDING"
# (If there are more, you can expand these dictionaries accordingly)
line_style_map = {
    "INFARCT": "solid",
    "BLEEDING": "dash"
}
# Choose two colors using seaborn's Set1 palette
palette = sns.color_palette("Set1", n_colors=2)
stroke_color_map = {
    "INFARCT": mcolors.to_hex(palette[0]),
    "BLEEDING": mcolors.to_hex(palette[1])
}

###############################
# Create Plotly subplots for each assessment
###############################
fig = make_subplots(rows=1, cols=n_assess, subplot_titles=assessments)

# For x-axis, create a mapping from the categorical time points to numeric positions.
tp_categories = list(df_all['tp'].cat.categories)
x_positions = {tp: i for i, tp in enumerate(tp_categories)}

for col, assess in enumerate(assessments, start=1):
    # Get the data for the current assessment
    df_assess = df_all[df_all['assessment'] == assess].copy()
    
    # Compute overall y-axis limits with a ±15% margin for the adjusted score
    overall_min = df_assess['adjusted_score'].min()
    overall_max = df_assess['adjusted_score'].max()
    overall_range = overall_max - overall_min if overall_max != overall_min else 1
    y_lim_lower = overall_min - 0.15 * overall_range
    y_lim_upper = overall_max + 0.15 * overall_range

    # --- Add boxplots for each time point (background distribution) ---
    for tp in tp_categories:
        tp_data = df_assess[df_assess['tp'] == tp]['adjusted_score']
        fig.add_trace(
            go.Box(
                y=tp_data,
                name=str(tp),
                marker_color='lightgray',
                boxpoints=False,
                showlegend=False
            ),
            row=1, col=col
        )
    
    # --- Add spaghetti plots (subject trajectories) ---
    for subject, subject_df in df_assess.groupby('record_id'):
        subject_df = subject_df.sort_values('tp')
        x_vals = [x_positions[tp] for tp in subject_df['tp']]
        y_vals = subject_df['adjusted_score'].tolist()
        # Use stroke category instead of recovery type to set line style and color.
        # We assume that the stroke_category is consistent for a given subject.
        stroke_cat = subject_df['stroke_category'].iloc[0]
        line_style = line_style_map.get(stroke_cat, "solid")
        color = stroke_color_map.get(stroke_cat, "black")
        
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='lines+markers',
                line=dict(
                    dash=line_style,
                    color=color,
                    width=1.5
                ),
                marker=dict(size=6),
                opacity=0.7,
                showlegend=False
            ),
            row=1, col=col
        )
    
    # Update the axes for the subplot
    fig.update_yaxes(range=[y_lim_lower, y_lim_upper], title_text="Score", row=1, col=col)
    fig.update_xaxes(
        tickmode='array',
        tickvals=list(x_positions.values()),
        ticktext=tp_categories,
        title_text="Time Point", row=1, col=col
    )

# Update overall layout and title
fig.update_layout(
    title_text="Spaghetti Plots by Assessment (Stroke Category Formatting)",
    width=900,
    height=400,
    template="simple_white"
)

# Save the figure as an SVG file (requires kaleido)
output_svg = os.path.join(svg_folder, "spaghetti_plots_by_assessment_stroke_category.svg")
fig.write_image(output_svg, format="svg", scale=2)
print(f"Saved combined spaghetti plots by assessment at {output_svg}")

# Optionally, display the interactive figure
fig.show("png")
