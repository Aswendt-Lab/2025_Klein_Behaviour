import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

###############################
# Setup directories and load data
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
df_all['tp'] = pd.Categorical(df_all['tp'], categories=sorted(df_all['tp'].unique()), ordered=True)

# Keep only subjects that have exactly three time points
subject_counts = df_all.groupby('record_id')['tp'].nunique()
valid_ids = subject_counts[subject_counts == 3].index
df_all = df_all[df_all['record_id'].isin(valid_ids)]

###############################
# Define assessment order and stroke mappings
###############################
assessment_order = ["FM-lex", "FM-uex", "BI", "MRS", "NIHSS"]
df_all['assessment'] = pd.Categorical(df_all['assessment'], categories=assessment_order, ordered=True)
df_all = df_all.sort_values('assessment')
assessments = [a for a in assessment_order if a in df_all['assessment'].unique()]
n_assess = len(assessments)

# Define stroke category line style and color mappings
line_style_map = {
    "INFARCT": "solid",
    "BLEEDING": "dashed"  # Matplotlib recognizes 'dashed' (or '--')
}
stroke_color_map = {
    "INFARCT": "#4d4d4d",  # gray
    "BLEEDING": "#e41a1c"   # red
}

# Get ordered time points for the x-axis
tp_categories = list(df_all['tp'].cat.categories)

###############################
# Define figure dimensions and create subplots
###############################
cm_to_inch = 0.3937
fig_width = 30 * cm_to_inch  * (18/30)
subplot_height = 9.5 * cm_to_inch * (18/30) 

fig, axes = plt.subplots(1, n_assess, figsize=(fig_width, subplot_height), dpi=300)
# Ensure axes is iterable when only one subplot exists
if n_assess == 1:
    axes = [axes]

###############################
# Plotting: Boxplots with overlaid spaghetti lines
###############################
for ax, assess in zip(axes, assessments):
    # Filter data for current assessment and use defined order for time points
    df_assess = df_all[df_all['assessment'] == assess].copy()
    order = tp_categories
    
    # --- Add boxplots (background distribution) ---
    sns.boxplot(x='tp', y='adjusted_score', data=df_assess, order=order,
                color='lightgray', ax=ax, showfliers=False)
    
    # --- Add spaghetti plots (subject trajectories) ---
    for subject, subject_df in df_assess.groupby('record_id'):
        subject_df = subject_df.sort_values('tp')
        # Convert categorical time points to their corresponding ordinal indices
        x_vals = [order.index(tp) for tp in subject_df['tp']]
        y_vals = subject_df['adjusted_score'].tolist()
        stroke_cat = subject_df['stroke_category'].iloc[0]
        line_style = line_style_map.get(stroke_cat, "solid")
        color = stroke_color_map.get(stroke_cat, "black")
        
        ax.plot(x_vals, y_vals, marker='o', markersize=2* (18/30), linestyle=line_style, color=color,
                linewidth=1.5* (18/30), alpha=0.7)
    
    # --- Update axes ---
    overall_min = df_assess['adjusted_score'].min()
    overall_max = df_assess['adjusted_score'].max()
    overall_range = overall_max - overall_min if overall_max != overall_min else 1
    y_lim_lower = overall_min - 0.15 * overall_range
    y_lim_upper = overall_max + 0.15 * overall_range
    ax.set_ylim(y_lim_lower, y_lim_upper)
    #ax.set_title(f"{assess}", fontsize=10)
    # Adjust the y-axis label based on the assessment type
    if assess in ['MRS', 'NIHSS']:
        ax.set_ylabel("Max - Score", fontsize=12)
    else:
        ax.set_ylabel("Score", fontsize=12)
    ax.set_xlabel("Time Point")
    ax.set_title(assess, fontsize=12)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_xlabel("Time Point", fontsize=12)
    ax.tick_params(labelsize=8)
    sns.despine(ax=ax, top=True, right=True)

plt.tight_layout()

# Save the figure as an SVG file
output_svg = os.path.join(svg_folder, "spaghetti_plots_by_assessment_stroke_category_sns.svg")
plt.savefig(output_svg, format="svg")
print(f"Saved combined spaghetti plots by assessment at {output_svg}")

plt.show()
