# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 09:02:44 2025

@author: arefk
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

# -----------------------------
# Setup directories and file paths
# -----------------------------
code_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(code_dir)
output_dir = os.path.join(parent_dir, "output")
# Folder for recovery type figures (to avoid overwriting previous figures)
figures_dir_recovery = os.path.join(output_dir, "figures_recovery_types")
os.makedirs(figures_dir_recovery, exist_ok=True)

# -----------------------------
# Read in the CSV file with distance data
# -----------------------------
csv_file = os.path.join(output_dir, 'behavioral_data_with_distance_FM.csv')
df = pd.read_csv(csv_file)
df = df.dropna().reset_index(drop=True)

# Ensure variables have the correct type
df['tp'] = df['tp'].astype('category')
df['fixed_type'] = df['fixed_type'].astype('category')
df['record_id'] = df['record_id'].astype('string')

# Force an ordering on timepoints based on their sorted order
timepoints = sorted(df['tp'].unique())
df['tp'] = pd.Categorical(df['tp'], categories=timepoints, ordered=True)

# -----------------------------
# Filter: Only include subjects with complete data (i.e. available in all timepoints)
# -----------------------------
complete_subjects = df.groupby('record_id')['tp'].nunique()
complete_subjects = complete_subjects[complete_subjects == len(timepoints)].index
df = df[df['record_id'].isin(complete_subjects)]

# -----------------------------
# Determine recovery types using the new labels:
#
# Let T1, T2, T3 be the three time point values for distance_to_boundary.
#
# - Steady recovery:                T1 < T2 < T3
# - Steady decline:                 T1 > T2 > T3
# - Early recovery with chronic decline: T1 < T2 and T2 > T3
# - Late recovery with acute decline:    T1 > T2 and T2 < T3
# -----------------------------
def assign_recovery_type(series):
    """Assigns one of the four recovery types based on a series of three values."""
    values = list(series)
    if len(values) < 3:
        return "Unclassified"
    a, b, c = values[0], values[1], values[2]
    if a < b and b < c:
        return "Steady recovery"
    elif a > b and b > c:
        return "Steady decline"
    elif a < b and b > c:
        return "Early recovery with chronic decline"
    elif a > b and b < c:
        return "Late recovery with acute decline"
    else:
        return "Unclassified"

# Create a mapping for each subject's recovery type
recovery_type_map = {}
for subject, subject_df in df.groupby('record_id'):
    subject_df_sorted = subject_df.sort_values('tp')
    recovery_type = assign_recovery_type(subject_df_sorted['distance_to_boundary'])
    recovery_type_map[subject] = recovery_type

# Add the recovery type assignment to the dataframe
df['recovery_type'] = df['record_id'].map(recovery_type_map)

# Save the updated CSV with recovery types added
new_csv_file = os.path.join(output_dir, 'behavioral_data_with_distance_with_recovery_types.csv')
df.to_csv(new_csv_file, index=False)
print(f'New CSV with recovery types saved at: {new_csv_file}')

# -----------------------------
# Global Spaghetti Plot (All subjects together)
# -----------------------------
measure = 'distance_to_boundary'
ylabel = 'Distance to Boundary'
xlabel = 'Timepoint'

# Calculate overall min and max for the measure and extend by ±15%
overall_min = df[measure].min()
overall_max = df[measure].max()
overall_range = overall_max - overall_min
y_lim_lower = overall_min - 0.15 * overall_range
y_lim_upper = overall_max + 0.15 * overall_range

# Convert cm to inches (1 cm ≈ 0.3937 inches)
cm_to_inch = 0.3937
fig_width = 6 * cm_to_inch  
fig_height = 10 * cm_to_inch

fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300)

# Create a single boxplot for each timepoint over all subjects
sns.boxplot(
    x='tp',
    y=measure,
    data=df,
    color='lightgray',
    ax=ax,
    showfliers=False,
    width=0.6
)

# Define line styles and fixed color mapping for recovery types
line_style_map = {
    "Steady recovery": "-",
    "Steady decline": "--",
    "Early recovery with chronic decline": ":",
    "Late recovery with acute decline": "-."
}

recovery_types = ["Steady recovery", "Steady decline", 
                  "Early recovery with chronic decline", "Late recovery with acute decline"]
colors = sns.color_palette("Set1", n_colors=4)
recovery_color_map = dict(zip(recovery_types, colors))

# Plot each subject's trajectory
for subject, subject_df in df.groupby('record_id'):
    subject_df = subject_df.sort_values('tp')
    x_vals = subject_df['tp'].cat.codes  # numeric codes for timepoints
    y_vals = subject_df[measure]
    recovery_type = recovery_type_map[subject]
    line_style = line_style_map.get(recovery_type, "-")
    color = recovery_color_map.get(recovery_type, "black")
    
    ax.plot(x_vals, y_vals, marker='o', markersize=2, linewidth=1.2,
            linestyle=line_style, color=color, alpha=0.8)

# Create legend for recovery types
legend_elements = [
    Line2D([0], [0], color=recovery_color_map[rtype], lw=2, linestyle=line_style_map[rtype], label=rtype)
    for rtype in recovery_types
]
ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)

ax.set_title("Distance to Boundary Over Time by Recovery Type", fontsize=7)
ax.set_xlabel(xlabel, fontsize=12)
ax.set_ylabel(ylabel, fontsize=12)
ax.tick_params(labelsize=8)
ax.set_ylim(y_lim_lower, y_lim_upper)
sns.despine(ax=ax, top=True, right=True)
plt.tight_layout()

fig_svg = os.path.join(figures_dir_recovery, 'distance_boxplots_by_recovery_type_spaghetti.svg')
fig_png = os.path.join(figures_dir_recovery, 'distance_boxplots_by_recovery_type_spaghetti.png')
plt.savefig(fig_svg, dpi=300, bbox_inches='tight')
plt.savefig(fig_png, dpi=300, bbox_inches='tight')
plt.show()

print(f'Global plot saved as SVG: {fig_svg}')
print(f'Global plot saved as PNG: {fig_png}')

# -----------------------------
# Plotting: Spaghetti Plots for Each Fixed_Type Group (e.g., "good" and "bad")
# -----------------------------
for ft in df['fixed_type'].cat.categories:
    df_ft = df[df['fixed_type'] == ft]
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300)
    
    # Create boxplot for current group
    sns.boxplot(
        x='tp',
        y=measure,
        data=df_ft,
        color='lightgray',
        ax=ax,
        showfliers=False,
        width=0.6
    )
    
    # Plot subject trajectories for current group
    for subject, subject_df in df_ft.groupby('record_id'):
        subject_df = subject_df.sort_values('tp')
        x_vals = subject_df['tp'].cat.codes
        y_vals = subject_df[measure]
        recovery_type = recovery_type_map[subject]
        line_style = line_style_map.get(recovery_type, "-")
        color = recovery_color_map.get(recovery_type, "black")
        
        ax.plot(x_vals, y_vals, marker='o', markersize=2, linewidth=1.2,
                linestyle=line_style, color=color, alpha=0.8)
    
    ax.set_title(f"Distance to Boundary ({ft} group)", fontsize=7)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(labelsize=8)
    ax.set_ylim(y_lim_lower, y_lim_upper)
    sns.despine(ax=ax, top=True, right=True)
    plt.tight_layout()
    
    # Save the figure for this group
    ft_svg = os.path.join(figures_dir_recovery, f'distance_boxplots_by_recovery_type_spaghetti_{ft}.svg')
    ft_png = os.path.join(figures_dir_recovery, f'distance_boxplots_by_recovery_type_spaghetti_{ft}.png')
    plt.savefig(ft_svg, dpi=300, bbox_inches='tight')
    plt.savefig(ft_png, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f'Plot for fixed_type "{ft}" saved as SVG: {ft_svg}')
    print(f'Plot for fixed_type "{ft}" saved as PNG: {ft_png}')

# -----------------------------
# Compute and save recovery percentages
# -----------------------------
df_unique = df.drop_duplicates(subset='record_id')
total_subjects = len(df_unique)

results_text = []
results_text.append("Overall Recovery Type Percentages:")
for rtype in recovery_types:
    count = (df_unique['recovery_type'] == rtype).sum()
    percent = count / total_subjects * 100
    results_text.append(f"{rtype}: {percent:.2f}% ({count} out of {total_subjects})")
results_text.append("")
results_text.append("Recovery Type Percentages by Fixed_Type:")

for ft in df_unique['fixed_type'].cat.categories:
    group_df = df_unique[df_unique['fixed_type'] == ft]
    group_total = len(group_df)
    results_text.append(f"Fixed_Type: {ft}")
    for rtype in recovery_types:
        count = (group_df['recovery_type'] == rtype).sum()
        percent = count / group_total * 100 if group_total > 0 else 0
        results_text.append(f"  {rtype}: {percent:.2f}% ({count} out of {group_total})")
    results_text.append("")

print("\n".join(results_text))

results_txt_file = os.path.join(figures_dir_recovery, 'recovery_type_percentages.txt')
with open(results_txt_file, 'w') as f:
    f.write("\n".join(results_text))
    
print(f'Recovery type percentages saved at: {results_txt_file}')
