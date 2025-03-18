# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 10:25:02 2025

@author: arefk
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

###############################
# Setup directories
###############################
code_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(code_dir)
output_dir = os.path.join(parent_dir, "output")
figures_dir = os.path.join(output_dir, "figures")
os.makedirs(figures_dir, exist_ok=True)

# Create a dedicated folder for individual assessment SVG files
svg_folder = os.path.join(figures_dir, "assessment_svgs")
os.makedirs(svg_folder, exist_ok=True)

###############################
# Load and update FM data
###############################
csv_file_FM = os.path.join(output_dir, 'behavioral_data_cleaned_FM.csv')
df_FM = pd.read_csv(csv_file_FM)
# Add a new column "assessment" as 'FM-' concatenated with the 'position' value, then drop 'position'
df_FM['assessment'] = 'FM-' + df_FM['position']
df_FM = df_FM.drop('position', axis=1)

# Define the required score locations (order is important for later calculations)
required_locations = ['uex', 'lex']

###############################
# Define maximum scores (for reference)
###############################
max_uex_score = 126
max_lex_score = 86

###############################
# Load BI, MRS, NIHSS data
###############################
csv_file_BI_NIHSS = os.path.join(output_dir, 'behavioral_data_cleaned_BI_MRS_NIHSS.csv')
df_BI_NIHSS = pd.read_csv(csv_file_BI_NIHSS)
# (No extra columns are added here)

###############################
# Stitch the two DataFrames together and save as CSV
###############################
df_stitched = pd.concat([df_FM, df_BI_NIHSS], axis=0, ignore_index=True)
stitched_csv_file = os.path.join(output_dir, 'behavioral_data_cleaned_FM_BI_MRS_NIHSS_all_asssessment_types.csv')
df_stitched.to_csv(stitched_csv_file, index=False)
print(f"Stitched data saved to {stitched_csv_file}")

########################################
# Define recovery type assignment function
########################################
def assign_recovery_type(series):
    """
    Assigns one of the four recovery types based on a series of three values,
    allowing for non-strict (<= or >=) comparisons.
    """
    values = list(series)
    if len(values) < 3:
        return "Unclassified"
    a, b, c = values[0], values[1], values[2]
    
    # If all values are equal, then no trend is observed.
    if a == b == c:
        return "Unclassified"
    
    if a <= b and b <= c:
        return "Steady recovery"
    elif a >= b and b >= c:
        return "Steady decline"
    elif a <= b and b >= c:
        return "Early recovery with chronic decline"
    elif a >= b and b <= c:
        return "Late recovery with acute decline"
    else:
        return "Unclassified"

# Define line styles and fixed color mapping for recovery types
line_style_map = {
    "Steady recovery": "-",
    "Steady decline": "--",
    "Early recovery with chronic decline": ":",
    "Late recovery with acute decline": "-."
}
recovery_types = [
    "Steady recovery", 
    "Steady decline", 
    "Early recovery with chronic decline", 
    "Late recovery with acute decline"
]
colors = sns.color_palette("Set1", n_colors=4)
recovery_color_map = dict(zip(recovery_types, colors))

# Set measure and axis labels for plotting
measure = 'score'
xlabel = "Time Point (tp)"
ylabel = "Score"

# Reload the stitched data (or use df_stitched) and ensure 'tp' is categorical
df_plot = pd.read_csv(stitched_csv_file)
df_plot['tp'] = df_plot['tp'].astype('category')

# Remove any record_id (subject) that does not have all three timepoints
df_plot = df_plot.groupby('record_id').filter(lambda x: x['tp'].nunique() == 3)

# Define figure dimensions: 30 cm wide (converted to inches) and a fixed height per subplot
cm_to_inch = 0.3937
fig_width = 30 * cm_to_inch  
subplot_height = 6 * cm_to_inch  

# Determine unique assessment types and number of subplots
assessments = df_plot['assessment'].unique()
n_assess = len(assessments)

# Create a figure with one subplot per assessment type (side-by-side)
fig, axes = plt.subplots(1, n_assess, figsize=(fig_width, subplot_height), dpi=300)
if n_assess == 1:
    axes = [axes]

# Prepare a list to accumulate recovery percentages for all assessments
all_results = []

# Loop over each assessment type to create subplots and compute recovery percentages
for ax, assess in zip(axes, assessments):
    df_assess = df_plot[df_plot['assessment'] == assess].copy()
    # Get all record_ids that have a tp value of 2
    valid_record_ids = df_assess.loc[df_assess['tp'] == 2, 'record_id'].unique()
    # Filter the DataFrame to only include rows with those record_ids
    df_assess = df_assess[df_assess['record_id'].isin(valid_record_ids)]
    
    # Compute y-axis limits with a ±15% margin
    overall_min = df_assess[measure].min()
    overall_max = df_assess[measure].max()
    overall_range = overall_max - overall_min
    y_lim_lower = overall_min - 0.15 * overall_range
    y_lim_upper = overall_max + 0.15 * overall_range

    # Plot a boxplot for each timepoint (background distribution) on the combined subplot
    sns.boxplot(
        x='tp',
        y=measure,
        data=df_assess,
        color='lightgray',
        ax=ax,
        showfliers=False,
        width=0.6
    )
    
    # Plot each subject's trajectory as a spaghetti plot while computing recovery type per subject
    subject_recovery = {}
    for subject, subject_df in df_assess.groupby('record_id'):
        subject_df = subject_df.sort_values('tp')
        recovery_type = assign_recovery_type(subject_df[measure])
        subject_recovery[subject] = recovery_type
        x_vals = subject_df['tp'].cat.codes
        y_vals = subject_df[measure]
        
        line_style = line_style_map.get(recovery_type, "-")
        color = recovery_color_map.get(recovery_type, "black")
        
        ax.plot(x_vals, y_vals, marker='o', markersize=2, linewidth=1.2,
                linestyle=line_style, color=color, alpha=0.8)
    
    ax.set_ylim(y_lim_lower, y_lim_upper)
    ax.set_title(f"Assessment: {assess}", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(labelsize=8)
    ax.set_xlabel(xlabel, fontsize=12)
    sns.despine(ax=ax, top=True, right=True)
    
    # Compute recovery percentages for this assessment
    df_recovery = df_assess.drop_duplicates(subset='record_id').copy()
    df_recovery['recovery_type'] = df_recovery['record_id'].map(subject_recovery)
    total_subjects = len(df_recovery)
    for rtype in recovery_types:
        count = (df_recovery['recovery_type'] == rtype).sum()
        percent = (count / total_subjects * 100) if total_subjects > 0 else 0
        all_results.append({
            'Assessment': assess,
            'Recovery_Type': rtype,
            'Count': count,
            'Total': total_subjects,
            'Percentage': round(percent, 2)
        })
    
    # Now, also create and save an individual figure for this assessment as SVG
    fig_ind, ax_ind = plt.subplots(figsize=(9*cm_to_inch, 9*cm_to_inch), dpi=300)
    
    # Plot the same boxplot and spaghetti plot on the individual axis
    sns.boxplot(
        x='tp',
        y=measure,
        data=df_assess,
        color='lightgray',
        ax=ax_ind,
        showfliers=False,
        width=0.6
    )
    
    for subject, subject_df in df_assess.groupby('record_id'):
        subject_df = subject_df.sort_values('tp')
        recovery_type = assign_recovery_type(subject_df[measure])
        x_vals = subject_df['tp'].cat.codes
        y_vals = subject_df[measure]
        line_style = line_style_map.get(recovery_type, "-")
        color = recovery_color_map.get(recovery_type, "black")
        ax_ind.plot(x_vals, y_vals, marker='o', markersize=2, linewidth=1.2,
                    linestyle=line_style, color=color, alpha=0.8)
    
    ax_ind.set_ylim(y_lim_lower, y_lim_upper)
    ax_ind.set_title(f"Assessment: {assess}", fontsize=10)
    ax_ind.set_ylabel(ylabel, fontsize=12)
    ax_ind.set_xlabel(xlabel, fontsize=12)
    ax_ind.tick_params(labelsize=8)
    sns.despine(ax=ax_ind, top=True, right=True)
    
    # Save the individual plot as an SVG file in the dedicated folder
    svg_filename = os.path.join(svg_folder, f"{assess}.svg")
    fig_ind.savefig(svg_filename, format='svg', dpi=300)
    plt.close(fig_ind)
    print(f"Saved individual SVG for assessment '{assess}' at {svg_filename}")

plt.tight_layout(rect=[0, 0, 1, 1])
# Save the combined figure without a legend
combined_fig_filename = os.path.join(figures_dir, "spaghetti_plots_all_assessments.png")
plt.savefig(combined_fig_filename, dpi=300)
#plt.close(fig)
print(f"Saved combined spaghetti plots at {combined_fig_filename}")

# Save recovery percentages to CSV
recovery_csv_file = os.path.join(figures_dir, 'recovery_type_percentages_all_assessments.csv')
df_results = pd.DataFrame(all_results)
df_results.to_csv(recovery_csv_file, index=False)
print(f"Recovery type percentages saved at: {recovery_csv_file}")
