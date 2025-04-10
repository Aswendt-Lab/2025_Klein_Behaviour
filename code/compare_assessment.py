# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 10:25:02 2025

@author: arefk
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
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

###############################
# Load and update FM data
###############################
csv_file_FM = os.path.join(output_dir, 'behavioral_data_cleaned_FM.csv')
df_stroketype = pd.read_csv(os.path.join(input_dir, 'Stroke_types.csv'))
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

# Stitch the two DataFrames together
df_stitched = pd.concat([df_FM, df_BI_NIHSS], axis=0, ignore_index=True)

# Merge the stitched DataFrame with df_stroketype based on "record_id"
df_merged = pd.merge(df_stitched, df_stroketype, on='record_id', how='left')

# Keep only groups (by record_id and assessment) that have exactly 3 rows
df_merged = df_merged.groupby(['record_id', 'assessment']).filter(lambda x: len(x) == 3)
df_merged = df_merged.groupby(['record_id', 'tp']).filter(lambda group: len(group) == 5)

import matplotlib as mpl
mpl.rcParams['font.family'] = 'Calibri'
mpl.rcParams['font.size'] = 12

########################################
# Apply score transformation for MRS and NIHSS
########################################
# Create a new column 'adjusted_score'. For MRS and NIHSS, adjust the score; otherwise, keep original.
def adjust_score(row):
    if row['assessment'] == 'MRS':
        return 5 - row['score']
    elif row['assessment'] == 'NIHSS':
        return 42 - row['score']
    else:
        return row['score']

df_merged['adjusted_score'] = df_merged.apply(adjust_score, axis=1)

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

# Ensure the data is sorted by timepoint to have the proper order for each group.
df_merged = df_merged.sort_values('tp')

# Use groupby-transform to compute the recovery type for each group and assign it to every row,
# using the adjusted_score for MRS and NIHSS.
df_merged['recovery_type'] = df_merged.groupby(['record_id', 'assessment'])['adjusted_score'] \
                                     .transform(assign_recovery_type)

df_merged = df_merged.drop_duplicates()

########################################
# Add stroke category information
########################################
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
df_merged["stroke_category"] = df_merged["stroke_type"].apply(categorize_stroke)

########################################
# Save the merged DataFrame with the new "recovery_type" and "stroke_category" columns
########################################
stitched_csv_file = os.path.join(output_dir, 'behavioral_data_cleaned_FM_BI_MRS_NIHSS_all_asssessment_types.csv')
df_merged.to_csv(stitched_csv_file, index=False)
print(f"Stitched and merged data saved to {stitched_csv_file}")

########################################
# Plotting setup and recovery percentage calculations
########################################
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
    "Unclassified"
]
# Define a color mapping using colors from the Okabe-Ito palette:
recovery_color_map = {
    "Steady recovery": "#009E73",                   # Bluish green for growth
    "Steady decline": "#D55E00",                    # Vermilion red for decline
    "Early recovery with chronic decline": "#E69F00",# Orange for early improvement but caution
    "Late recovery with acute decline": "#0072B2",   # Deep blue for late recovery after an acute drop
    "Unclassified": "#D9D9D9"
}

# Set measure and axis labels for plotting
measure = 'adjusted_score'
xlabel = "Time Point"
ylabel = "Score"  # This will be used only for FM-lex

# Reload the stitched data and ensure 'tp' is categorical
df_plot = pd.read_csv(stitched_csv_file)
df_plot['tp'] = df_plot['tp'].astype('category')

# Remove any record_id (subject) that does not have all three timepoints
df_plot = df_plot.groupby('record_id').filter(lambda x: x['tp'].nunique() == 3)

# Define the desired assessment order and enforce it
assessment_order = ["FM-lex", "FM-uex", "BI", "MRS", "NIHSS"]
# Option 1: Reorder the DataFrame using pd.Categorical
df_plot['assessment'] = pd.Categorical(df_plot['assessment'], categories=assessment_order, ordered=True)
df_plot = df_plot.sort_values('assessment')

# Get the unique assessments based on our custom order
assessments = [a for a in assessment_order if a in df_plot['assessment'].unique()]
n_assess = len(assessments)

# Define figure dimensions: 30 cm wide (converted to inches) and a fixed height per subplot
cm_to_inch = 0.3937
fig_width = 30 * cm_to_inch * (18/30)
subplot_height = 9.5 * cm_to_inch * (18/30)  

# Create a figure with one subplot per assessment type (side-by-side)
fig, axes = plt.subplots(1, n_assess, figsize=(fig_width, subplot_height), dpi=300)
if n_assess == 1:
    axes = [axes]

# Prepare a list to accumulate recovery percentages for all assessments
all_results = []

for ax, assess in zip(axes, assessments):
    # Get subset of data for the current assessment
    df_assess = df_plot[df_plot['assessment'] == assess].copy()
    
    # For assessments MRS and NIHSS, the score has already been adjusted.
    # For consistency, we use 'adjusted_score' throughout.
    
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
        
        ax.plot(x_vals, y_vals, marker='o', markersize=2*(18/30), linewidth=1.2*(18/30),
                linestyle=line_style, color=color, alpha=0.7)
    
    ax.set_ylim(y_lim_lower, y_lim_upper)
    ax.set_title(f"{assess}", fontsize=12)
    # Set y-label only for FM-lex and MRS
    if assess in ['FM-lex', 'MRS']:
        if assess == 'MRS':
            ax.set_ylabel("Max - Score", fontsize=12)
        else:
            ax.set_ylabel(ylabel, fontsize=12)
    else:
        ax.set_ylabel('')
        
    ax.set_xlabel(xlabel, fontsize=12)
    ax.tick_params(labelsize=8)
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
    
    # Now, create and save an individual figure for this assessment as SVG
    fig_ind, ax_ind = plt.subplots(figsize=(9 * cm_to_inch, 9 * cm_to_inch), dpi=300)
    
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
    # Set y-label only for FM-lex and MRS in the individual plots
    if assess in ['FM-lex', 'MRS']:
        if assess == 'MRS':
            ax_ind.set_ylabel("Max - Score", fontsize=12)
        else:
            ax_ind.set_ylabel(ylabel, fontsize=12)
    else:
        ax_ind.set_ylabel('')
        
    ax_ind.set_xlabel("Time Point", fontsize=12)
    ax_ind.tick_params(labelsize=8)
    sns.despine(ax=ax_ind, top=True, right=True)
    
    # Save the individual plot as an SVG file in the dedicated folder
    svg_filename = os.path.join(svg_folder, f"{assess}.svg")
    fig_ind.savefig(svg_filename, format='svg', dpi=300)
    plt.close(fig_ind)
    print(f"Saved individual SVG for assessment '{assess}' at {svg_filename}")

plt.tight_layout(rect=[0, 0, 1, 1])
plt.subplots_adjust(wspace=0.37)  # Adjust wspace to reduce horizontal space between subplots
# Save the combined figure without a legend
combined_fig_filename = os.path.join(figures_dir, "spaghetti_plots_all_assessments.svg")
plt.savefig(combined_fig_filename, dpi=300)
print(f"Saved combined spaghetti plots at {combined_fig_filename}")

# Save recovery percentages to CSV
recovery_csv_file = os.path.join(figures_dir, 'recovery_type_percentages_all_assessments.csv')
df_results = pd.DataFrame(all_results)
df_results.to_csv(recovery_csv_file, index=False)
print(f"Recovery type percentages saved at: {recovery_csv_file}")

########################################
# Additional Calculation:
# Calculate the percentage of non-"Steady recovery" subjects in the upper and lower half of the baseline (first timepoint)
########################################
baseline_results = []

# Loop through each assessment in our defined order
for assess in assessments:
    # Filter data for the current assessment
    df_assess = df_plot[df_plot['assessment'] == assess].copy()
    
    # Compute subject-level baseline score and recovery type
    subjects = []
    for subject, subject_df in df_assess.groupby('record_id'):
        subject_df = subject_df.sort_values('tp')
        # Calculate recovery type using the series of adjusted_score values
        rtype = assign_recovery_type(subject_df[measure])
        # Save baseline (first timepoint) score and recovery type
        subjects.append({
            'record_id': subject,
            'baseline_score': subject_df.iloc[0][measure],
            'recovery_type': rtype
        })
    
    # Create DataFrame with one row per subject for this assessment
    df_subjects = pd.DataFrame(subjects)
    
    # Filter to include only subjects that are NOT "Steady recovery"
    df_not_steady = df_subjects[df_subjects['recovery_type'] != "Steady recovery"]
    total_not_steady = len(df_not_steady)
    
    if total_not_steady == 0:
        print(f"No non-Steady recovery subjects found for assessment {assess}.")
        continue
    
    # Compute the median of the baseline scores for non-Steady recovery subjects
    median_baseline = df_not_steady['baseline_score'].median()
    
    # For this example, scores equal to the median are assigned to the lower half.
    lower_count = (df_not_steady['baseline_score'] <= median_baseline).sum()
    upper_count = (df_not_steady['baseline_score'] > median_baseline).sum()
    
    lower_percentage = round((lower_count / total_not_steady * 100), 2)
    upper_percentage = round((upper_count / total_not_steady * 100), 2)
    
    baseline_results.append({
        'Assessment': assess,
        'Total_non_Steady': total_not_steady,
        'Median_baseline': median_baseline,
        'Lower_half_count': lower_count,
        'Lower_half_percentage': lower_percentage,
        'Upper_half_count': upper_count,
        'Upper_half_percentage': upper_percentage
    })

# Save the baseline halves results to CSV
baseline_csv_file = os.path.join(figures_dir, 'non_steady_baseline_halves_percentages.csv')
df_baseline_results = pd.DataFrame(baseline_results)
df_baseline_results.to_csv(baseline_csv_file, index=False)
print(f"Non-Steady recovery baseline half percentages saved at: {baseline_csv_file}")
