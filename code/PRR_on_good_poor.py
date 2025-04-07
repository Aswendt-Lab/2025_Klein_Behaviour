import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
import matplotlib as mpl

###############################
# Setup directories and paths
###############################
code_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(code_dir)
output_dir = os.path.join(parent_dir, "output")
figures_dir = os.path.join(output_dir, "figures")
os.makedirs(figures_dir, exist_ok=True)

# File paths for the PRR data and the clustering output
prr_csv_file = os.path.join(output_dir, 'behavioral_data_cleaned_FM_BI_MRS_NIHSS_all_asssessment_types.csv')
cluster_csv_file = os.path.join(output_dir, 'behavioral_data_with_distance_FM.csv')

###############################
# Load data
###############################
# Load PRR data
df_prr = pd.read_csv(prr_csv_file)

# Load clustering data (from the first code snippet)
df_cluster = pd.read_csv(cluster_csv_file)
# Filter for tp == 0 if applicable
if 'tp' in df_cluster.columns:
    df_cluster_tp0 = df_cluster[df_cluster['tp'] == 0].copy()
else:
    df_cluster_tp0 = df_cluster.copy()

# Keep only one row per record_id with the fixed_type assignment
df_cluster_unique = df_cluster_tp0[['record_id', 'fixed_type']].drop_duplicates()

# Merge the PRR data with clustering assignment on record_id
df_merged = pd.merge(df_prr, df_cluster_unique, on='record_id', how='left')

###############################
# Settings for PRR analysis
###############################
# Assessment types and corresponding maximum scores
assessment_types = ['FM-lex', 'FM-uex', 'BI', 'MRS', 'NIHSS']
max_scores = {
    'FM-lex': 86,
    'FM-uex': 126,
    'BI': 100,
    'MRS': 5,
    'NIHSS': 42
}
assessment_map = {"FM-lex": "FM-LE", "FM-uex": "FM-UE", "BI": "BI", "MRS": "MRS", "NIHSS": "NIHSS"}

# Set matplotlib parameters
mpl.rcParams['font.family'] = 'Calibri'
mpl.rcParams['font.size'] = 12

# Euclidean distance function used for optimization
def euclidean_distance(x, y, slope, intercept):
    return np.abs(slope * x - y + intercept) / np.sqrt(slope**2 + 1)

###############################
# Create combined plot for both clusters
###############################
# Set up a subplot grid: one row with one subplot per assessment
fig, axs = plt.subplots(1, len(assessment_types), figsize=(18/2.54, 5/2.54), dpi=300)
axs = axs.flatten()

# For each assessment, add both clusters' data into one subplot.
for i, assessment in enumerate(assessment_types):
    ax = axs[i]
    # Filter merged data for the current assessment
    df_assess = df_merged[df_merged["assessment"] == assessment].copy()
    if df_assess.empty:
        ax.set_title(assessment_map.get(assessment, assessment))
        ax.text(0.5, 0.5, 'No data', horizontalalignment='center', verticalalignment='center')
        continue

    # Loop over clusters: "good" (green) and "bad" (red)
    for cluster in ['good', 'bad']:
        # Set cluster-specific color
        cluster_color = 'green' if cluster == 'good' else 'red'
        df_assess_cluster = df_assess[df_assess['fixed_type'] == cluster].copy()
        if df_assess_cluster.empty:
            continue

        # Drop unwanted column if present
        if 'redcap_event_name' in df_assess_cluster.columns:
            df_assess_cluster.drop('redcap_event_name', axis=1, inplace=True)
        
        # Identify constant and time-dependent columns
        constant_cols = ["record_id", "stroke_type", "recovery_type", "stroke_category", "assessment"]
        time_dependent_cols = [col for col in df_assess_cluster.columns if col not in constant_cols + ['tp']]
        
        # Pivot the data so that each record_id gets its timepoints as separate columns
        df_time = df_assess_cluster.pivot(index='record_id', columns='tp', values=time_dependent_cols)
        # Flatten the multi-index columns (resulting in names like "adjusted_score_tp0")
        df_time.columns = [f"{col[0]}_tp{col[1]}" for col in df_time.columns]
        df_time.reset_index(inplace=True)
        
        # Merge with the constant columns (identical across timepoints)
        df_constants = df_assess_cluster[constant_cols].drop_duplicates()
        df_wide = pd.merge(df_time, df_constants, on='record_id')
        df_wide = df_wide.drop_duplicates()

        # Check that required score columns exist (tp0 and tp2)
        if not {'adjusted_score_tp0', 'adjusted_score_tp2'}.issubset(df_wide.columns):
            continue
        
        # Calculate INITIAL_IMPAIRMENT and CHANGE_OBSERVED
        max_score = max_scores.get(assessment, None)
        df_wide['INITIAL_IMPAIRMENT'] = max_score - df_wide['adjusted_score_tp0']
        df_wide['CHANGE_OBSERVED'] = df_wide['adjusted_score_tp2'] - df_wide['adjusted_score_tp0']
        
        # --- Optimize intercept with fixed slope (PRR) of 0.7 ---
        fixed_slope = 0.7
        candidate_intercepts = np.linspace(-100, 100, 200)
        errors_intercept = []
        for intercept in candidate_intercepts:
            distances = euclidean_distance(df_wide['INITIAL_IMPAIRMENT'], df_wide['CHANGE_OBSERVED'],
                                           fixed_slope, intercept)
            errors_intercept.append(distances.sum())
        best_intercept = candidate_intercepts[np.argmin(errors_intercept)]
        
        # --- Optimize slope with the best intercept fixed ---
        candidate_slopes = np.linspace(0.1, 2.0, 200)
        errors_slope = []
        for slope in candidate_slopes:
            distances = euclidean_distance(df_wide['INITIAL_IMPAIRMENT'], df_wide['CHANGE_OBSERVED'],
                                           slope, best_intercept)
            errors_slope.append(distances.sum())
        best_slope = candidate_slopes[np.argmin(errors_slope)]
        
        # Scatter plot the observed data points
        ax.scatter(df_wide['INITIAL_IMPAIRMENT'], df_wide['CHANGE_OBSERVED'],
                   color=cluster_color, alpha=0.6, s=15, label=f"{cluster.capitalize()} Data")
        
        # Generate x values for line plotting (spanning the range of INITIAL_IMPAIRMENT)
        x_vals = np.linspace(df_wide['INITIAL_IMPAIRMENT'].min(), df_wide['INITIAL_IMPAIRMENT'].max(), 100)
        
        # Plot the PRR line (dashed; fixed slope) for the current cluster
        y_prr = fixed_slope * x_vals + best_intercept
        ax.plot(x_vals, y_prr, linestyle='--', color=cluster_color, label=f"{cluster.capitalize()} PRR")
        
        # Plot the Best Fit line (solid; optimized slope) for the current cluster
        y_best = best_slope * x_vals + best_intercept
        ax.plot(x_vals, y_best, linestyle='-', color=cluster_color, label=f"{cluster.capitalize()} Best Fit")
        
        # --- Outlier detection based on distance to the PRR line ---
        distances = euclidean_distance(df_wide['INITIAL_IMPAIRMENT'], df_wide['CHANGE_OBSERVED'],
                                       fixed_slope, best_intercept)
        Q1 = np.percentile(distances, 25)
        Q3 = np.percentile(distances, 75)
        IQR = Q3 - Q1
        threshold = Q3 + 1.5 * IQR
        outlier_mask = distances > threshold
        
        # Mark outliers with an 'x' (using the same cluster color for consistency)
        ax.scatter(df_wide.loc[outlier_mask, 'INITIAL_IMPAIRMENT'],
                   df_wide.loc[outlier_mask, 'CHANGE_OBSERVED'],
                   facecolors='none', edgecolors=cluster_color, s=40, marker='x', alpha=0.8)
    
    # Set title and labels for the subplot
    ax.set_title(assessment_map.get(assessment, assessment), fontsize=12)
    if i == 0:
        ax.set_ylabel('Change observed', fontsize=12)
        
    # Add a general x label for all subplots
    fig.supxlabel('Initial impairment', fontsize=12)

    # Harmonize tick numbers if the assessment is not MRS
    if assessment != 'MRS':
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))
    
    # Remove top and right spines for a cleaner look
    sns.despine(ax=ax, top=True, right=True)


plt.tight_layout(rect=[0, 0, 1, 0.90])

plt.subplots_adjust(bottom=0.25)

plt.subplots_adjust(wspace=0.37) 
# Save the combined figure
fig_filename_svg = os.path.join(figures_dir, "all_assessments_prr_vs_best_fit_combined.svg")
fig_filename_png = os.path.join(figures_dir, "all_assessments_prr_vs_best_fit_combined.png")
plt.savefig(fig_filename_svg)
plt.savefig(fig_filename_png)
plt.show()
