import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
import matplotlib as mpl
from scipy import stats  # for statistical testing

###############################
# User setting: choose distance metric: "euclidean" or "vertical"
###############################
distance_metric = "euclidean"   # Change to "vertical" to use vertical distances

###############################
# Setup directories and paths
###############################
code_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(code_dir)
output_dir = os.path.join(parent_dir, "output")
figures_dir = os.path.join(output_dir, "figures", "pythonFigs")
os.makedirs(figures_dir, exist_ok=True)

# File paths for the mice data and the clustering output
# Here we use the classified mice data
mice_csv_file = os.path.join(output_dir, 'Mice_data_classified.csv')

###############################
# Load data
###############################
df_mice = pd.read_csv(mice_csv_file)
# For mice, we assume two timepoints (e.g., baseline at 3 and follow-up at 28)
# If needed, you can filter by the timepoint column (here "TimePointMerged")
# We also assume that each StudyID appears twice and that the "fixed_type" assignment
# is consistent for a given StudyID.

###############################
# Prepare data for analysis
###############################
# For the mice data the assessments are the three measurement columns:
assessment_types = ['C_PawDragPercent', 'GW_FootFault', 'RB_HindlimbDrop']
# Define friendly labels (as you use the same colors as in the humans code)
assessment_map = {
    'C_PawDragPercent': 'Paw Drags',
    'GW_FootFault': 'Foot Faults',
    'RB_HindlimbDrop': 'Hindlimb Drops'
}

# For mice, we assume that lower values indicate a better outcome.
# In the human analysis, initial impairment was computed as (max_score - score_tp0).
# Here we assume a minimum possible score of 0, so we define the initial impairment as:
# INITIAL_IMPAIRMENT = score at baseline (timepoint 3)  
# and CHANGE_OBSERVED = score at follow-up (timepoint 28) minus baseline.
# To scale error values we use the maximum observed value at baseline per assessment.
# You can alternatively set fixed maximum scores if available.
mice_baseline = df_mice[df_mice['TimePointMerged'] == 3]
max_scores = {assess: mice_baseline[assess].max() for assess in assessment_types}

# Define constant columns (identifiers) and the timepoint column.
constant_cols = ["StudyID", "Group", "fixed_type"]
time_col = "TimePointMerged"

# Pivot the data so that for each StudyID we have one row for each assessment per timepoint.
# Since our mice file is in a "long" format (each row is one timepoint with all assessment scores),
# we pivot the data such that each assessment gets its own column per timepoint.
# We set the new column names as <assessment>_tp<timepoint>.
# Pivot the data, using `pivot_table` to handle duplicates by aggregating
df_pivot = df_mice.pivot_table(index='StudyID', columns=time_col, values=assessment_types, aggfunc='first')
df_pivot.columns = [f"{col[0]}_tp{col[1]}" for col in df_pivot.columns]
df_pivot.reset_index(inplace=True)

# Merge the constant columns from (one copy of) the mice data (they are the same across timepoints)
df_constants = df_mice[constant_cols].drop_duplicates(subset='StudyID')
df_wide_all = pd.merge(df_pivot, df_constants, on="StudyID", how="left")

###############################
# Settings for plotting and regression
###############################
mpl.rcParams['font.family'] = 'Calibri'
mpl.rcParams['font.size'] = 12

###############################
# Distance function
###############################
def compute_distance(x, y, slope, intercept, metric="euclidean"):
    """
    Compute the distance between observed points and the line y = slope * x + intercept.
    If metric is "euclidean", uses the perpendicular (Euclidean) distance.
    If metric is "vertical", uses the vertical distance.
    """
    if metric == "vertical":
        return np.abs(y - (slope * x + intercept))
    else:
        return np.abs(slope * x - y + intercept) / np.sqrt(slope**2 + 1)

# Helper for regression errors always computed via Euclidean distance
def euclidean_distance(x, y, slope, intercept):
    return compute_distance(x, y, slope, intercept, metric="euclidean")

###############################
# Containers to store results
###############################
fit_parameters_all = []  # to store fit parameters for each assessment and cluster
errors_by_assessment = {}         # key: (assessment, cluster, model)
errors_by_assessment_clean = {}   # key: (assessment, cluster, model)
outliers_all = []      # to store outlier info (with a 'model' column)
significance_results = []  # for paired t-test results (if desired)

# We will also store wide-format data frames for each assessment for later use if needed.
df_wide_list = []

###############################
# Create combined plots: one for Best Fit and one for PRR
###############################
fig_size = (18/2.54, 5/2.54)  # width, height in inches
fig_best, axs_best = plt.subplots(1, len(assessment_types), figsize=fig_size, dpi=300)
fig_mice_prr, axs_prr = plt.subplots(1, len(assessment_types), figsize=fig_size, dpi=300)
axs_best = axs_best.flatten()
axs_prr = axs_prr.flatten()

# Loop over the three assessments
for i, assessment in enumerate(assessment_types):
    ax_best = axs_best[i]
    ax_prr = axs_prr[i]
    
    # For each assessment, extract the baseline (tp3) and follow-up (tp28) scores.
    baseline_col = f"{assessment}_tp3.0"
    followup_col = f"{assessment}_tp28.0"
    
    # Remove rows with missing values
    df_assess = df_wide_all.dropna(subset=[baseline_col, followup_col]).copy()
    if df_assess.empty:
        title = assessment_map.get(assessment, assessment)
        for ax in [ax_best, ax_prr]:
            ax.set_title(title)
            ax.text(0.5, 0.5, 'No data', horizontalalignment='center', verticalalignment='center')
        continue

    # For each cluster (good and poor) separately
    for cluster in ['good', 'poor']:
        cluster_color = '#E69F00' if cluster == 'good' else '#CC79A7'
        df_cluster_subset = df_assess[df_assess['fixed_type'] == cluster].copy()
        if df_cluster_subset.empty:
            continue
        
        # Calculate "INITIAL_IMPAIRMENT" and "CHANGE_OBSERVED"
        df_cluster_subset['INITIAL_IMPAIRMENT'] = df_cluster_subset[baseline_col] - df_cluster_subset[f"{assessment}_tp0.0"]
        df_cluster_subset['CHANGE_OBSERVED'] = -df_cluster_subset[followup_col] + df_cluster_subset[baseline_col]
        
        # Drop rows that have NaN values in the new columns
        df_cluster_subset = df_cluster_subset.dropna(subset=['INITIAL_IMPAIRMENT', 'CHANGE_OBSERVED'])

        # Save wide-format data for later (if needed)
        df_cluster_subset['assessment'] = assessment
        df_wide_list.append(df_cluster_subset.copy())
        
        # Determine the maximum baseline score for scaling error computations.
        max_score = max_scores.get(assessment, df_cluster_subset[baseline_col].max())
        
        # -----------------------------
        # Optimize intercept for the PRR model (fixed slope = 0.7)
        # -----------------------------
        fixed_slope = 0.7
        candidate_intercepts = np.linspace(-100, 100, 800)
        errors_intercept = []
        for intercept in candidate_intercepts:
            distances = compute_distance(df_cluster_subset['INITIAL_IMPAIRMENT'],
                                         df_cluster_subset['CHANGE_OBSERVED'],
                                         fixed_slope, intercept, metric="euclidean")
            errors_intercept.append(distances.sum())
        best_intercept = candidate_intercepts[np.argmin(errors_intercept)]
    
        # -----------------------------
        # Optimize the slope for the Best Fit model (empirical) with the best intercept fixed
        # -----------------------------
        candidate_slopes = np.linspace(-1, 2.0, 500)
        errors_slope = []
        for slope in candidate_slopes:
            distances = compute_distance(df_cluster_subset['INITIAL_IMPAIRMENT'],
                                         df_cluster_subset['CHANGE_OBSERVED'],
                                         slope, best_intercept, metric="euclidean")
            errors_slope.append(distances.sum())
        best_slope = candidate_slopes[np.argmin(errors_slope)]
    
        # -----------------------------
        # Calculate errors for both models
        # -----------------------------
        errors_prr = compute_distance(df_cluster_subset['INITIAL_IMPAIRMENT'],
                                      df_cluster_subset['CHANGE_OBSERVED'],
                                      fixed_slope, best_intercept, metric=distance_metric)
        avg_error_prr = (errors_prr.mean() / max_score) * 100

        errors_best = compute_distance(df_cluster_subset['INITIAL_IMPAIRMENT'],
                                       df_cluster_subset['CHANGE_OBSERVED'],
                                       best_slope, best_intercept, metric=distance_metric)
        avg_error_best = (errors_best.mean() / max_score) * 100

        # -----------------------------
        # Outlier detection for both models
        # -----------------------------
        # For PRR model
        Q1_prr = np.percentile(errors_prr, 25)
        Q3_prr = np.percentile(errors_prr, 75)
        IQR_prr = Q3_prr - Q1_prr
        threshold_prr = Q3_prr + 1.5 * IQR_prr
        outlier_mask_prr = errors_prr > threshold_prr
        # For Best Fit model
        Q1_best = np.percentile(errors_best, 25)
        Q3_best = np.percentile(errors_best, 75)
        IQR_best = Q3_best - Q1_best
        threshold_best = Q3_best + 1.5 * IQR_best
        outlier_mask_best = errors_best > threshold_best

        # Store clean errors for significance tests
        errors_by_assessment_clean[(assessment, cluster, "PRR")] = errors_prr[~outlier_mask_prr]
        errors_by_assessment_clean[(assessment, cluster, "Best Fit")] = errors_best[~outlier_mask_best]
        errors_by_assessment[(assessment, cluster, "PRR")] = errors_prr
        errors_by_assessment[(assessment, cluster, "Best Fit")] = errors_best

        # (Optional: Paired t-test comparing the errors between models for the same cluster)
        t_stat, p_val = stats.ttest_rel(euclidean_distance(df_cluster_subset['INITIAL_IMPAIRMENT'],
                                                           df_cluster_subset['CHANGE_OBSERVED'],
                                                           fixed_slope, best_intercept),
                                        euclidean_distance(df_cluster_subset['INITIAL_IMPAIRMENT'],
                                                           df_cluster_subset['CHANGE_OBSERVED'],
                                                           best_slope, best_intercept))
        if np.isnan(p_val):
            sig_str = "n/a"
        elif p_val < 0.001:
            sig_str = "Yes (p < 0.001)"
        elif p_val < 0.01:
            sig_str = "Yes (p < 0.01)"
        elif p_val < 0.05:
            sig_str = "Yes (p < 0.05)"
        else:
            sig_str = "No (p > 0.05)"
        
        significance_results.append({
            "assessment": assessment,
            "cluster": cluster,
            "model_comparison": "PRR vs Best Fit",
            "PRR_reg_error": avg_error_prr,
            "BestFit_reg_error": avg_error_best,
            "t_statistic": t_stat,
            "p_value": p_val,
            "significant": sig_str
        })
    
        # Save the fit parameters for both models
        fit_parameters_all.append({
            "assessment": assessment,
            "cluster": cluster,
            "model": "PRR",
            "slope": fixed_slope,
            "intercept": best_intercept,
            "n_points": len(df_cluster_subset),
            "avg_error": avg_error_prr
        })
        fit_parameters_all.append({
            "assessment": assessment,
            "cluster": cluster,
            "model": "Best Fit",
            "slope": best_slope,
            "intercept": best_intercept,
            "n_points": len(df_cluster_subset),
            "avg_error": avg_error_best
        })

        # -----------------------------
        # Plotting for both models (scatter points)
        # -----------------------------
        for ax in [ax_best, ax_prr]:
            ax.scatter(df_cluster_subset['INITIAL_IMPAIRMENT'], df_cluster_subset['CHANGE_OBSERVED'],
                       alpha=0.9, s=15, label=f"{cluster.capitalize()} Data", color=cluster_color)
        
        # In the Best Fit figure, plot the fitted line (solid)
        x_vals = np.linspace(df_cluster_subset['INITIAL_IMPAIRMENT'].min(),
                             df_cluster_subset['INITIAL_IMPAIRMENT'].max(), 100)
        y_best_line = best_slope * x_vals + best_intercept
        ax_best.plot(x_vals, y_best_line, linestyle='-', color=cluster_color,
                     label=f"{cluster.capitalize()} Best Fit")
        
        # In the PRR figure, plot the PRR line (dashed)
        y_prr_line = fixed_slope * x_vals + best_intercept
        ax_prr.plot(x_vals, y_prr_line, linestyle='--', color=cluster_color,
                    label=f"{cluster.capitalize()} PRR")
        
        # Mark outliers with red 'x'
        ax_best.scatter(df_cluster_subset.loc[outlier_mask_best, 'INITIAL_IMPAIRMENT'],
                        df_cluster_subset.loc[outlier_mask_best, 'CHANGE_OBSERVED'],
                        facecolors="red", s=40, marker='x', alpha=0.5)
        ax_prr.scatter(df_cluster_subset.loc[outlier_mask_prr, 'INITIAL_IMPAIRMENT'],
                       df_cluster_subset.loc[outlier_mask_prr, 'CHANGE_OBSERVED'],
                       facecolors="red", s=40, marker='x', alpha=0.5)

        # Save outlier details for both models
        df_outliers_prr = df_cluster_subset.loc[outlier_mask_prr, ['StudyID', 'INITIAL_IMPAIRMENT', 'CHANGE_OBSERVED']].copy()
        df_outliers_prr['assessment'] = assessment
        df_outliers_prr['cluster'] = cluster
        df_outliers_prr['distance'] = errors_prr[outlier_mask_prr].values
        df_outliers_prr['threshold'] = threshold_prr
        df_outliers_prr['model'] = "PRR"
        outliers_all.append(df_outliers_prr)

        df_outliers_best = df_cluster_subset.loc[outlier_mask_best, ['StudyID', 'INITIAL_IMPAIRMENT', 'CHANGE_OBSERVED']].copy()
        df_outliers_best['assessment'] = assessment
        df_outliers_best['cluster'] = cluster
        df_outliers_best['distance'] = errors_best[outlier_mask_best].values
        df_outliers_best['threshold'] = threshold_best
        df_outliers_best['model'] = "Best Fit"
        outliers_all.append(df_outliers_best)
    
    # Set subplot titles and labels
    title = assessment_map.get(assessment, assessment)
    ax_best.set_title(title, fontsize=12)
    ax_prr.set_title(title, fontsize=12)
    if i == 0:
        ax_best.set_ylabel('Change observed', fontsize=12)
        ax_prr.set_ylabel('Change observed', fontsize=12)
        
    # Format ticks for clarity
    ax_best.xaxis.set_major_locator(MaxNLocator(4))
    ax_best.yaxis.set_major_locator(MaxNLocator(4))
    ax_prr.xaxis.set_major_locator(MaxNLocator(4))
    ax_prr.yaxis.set_major_locator(MaxNLocator(4))
    
    sns.despine(ax=ax_best, top=True, right=True)
    sns.despine(ax=ax_prr, top=True, right=True)

# Add common x-axis labels to both figures
fig_best.supxlabel('Initial impairment', fontsize=12)
fig_mice_prr.supxlabel('Initial impairment', fontsize=12)

# Adjust layouts and save figures
fig_best.tight_layout(rect=[0, 0, 1, 1])
fig_best.subplots_adjust(bottom=0.25, wspace=0.37)
fig_mice_prr.tight_layout(rect=[0, 0, 1, 1])
fig_mice_prr.subplots_adjust(bottom=0.25, wspace=0.37)

best_fit_svg = os.path.join(figures_dir, "mice_assessments_best_fit_combined.svg")
best_fit_png = os.path.join(figures_dir, "mice_assessments_best_fit_combined.png")
prr_svg = os.path.join(figures_dir, "mice_assessments_prr_combined.svg")
prr_png = os.path.join(figures_dir, "mice_assessments_prr_combined.png")
fig_best.savefig(best_fit_svg)
fig_best.savefig(best_fit_png)
fig_mice_prr.savefig(prr_svg)
fig_mice_prr.savefig(prr_png)
plt.show()

###############################
# Export combined fit parameters to CSV
###############################
df_fit_params = pd.DataFrame(fit_parameters_all)

###############################
# Export combined outliers for both models to one CSV
###############################
if outliers_all:
    df_outliers_all = pd.concat(outliers_all, ignore_index=True)
    outliers_csv = os.path.join(output_dir, "combined_outliers_PRR_BestFit_good_poor_MICE.csv")
    df_outliers_all.to_csv(outliers_csv, index=False)

##############################################
# Create a formula-type pivot table with significance comparisons
##############################################
# Create a new column "Formula" formatted as "Y = MX + B"
df_fit_params['Formula'] = df_fit_params.apply(lambda row: f"Y = {row['slope']:.2f}X + {row['intercept']:.2f}", axis=1)

# Pivot the table so that each assessment appears once with separate columns for each combination of model and cluster
pivot_formula = df_fit_params.pivot_table(index='assessment', columns=['model', 'cluster'], values='Formula', aggfunc='first')
pivot_error = df_fit_params.pivot_table(index='assessment', columns=['model', 'cluster'], values='avg_error', aggfunc='first')

# Flatten the multi-index in the columns
pivot_formula.columns = [f"Formula-{model}_{cluster} cluster" for model, cluster in pivot_formula.columns]
pivot_error.columns = [f"Ø error-{model}-{cluster} cluster (%)" for model, cluster in pivot_error.columns]

pivot_formula.reset_index(inplace=True)
pivot_error.reset_index(inplace=True)

# Merge the formula and error pivot tables
df_merged_pivot = pd.merge(pivot_formula, pivot_error, on='assessment')
df_merged_pivot.rename(columns={'assessment': 'Assessment Model'}, inplace=True)

##############################################
# Compute Good vs. Poor significance for each model (PRR and Best Fit)
##############################################
good_vs_poor_list = []
assessments = sorted({key[0] for key in errors_by_assessment_clean.keys()})
for assessment in assessments:
    for model in ["PRR", "Best Fit"]:
        key_good = (assessment, "good", model)
        key_poor = (assessment, "poor", model)
        if key_good in errors_by_assessment_clean and key_poor in errors_by_assessment_clean:
            errors_good = np.asarray(errors_by_assessment_clean[(assessment, "good", model)], dtype=float)
            errors_poor = np.asarray(errors_by_assessment_clean[(assessment, "poor", model)], dtype=float)
            t_stat2, p_val2 = stats.ttest_ind(errors_good, errors_poor, equal_var=False)

            if np.isnan(p_val2):
                sig_str2 = "n/a"
            elif p_val2 < 0.001:
                sig_str2 = "Yes (p < 0.001)"
            elif p_val2 < 0.01:
                sig_str2 = "Yes (p < 0.01)"
            elif p_val2 < 0.05:
                sig_str2 = "Yes (p < 0.05)"
            else:
                sig_str2 = "No (p > 0.05)"
            good_vs_poor_list.append({
                "assessment": assessment,
                "model": model,
                "Good vs Poor": sig_str2
            })
            
df_good_vs_poor = pd.DataFrame(good_vs_poor_list)
df_good_vs_poor_pivot = df_good_vs_poor.pivot(index='assessment', columns='model', values='Good vs Poor')
df_good_vs_poor_pivot.columns = [f"Significantly Different Error?-{col}" for col in df_good_vs_poor_pivot.columns]
df_good_vs_poor_pivot.reset_index(inplace=True)
df_good_vs_poor_pivot.rename(columns={'assessment': 'Assessment Model'}, inplace=True)

# Merge significance info into the pivot table
df_merged_pivot = pd.merge(df_merged_pivot, df_good_vs_poor_pivot, on='Assessment Model', how='left')

# Save the transformed pivot table as CSV; append "MICE" to the file name.
pivot_csv = os.path.join(output_dir, "combined_fit_parameters_formula_PRR_BestFit_good_poor_MICE2.csv")
df_merged_pivot.to_csv(pivot_csv, index=False, encoding="utf-8-sig")
print("Transformed formula CSV saved at:", pivot_csv)
