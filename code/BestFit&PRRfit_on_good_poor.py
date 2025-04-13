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

# File paths for the PRR data and the clustering output
prr_csv_file = os.path.join(output_dir, 'behavioral_data_cleaned_FM_BI_MRS_NIHSS_all_asssessment_types.csv')
cluster_csv_file = os.path.join(output_dir, 'behavioral_data_with_distance_FM.csv')

###############################
# Load data
###############################
# Load PRR data
df_prr = pd.read_csv(prr_csv_file)

# Load clustering data
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
df_merged.to_csv(os.path.join(output_dir, "all_assessment_with_merged_good_poor_cluster.csv"))

###############################
# Settings for PRR analysis
###############################
assessment_types = ['FM-lex', 'FM-uex', 'BI', 'MRS', 'NIHSS']
max_scores = {
    'FM-lex': 86,
    'FM-uex': 126,
    'BI': 100,
    'MRS': 5,
    'NIHSS': 42
}
assessment_map = {"FM-lex": "FM-LE", "FM-uex": "FM-UE", "BI": "BI", "MRS": "MRS", "NIHSS": "NIHSS"}

# Set matplotlib parameters (will be applied to both figures)
mpl.rcParams['font.family'] = 'Calibri'
mpl.rcParams['font.size'] = 12

###############################
# Distance function
###############################
def compute_distance(x, y, slope, intercept, metric="euclidean"):
    """
    Compute the distance between observed points and the line y = slope*x + intercept.
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
# Prepare containers for combined results
###############################
# Store fit parameters for each assessment, cluster, and model type
fit_parameters_all = []

# Dictionaries to store raw errors and cleaned errors (after outlier removal) for each model.
errors_by_assessment = {}         # key: (assessment, cluster, model)
errors_by_assessment_clean = {}   # key: (assessment, cluster, model)

# List to store wide format data frames for each assessment and cluster
df_wide_all = []

# Lists to store outlier info for both models (with a column 'model' indicating "PRR" or "Best Fit")
outliers_all = []

# List to store paired t-test significance for model comparisons (if you want to compare PRR vs. Best Fit)
# (We leave that in case you wish to examine differences across models too)
significance_results = []

###############################
# Create two combined plots: one for Best Fit and one for PRR
###############################
fig_size = (18/2.54, 5/2.54)  # width, height in inches
fig_best, axs_best = plt.subplots(1, len(assessment_types), figsize=fig_size, dpi=300)
fig_prr, axs_prr = plt.subplots(1, len(assessment_types), figsize=fig_size, dpi=300)
axs_best = axs_best.flatten()
axs_prr = axs_prr.flatten()

for i, assessment in enumerate(assessment_types):
    ax_best = axs_best[i]
    ax_prr = axs_prr[i]
    
    # Filter merged data for the current assessment
    df_assess = df_merged[df_merged["assessment"] == assessment].copy()
    if df_assess.empty:
        title = assessment_map.get(assessment, assessment)
        for ax in [ax_best, ax_prr]:
            ax.set_title(title)
            ax.text(0.5, 0.5, 'No data', horizontalalignment='center', verticalalignment='center')
        continue

    for cluster in ['good', 'poor']:
        # Set cluster-specific colors (good: #E69F00, poor: #CC79A7)
        cluster_color = '#E69F00' if cluster == 'good' else '#CC79A7'
        df_assess_cluster = df_assess[df_assess['fixed_type'] == cluster].copy()
        if df_assess_cluster.empty:
            continue

        # Drop unwanted column if present
        if 'redcap_event_name' in df_assess_cluster.columns:
            df_assess_cluster.drop('redcap_event_name', axis=1, inplace=True)

        constant_cols = ["record_id", "stroke_type", "recovery_type", "stroke_category", "assessment"]
        time_dependent_cols = [col for col in df_assess_cluster.columns if col not in constant_cols + ['tp']]

        # Pivot the data so that each record_id gets its timepoints as separate columns
        df_time = df_assess_cluster.pivot(index='record_id', columns='tp', values=time_dependent_cols)
        df_time.columns = [f"{col[0]}_tp{col[1]}" for col in df_time.columns]
        df_time.reset_index(inplace=True)

        df_constants = df_assess_cluster[constant_cols].drop_duplicates()
        df_wide = pd.merge(df_time, df_constants, on='record_id')
        df_wide = df_wide.drop_duplicates()
        
        # Add cluster and assessment info
        df_wide['cluster'] = cluster
        df_wide['assessment'] = assessment
        
        if not {'adjusted_score_tp0', 'adjusted_score_tp2'}.issubset(df_wide.columns):
            continue

        max_score = max_scores.get(assessment, None)
        df_wide['INITIAL_IMPAIRMENT'] = max_score - df_wide['adjusted_score_tp0']
        df_wide['CHANGE_OBSERVED'] = df_wide['adjusted_score_tp2'] - df_wide['adjusted_score_tp0']

        # -----------------------------
        # Optimize intercept for the PRR model (fixed slope = 0.7)
        # -----------------------------
        fixed_slope = 0.7
        candidate_intercepts = np.linspace(-100, 100, 800)
        errors_intercept = []
        for intercept in candidate_intercepts:
            distances = compute_distance(df_wide['INITIAL_IMPAIRMENT'],
                                         df_wide['CHANGE_OBSERVED'],
                                         fixed_slope, intercept, metric="euclidean")
            errors_intercept.append(distances.sum())
        best_intercept = candidate_intercepts[np.argmin(errors_intercept)]
    
        # -----------------------------
        # Optimize the slope for the Best Fit (empirical) model with best intercept fixed
        # -----------------------------
        candidate_slopes = np.linspace(0.1, 2.0, 200)
        errors_slope = []
        for slope in candidate_slopes:
            distances = compute_distance(df_wide['INITIAL_IMPAIRMENT'],
                                         df_wide['CHANGE_OBSERVED'],
                                         slope, best_intercept, metric="euclidean")
            errors_slope.append(distances.sum())
        best_slope = candidate_slopes[np.argmin(errors_slope)]
    
        # -----------------------------
        # Calculate errors for both models
        # -----------------------------
        # PRR model (using fixed slope)
        errors_prr = compute_distance(df_wide['INITIAL_IMPAIRMENT'],
                                      df_wide['CHANGE_OBSERVED'],
                                      fixed_slope, best_intercept, metric=distance_metric)
        avg_error_prr = (errors_prr.mean() / max_score) * 100

        # Best Fit model (using optimized slope)
        errors_best = compute_distance(df_wide['INITIAL_IMPAIRMENT'],
                                       df_wide['CHANGE_OBSERVED'],
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

        # Store clean errors for t-tests (if needed)
        errors_by_assessment_clean[(assessment, cluster, "PRR")] = errors_prr[~outlier_mask_prr]
        errors_by_assessment_clean[(assessment, cluster, "Best Fit")] = errors_best[~outlier_mask_best]
        errors_by_assessment[(assessment, cluster, "PRR")] = errors_prr
        errors_by_assessment[(assessment, cluster, "Best Fit")] = errors_best

        # (Optional: Paired t-test comparing the errors between models for the same cluster)
        t_stat, p_val = stats.ttest_rel(euclidean_distance(df_wide['INITIAL_IMPAIRMENT'],
                                                           df_wide['CHANGE_OBSERVED'],
                                                           fixed_slope, best_intercept),
                                        euclidean_distance(df_wide['INITIAL_IMPAIRMENT'],
                                                           df_wide['CHANGE_OBSERVED'],
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
    
        # Save the fit parameters for both models (include regression errors)
        fit_parameters_all.append({
            "assessment": assessment,
            "cluster": cluster,
            "model": "PRR",
            "slope": fixed_slope,
            "intercept": best_intercept,
            "n_points": len(df_wide),
            "avg_error": avg_error_prr
        })
        fit_parameters_all.append({
            "assessment": assessment,
            "cluster": cluster,
            "model": "Best Fit",
            "slope": best_slope,
            "intercept": best_intercept,
            "n_points": len(df_wide),
            "avg_error": avg_error_best
        })

        # -----------------------------
        # Plotting for both models (using the same scatter points)
        # -----------------------------
        for ax in [ax_best, ax_prr]:
            ax.scatter(df_wide['INITIAL_IMPAIRMENT'], df_wide['CHANGE_OBSERVED'],
                       alpha=0.9, s=15, label=f"{cluster.capitalize()} Data", color=cluster_color)
        
        # In the Best Fit figure, plot the fitted line using solid line in cluster-specific color
        x_vals = np.linspace(df_wide['INITIAL_IMPAIRMENT'].min(), df_wide['INITIAL_IMPAIRMENT'].max(), 100)
        y_best_line = best_slope * x_vals + best_intercept
        ax_best.plot(x_vals, y_best_line, linestyle='-', color=cluster_color,
                     label=f"{cluster.capitalize()} Best Fit")
        
        # In the PRR figure, plot the PRR line (dashed in cluster-specific color)
        y_prr_line = fixed_slope * x_vals + best_intercept
        ax_prr.plot(x_vals, y_prr_line, linestyle='--', color=cluster_color,
                    label=f"{cluster.capitalize()} PRR")
        
        # Mark outliers on both figures with red 'x'
        ax_best.scatter(df_wide.loc[outlier_mask_best, 'INITIAL_IMPAIRMENT'],
                        df_wide.loc[outlier_mask_best, 'CHANGE_OBSERVED'],
                        facecolors="red", s=40, marker='x', alpha=0.5)
        ax_prr.scatter(df_wide.loc[outlier_mask_prr, 'INITIAL_IMPAIRMENT'],
                       df_wide.loc[outlier_mask_prr, 'CHANGE_OBSERVED'],
                       facecolors="red", s=40, marker='x', alpha=0.5)

        # Save outlier details for both models with a 'model' column
        df_outliers_prr = df_wide.loc[outlier_mask_prr, ['record_id', 'INITIAL_IMPAIRMENT', 'CHANGE_OBSERVED']].copy()
        df_outliers_prr['assessment'] = assessment
        df_outliers_prr['cluster'] = cluster
        df_outliers_prr['distance'] = errors_prr[outlier_mask_prr].values
        df_outliers_prr['threshold'] = threshold_prr
        df_outliers_prr['model'] = "PRR"
        outliers_all.append(df_outliers_prr)

        df_outliers_best = df_wide.loc[outlier_mask_best, ['record_id', 'INITIAL_IMPAIRMENT', 'CHANGE_OBSERVED']].copy()
        df_outliers_best['assessment'] = assessment
        df_outliers_best['cluster'] = cluster
        df_outliers_best['distance'] = errors_best[outlier_mask_best].values
        df_outliers_best['threshold'] = threshold_best
        df_outliers_best['model'] = "Best Fit"
        outliers_all.append(df_outliers_best)

        df_wide_all.append(df_wide.copy())
        
    # Set subplot titles (without individual x-axis labels)
    title = assessment_map.get(assessment, assessment)
    ax_best.set_title(title, fontsize=12)
    ax_prr.set_title(title, fontsize=12)
    if i == 0:
        ax_best.set_ylabel('Change observed', fontsize=12)
        ax_prr.set_ylabel('Change observed', fontsize=12)
        
    # Set tick formatting if needed
    if assessment != 'MRS':
        ax_best.xaxis.set_major_locator(MaxNLocator(4))
        ax_best.yaxis.set_major_locator(MaxNLocator(4))
        ax_prr.xaxis.set_major_locator(MaxNLocator(4))
        ax_prr.yaxis.set_major_locator(MaxNLocator(4))
    
    sns.despine(ax=ax_best, top=True, right=True)
    sns.despine(ax=ax_prr, top=True, right=True)

# Add common x-axis labels for both figures
fig_best.supxlabel('Initial impairment', fontsize=12)
fig_prr.supxlabel('Initial impairment', fontsize=12)

# Apply tight layout and adjust subplots for each figure individually
fig_best.tight_layout(rect=[0, 0, 1, 1])
fig_best.subplots_adjust(bottom=0.25, wspace=0.37)
fig_prr.tight_layout(rect=[0, 0, 1, 1])
fig_prr.subplots_adjust(bottom=0.25, wspace=0.37)

# Save the two figures
best_fit_svg = os.path.join(figures_dir, "all_assessments_best_fit_combined.svg")
best_fit_png = os.path.join(figures_dir, "all_assessments_best_fit_combined.png")
prr_svg = os.path.join(figures_dir, "all_assessments_prr_combined.svg")
prr_png = os.path.join(figures_dir, "all_assessments_prr_combined.png")
fig_best.savefig(best_fit_svg)
fig_best.savefig(best_fit_png)
fig_prr.savefig(prr_svg)
fig_prr.savefig(prr_png)
plt.show()

###############################
# Export combined fit parameters to CSV
###############################
# Create a DataFrame with all fit parameters (i.e. both good and poor clusters together)
df_fit_params = pd.DataFrame(fit_parameters_all)

###############################
# Export combined outliers for both models to one CSV
###############################
if outliers_all:
    df_outliers_all = pd.concat(outliers_all, ignore_index=True)
    outliers_csv = os.path.join(output_dir, "combined_outliers_PRR_BestFit_good_poor.csv")
    df_outliers_all.to_csv(outliers_csv, index=False)

##############################################
# Create a formula-type pivot table with significance comparisons
##############################################
# Create a new column "Formula" formatted as "Y = MX + B"
df_fit_params['Formula'] = df_fit_params.apply(lambda row: f"Y = {row['slope']:.2f}X + {row['intercept']:.2f}", axis=1)

# Pivot the table so that each assessment appears once with separate columns for each combination of model and cluster
pivot_formula = df_fit_params.pivot_table(index='assessment', columns=['model', 'cluster'], values='Formula', aggfunc='first')
pivot_error = df_fit_params.pivot_table(index='assessment', columns=['model', 'cluster'], values='avg_error', aggfunc='first')

# Flatten the column multi-index for easier reading
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
# Get unique assessments from the keys of errors_by_assessment_clean
assessments = sorted({key[0] for key in errors_by_assessment_clean.keys()})
for assessment in assessments:
    for model in ["PRR", "Best Fit"]:
        key_good = (assessment, "good", model)
        key_poor = (assessment, "poor", model)
        if key_good in errors_by_assessment_clean and key_poor in errors_by_assessment_clean:
            # Use an independent t-test (unequal variances)
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
            
# Create dataframes for the significance results
df_good_vs_poor = pd.DataFrame(good_vs_poor_list)

# Pivot such that for each assessment we have separate columns for each model's significance comparison
df_good_vs_poor_pivot = df_good_vs_poor.pivot(index='assessment', columns='model', values='Good vs Poor')
df_good_vs_poor_pivot.columns = [f"Good vs Poor Significance-{col}" for col in df_good_vs_poor_pivot.columns]
df_good_vs_poor_pivot.reset_index(inplace=True)
df_good_vs_poor_pivot.rename(columns={'assessment': 'Assessment Model'}, inplace=True)

# Merge the significance columns into the merged pivot table
df_merged_pivot = pd.merge(df_merged_pivot, df_good_vs_poor_pivot, on='Assessment Model', how='left')

# Save the transformed pivot table as CSV
pivot_csv = os.path.join(output_dir, "combined_fit_parameters_formula_PRR_BestFit_good_poor.csv")
df_merged_pivot.to_csv(pivot_csv, index=False, encoding="utf-8-sig")
print("Transformed formula CSV saved at:", pivot_csv)
