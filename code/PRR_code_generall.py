import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from matplotlib.ticker import MaxNLocator  # Import tick locator
import matplotlib as mpl
from scipy.stats import ttest_rel  # For the paired t-test

# ---------------------------
# File paths and environment
# ---------------------------
code_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(code_dir)
input_file_path = os.path.join(parent_dir, 'output', 'behavioral_data_cleaned_FM_BI_MRS_NIHSS_all_asssessment_types.csv')
figure_path = os.path.join(parent_dir, 'output', "figures", "prr")
os.makedirs(figure_path, exist_ok=True)

# ---------------------------
# Read the CSV file
# ---------------------------
df = pd.read_csv(input_file_path)

# Setup matplotlib font parameters
mpl.rcParams['font.family'] = 'Calibri'
mpl.rcParams['font.size'] = 12

# ---------------------------
# Set up assessment parameters
# ---------------------------
assessment_types = ['FM-lex','FM-uex', 'BI', 'MRS', 'NIHSS']
max_scores = {
    'FM-lex': 86,
    'FM-uex': 126,
    'BI': 100,
    'MRS': 5,
    'NIHSS': 42
}
assessment_map = {"FM-lex": "FM-LE", "FM-uex": "FM-UE", "BI": "BI", "MRS": "MRS", "NIHSS": "NIHSS"}

# ---------------------------
# Define helper function for Euclidean distance from a point to a line
# ---------------------------
def euclidean_distance(x, y, slope, intercept):
    return np.abs(slope * x - y + intercept) / np.sqrt(slope**2 + 1)

# ---------------------------
# Set up subplot for each assessment
# ---------------------------
fig, axs = plt.subplots(1, 5, figsize=(18/2.54, 5/2.54), dpi=300)
axs = axs.flatten()  # for easier indexing

# ---------------------------
# Prepare list to collect summary table results for each assessment
# ---------------------------
summary_results = []

# ---------------------------
# Prepare a list to collect outlier IDs for each assessment
# ---------------------------
outliers_list = []

# ---------------------------
# Loop over each assessment type
# ---------------------------
for i, assessment in enumerate(assessment_types):
    ax = axs[i]
    
    # Filter the DataFrame for the current assessment type
    df_assess = df[df["assessment"] == assessment].copy()
    
    # Drop an unwanted column if it exists
    if 'redcap_event_name' in df_assess.columns:
        df_assess.drop('redcap_event_name', axis=1, inplace=True)
    
    # Separate constant and time‐dependent columns
    constant_cols = ["record_id", "stroke_type", "recovery_type", "stroke_category", "assessment"]
    df_constants = df_assess[constant_cols].drop_duplicates()
    time_dependent_cols = [col for col in df_assess.columns if col not in constant_cols + ['tp']]
    
    # Pivot data so each record_id has a column per tp (e.g., adjusted_score_tp0, adjusted_score_tp2)
    df_time = df_assess.pivot(index='record_id', columns='tp', values=time_dependent_cols)
    df_time.columns = [f"{col[0]}_tp{col[1]}" for col in df_time.columns]  # flatten multi-index columns
    df_time.reset_index(inplace=True)
    df_wide = pd.merge(df_time, df_constants, on='record_id')
    df_wide = df_wide.drop_duplicates()

    # Calculate INITIAL_IMPAIRMENT and CHANGE_OBSERVED using the max score for the assessment
    max_score = max_scores.get(assessment, None)
    df_wide['INITIAL_IMPAIRMENT'] = max_score - df_wide['adjusted_score_tp0']
    df_wide['CHANGE_OBSERVED'] = df_wide['adjusted_score_tp2'] - df_wide['adjusted_score_tp0']
    
    # -----------------------------------------------
    # Optimize intercept for the PRR model (fixed slope = 0.7)
    # -----------------------------------------------
    fixed_slope = 0.7
    candidate_intercepts = np.linspace(-100, 100, 800)
    errors_intercept = []
    for intercept in candidate_intercepts:
        distances = euclidean_distance(df_wide['INITIAL_IMPAIRMENT'], df_wide['CHANGE_OBSERVED'],
                                       fixed_slope, intercept)
        errors_intercept.append(distances.sum())
    best_intercept = candidate_intercepts[np.argmin(errors_intercept)]
    
    # -----------------------------------------------
    # Optimize the slope for the Best Fit (empirical) model with best intercept fixed
    # -----------------------------------------------
    candidate_slopes = np.linspace(-1, 2.0, 500)
    errors_slope = []
    for slope in candidate_slopes:
        distances = euclidean_distance(df_wide['INITIAL_IMPAIRMENT'], df_wide['CHANGE_OBSERVED'],
                                       slope, best_intercept)
        errors_slope.append(distances.sum())
    best_slope = candidate_slopes[np.argmin(errors_slope)]
    
    # -----------------------------------------------
    # Plot the observed data and both fit lines
    # -----------------------------------------------
    ax.scatter(df_wide['INITIAL_IMPAIRMENT'], df_wide['CHANGE_OBSERVED'],
               color='#0072B2', alpha=0.6, s=15)
    x_vals = np.linspace(df_wide['INITIAL_IMPAIRMENT'].min(), df_wide['INITIAL_IMPAIRMENT'].max(), 100)
    
    # PRR fit (fixed slope 0.7) – dashed line
    y_prr = fixed_slope * x_vals + best_intercept
    ax.plot(x_vals, y_prr, linestyle='--', color='black', label="PRR (dashed)")
    
    # Best Fit (optimized slope) – solid line
    y_best = best_slope * x_vals + best_intercept
    ax.plot(x_vals, y_best, linestyle='-', color='black', label="Best Fit (solid)")
    
    # -----------------------------------------------
    # Identify outliers relative to the PRR line (optional)9118294973
    # -----------------------------------------------
    distances = euclidean_distance(df_wide['INITIAL_IMPAIRMENT'], df_wide['CHANGE_OBSERVED'],
                                   fixed_slope, best_intercept)
    Q1 = np.percentile(distances, 25)
    Q3 = np.percentile(distances, 75)
    IQR = Q3 - Q1
    threshold = Q3 + 1.5 * IQR
    outlier_mask = distances > threshold
    ax.scatter(df_wide.loc[outlier_mask, 'INITIAL_IMPAIRMENT'],
               df_wide.loc[outlier_mask, 'CHANGE_OBSERVED'],
               color='red', s=40, marker="x", alpha=0.5)
    
    # Save the outlier record IDs along with assessment information
    df_outliers = df_wide.loc[outlier_mask, ['record_id']].copy()
    df_outliers['Assessment'] = assessment_map[assessment]
    outliers_list.append(df_outliers)
    
    # Set subplot title and labels
    ax.set_title(assessment_map[assessment], fontsize=12)
    if i == 0:
        ax.set_ylabel('Change observed', fontsize=12)
    sns.despine(ax=ax, top=True, right=True)
    
    # Harmonize tick numbers for assessments other than MRS
    if assessment != 'MRS':
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))
    
    # -----------------------------------------------
    # Compute regression errors per subject for both models
    # (Mean Euclidean distance is used as the summary regression error)
    # -----------------------------------------------
    prr_errors = euclidean_distance(df_wide['INITIAL_IMPAIRMENT'], df_wide['CHANGE_OBSERVED'],
                                    fixed_slope, best_intercept)
    best_errors = euclidean_distance(df_wide['INITIAL_IMPAIRMENT'], df_wide['CHANGE_OBSERVED'],
                                     best_slope, best_intercept)
    reg_error_prr = (prr_errors.mean() / max_score) * 100
    reg_error_best = (best_errors.mean() / max_score) * 100
    
    # Run paired t-test to compare the error arrays
    t_stat, p_val = ttest_rel(prr_errors, best_errors)
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
    
    # Prepare the formula strings
    formula_prr = f"Y = {fixed_slope:.2f}X + {best_intercept:.2f}"
    formula_best = f"Y = {best_slope:.2f}X + {best_intercept:.2f}"
    
    summary_results.append({
        "Assessment Model": assessment_map[assessment],
        "Formula-PRR Fit": formula_prr,
        "Formula-Best Fit": formula_best,
        "Regression error PRR": f"{reg_error_prr:.2f}",
        "Regression Error Best Fit": f"{reg_error_best:.2f}",
        "Significantly differen error?": sig_str
    })

# -----------------------------------------------
# Finalize figure adjustments and save the figure
# -----------------------------------------------
plt.tight_layout(rect=[0, 0, 1, 1])
plt.subplots_adjust(bottom=0.25)
plt.subplots_adjust(wspace=0.37)
fig.supxlabel('Initial impairment', fontsize=12)
fig_filename = os.path.join(figure_path, "all_assessments_prr_vs_best_fit.svg")
plt.savefig(fig_filename)
plt.show()

# -----------------------------------------------
# Create the summary DataFrame and save the CSV
# -----------------------------------------------
summary_df = pd.DataFrame(summary_results, columns=[
    "Assessment Model",
    "Formula-PRR Fit",
    "Formula-Best Fit",
    "Regression error PRR",
    "Regression Error Best Fit",
    "Significantly differen error?"
])
summary_csv_path = os.path.join(parent_dir, 'output', "fit_parameters_whole_dataset_formula.csv")
summary_df.to_csv(summary_csv_path, index=False)
print("Transformed CSV saved at:", summary_csv_path)

# -----------------------------------------------
# Combine and save outlier record IDs for the whole dataset
# -----------------------------------------------
if outliers_list:
    outliers_df = pd.concat(outliers_list, ignore_index=True)
else:
    outliers_df = pd.DataFrame(columns=['record_id', 'Assessment'])

outliers_csv_path = os.path.join(parent_dir, 'output', "outliers_PRR_whole_dataset.csv")
outliers_df.to_csv(outliers_csv_path, index=False)
print("Outlier IDs saved at:", outliers_csv_path)
