import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from scipy.stats import ttest_rel
import matplotlib as mpl

# Setup matplotlib font parameters
mpl.rcParams['font.family'] = 'Calibri'
mpl.rcParams['font.size'] = 12
# === File paths ===
code_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(code_dir)
input_file_path = os.path.join(parent_dir, 'output', 'Mice_data_classified.csv')
output_dir = os.path.join(parent_dir, 'output')
figure_dir = os.path.join(parent_dir, 'output', "figures", "pythonFigs", "prr_mice")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(figure_dir, exist_ok=True)

# === Read the CSV file ===
df = pd.read_csv(input_file_path)

# === Step 1: Subset the DataFrame to the desired columns ===
cols_to_keep = ['StudyID', 'TimePointMerged', 'C_PawDragPercent', 'GW_FootFault', 'RB_HindlimbDrop', 'Group']
df_clean = df[cols_to_keep].copy()


# === Step 2: Filter rows for Group == "Stroke" ===
df_clean = df_clean[df_clean['Group'] == "Stroke"]

# === Step 3: Filter rows for the desired timepoints ===
desired_timepoints = [0, 3, 28]
df_clean = df_clean[df_clean['TimePointMerged'].isin(desired_timepoints)]

# === Step 4: Convert measurement columns to numeric and drop rows with NaNs or non-numeric entries ===
measurement_cols = ['C_PawDragPercent', 'GW_FootFault', 'RB_HindlimbDrop']
df_clean[measurement_cols] = df_clean[measurement_cols].apply(pd.to_numeric, errors='coerce')
df_clean = df_clean.dropna(subset=measurement_cols)

# === Step 5: Keep only StudyIDs that have exactly the required timepoints ===
required_timepoints = set(desired_timepoints)
study_timepoints = df_clean.groupby('StudyID')['TimePointMerged'].apply(set)
studyids_to_keep = study_timepoints[study_timepoints == required_timepoints].index

df_final = df_clean[df_clean['StudyID'].isin(studyids_to_keep)].copy()

# === Assessment types ===
assessment_types = ['C_PawDragPercent', 'GW_FootFault', 'RB_HindlimbDrop']
assessment_mapping = {
    'C_PawDragPercent': 'Paw Drags',
    'GW_FootFault': 'Foot Faults',
    'RB_HindlimbDrop': 'Hindlimb Drops'
}

# Function to calculate Euclidean distance
def euclidean_distance(x, y, slope, intercept):
    return np.abs(slope * x - y + intercept) / np.sqrt(slope**2 + 1)

# Lists to store fit parameters and outlier information
fit_parameters_list = []
outliers_list = []
summary_results = []

# Set up the subplot grid: 1 row x 3 columns
fig, axs = plt.subplots(1, 3, figsize=(18/2.54, 5/2.54), dpi=300)
axs = axs.flatten()

# Loop over each assessment type (measurement)
for i, assessment in enumerate(assessment_types):
    # Extract data for the current assessment
    df_assess = df_final[['StudyID', 'TimePointMerged', assessment]].copy()
    df_assess = df_assess.rename(columns={assessment: 'value'})
    df_wide = df_assess.pivot(index='StudyID', columns='TimePointMerged', values='value')
    df_wide = df_wide.dropna(subset=[0, 3, 28])
    df_wide.columns = [f"{assessment}_tp{int(col)}" for col in df_wide.columns]
    df_wide.reset_index(inplace=True)

    # Compute derived columns
    df_wide['INITIAL_IMPAIRMENT'] =  df_wide[f"{assessment}_tp3"] - df_wide[f"{assessment}_tp0"]
    df_wide['CHANGE_OBSERVED'] = -df_wide[f"{assessment}_tp28"] + df_wide[f"{assessment}_tp3"]

    # ---- Optimize intercept with fixed slope of 0.7 (PRR) ----
    fixed_slope = 0.7
    candidate_intercepts = np.linspace(-100, 100, 800)
    errors_intercept = []
    for intercept in candidate_intercepts:
        distances = euclidean_distance(df_wide['INITIAL_IMPAIRMENT'], df_wide['CHANGE_OBSERVED'],
                                       fixed_slope, intercept)
        errors_intercept.append(distances.sum())
    best_intercept = candidate_intercepts[np.argmin(errors_intercept)]

    # ---- Optimize slope with the best intercept fixed ----
    candidate_slopes = np.linspace(-1, 2.0, 500)
    errors_slope = []
    for slope in candidate_slopes:
        distances = euclidean_distance(df_wide['INITIAL_IMPAIRMENT'], df_wide['CHANGE_OBSERVED'],
                                       slope, best_intercept)
        errors_slope.append(distances.sum())
    best_slope = candidate_slopes[np.argmin(errors_slope)]

    # Calculate regression errors
    prr_errors = euclidean_distance(df_wide['INITIAL_IMPAIRMENT'], df_wide['CHANGE_OBSERVED'], fixed_slope, best_intercept)
    best_errors = euclidean_distance(df_wide['INITIAL_IMPAIRMENT'], df_wide['CHANGE_OBSERVED'], best_slope, best_intercept)

    # Calculate mean regression errors (as percentages of the max observed value)
    max_score = np.max(df_wide['INITIAL_IMPAIRMENT'])
    reg_error_prr = (prr_errors.mean() / max_score) * 100
    reg_error_best = (best_errors.mean() / max_score) * 100

    # Paired t-test to compare the errors
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

    # Store the results
    summary_results.append({
        "Assessment Test": assessment_mapping[assessment],
        "Formula-PRR": f"Y = {fixed_slope:.2f}X + {best_intercept:.2f}",
        "Formula-Best Fit": f"Y = {best_slope:.2f}X + {best_intercept:.2f}",
        "Ø Error-PRR %": f"{reg_error_prr:.2f}",
        "Ø Error-Best Fit %": f"{reg_error_best:.2f}",
        "Significantly Different Error?": sig_str
    })
    
    # ---- Plot the data ----
    ax = axs[i]
    ax.scatter(df_wide['INITIAL_IMPAIRMENT'], df_wide['CHANGE_OBSERVED'], color='#0072B2', alpha=0.6, s=15)

    # Generate x values for the line plots
    x_vals = np.linspace(df_wide['INITIAL_IMPAIRMENT'].min(), df_wide['INITIAL_IMPAIRMENT'].max(), 100)
    y_prr = fixed_slope * x_vals + best_intercept
    y_best = best_slope * x_vals + best_intercept

    ax.plot(x_vals, y_prr, linestyle='--', color='black', label="PRR (dashed)")
    ax.plot(x_vals, y_best, linestyle='-', color='black', label="Best fit (solid)")

    # --- Identify outliers based on Euclidean distance to the PRR line ---
    distances = euclidean_distance(df_wide['INITIAL_IMPAIRMENT'], df_wide['CHANGE_OBSERVED'],
                                   fixed_slope, best_intercept)
    Q1 = np.percentile(distances, 25)
    Q3 = np.percentile(distances, 75)
    IQR = Q3 - Q1
    threshold = Q3 + 1.5 * IQR
    outlier_mask = distances > threshold

    # Record outlier information
    for study_id in df_wide.loc[outlier_mask, 'StudyID']:
        outliers_list.append({
            "assessment": assessment,
            "StudyID": study_id
        })

    # Mark outliers with red "x" markers
    ax.scatter(df_wide.loc[outlier_mask, 'INITIAL_IMPAIRMENT'],
               df_wide.loc[outlier_mask, 'CHANGE_OBSERVED'],
               color='red', s=40, marker="x", alpha=0.5)

    # Set title and labels for subplot
    ax.set_title(assessment_mapping[assessment], fontsize=12)
    if i == 0:
        ax.set_ylabel('Change observed', fontsize=12)

    # Remove top and right spines using seaborn
    sns.despine(ax=ax, top=True, right=True)

    # Harmonize tick numbers (for consistency across plots)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))

# Adjust layout to provide extra space at the bottom for the tables


# Add a general x label for all subplots
fig.supxlabel('Initial impairment', fontsize=12)
#plt.subplots_adjust(bottom=0.3)
plt.tight_layout(rect=[0, 0, 1, 1])
# Adjust layouts and save figures
plt.subplots_adjust(bottom=0.25, wspace=0.37)
# Save and show the figure
fig_path = os.path.join(figure_dir, "all_assessments_prr_vs_best_fit_mice.svg")
plt.savefig(fig_path, dpi=300)
plt.show()

# Save the fit parameters to CSV
fit_parameters_df = pd.DataFrame(fit_parameters_list)
fit_parameters_csv_path = os.path.join(parent_dir, 'output', "fit_parameters_mice.csv")
#fit_parameters_df.to_csv(fit_parameters_csv_path, index=False)

# Save the outliers information to CSV
outliers_df = pd.DataFrame(outliers_list)
outliers_csv_path = os.path.join(parent_dir, 'output', "outlier_mice.csv")
outliers_df.to_csv(outliers_csv_path, index=False)

# Save the summary results to CSV
summary_csv_path = os.path.join(parent_dir, 'output', "Mice_fit_parameters_whole_dataset_formula.csv")
summary_df = pd.DataFrame(summary_results)
summary_df.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")

print(f"Summary results, fit parameters, and outliers have been saved.")
