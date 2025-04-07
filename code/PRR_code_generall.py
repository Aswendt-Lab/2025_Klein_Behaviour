import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from matplotlib.ticker import MaxNLocator  # Import tick locator
import matplotlib as mpl
# File paths
code_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(code_dir)
input_file_path = os.path.join(parent_dir, 'output', 'behavioral_data_cleaned_FM_BI_MRS_NIHSS_all_asssessment_types.csv')
pppath = os.path.join(parent_dir, 'output', "figures", "prr")
os.makedirs(pppath, exist_ok=True)

# Read the CSV file
df = pd.read_csv(input_file_path)

mpl.rcParams['font.family'] = 'Calibri'
mpl.rcParams['font.size'] = 12

# Assessment types and corresponding max scores
assessment_types = ['FM-lex','FM-uex',  'BI', 'MRS', 'NIHSS']
max_scores = {
    'FM-lex': 86,
    'FM-uex': 126,
    'BI': 100,
    'MRS': 5,
    'NIHSS': 42
}
assessment_map = {"FM-lex":"FM-LE", "FM-uex":"FM-UE", "BI":"BI", "MRS":"MRS", "NIHSS":"NIHSS"}

# Function to calculate Euclidean distance from a point (x, y) to a line (slope, intercept)
def euclidean_distance(x, y, slope, intercept):
    return np.abs(slope * x - y + intercept) / np.sqrt(slope**2 + 1)

# Set up the subplot grid: 1 row x 5 columns.
fig, axs = plt.subplots(1, 5, figsize=(18/2.54,5/2.54), dpi=300)
axs = axs.flatten()  # flatten for easier indexing

# Lists to collect fit parameters and outlier information
fit_parameters_list = []
outliers_list = []

# Loop over assessment types
for i, assessment in enumerate(assessment_types):
    ax = axs[i]
    # Filter the DataFrame for the current assessment
    df_assess = df[df["assessment"] == assessment].copy()
    # Drop unwanted column if it exists
    if 'redcap_event_name' in df_assess.columns:
        df_assess.drop('redcap_event_name', axis=1, inplace=True)
    
    # Constant columns
    constant_cols = ["record_id", "stroke_type", "recovery_type", "stroke_category", "assessment"]
    df_constants = df_assess[constant_cols].drop_duplicates()
    time_dependent_cols = [col for col in df_assess.columns if col not in constant_cols + ['tp']]
    # Pivot data: each record_id will have a column for each tp (e.g., score_tp0, score_tp2)
    df_time = df_assess.pivot(index='record_id', columns='tp', values=time_dependent_cols)
    # Flatten the multi-index columns
    df_time.columns = [f"{col[0]}_tp{col[1]}" for col in df_time.columns]
    df_time.reset_index(inplace=True)
    df_wide = pd.merge(df_time, df_constants, on='record_id')
    df_wide = df_wide.drop_duplicates()

    # Calculate INITIAL_IMPAIRMENT and CHANGE_OBSERVED using the max score for this assessment
    max_score = max_scores.get(assessment, None)
  
    df_wide['INITIAL_IMPAIRMENT'] = max_score - df_wide['adjusted_score_tp0']
    df_wide['CHANGE_OBSERVED'] = df_wide['adjusted_score_tp2'] - df_wide['adjusted_score_tp0']

    # ---- Optimize intercept with fixed slope of 0.7 ----
    fixed_slope = 0.7
    candidate_intercepts = np.linspace(-100, 100, 200)
    errors_intercept = []
    for intercept in candidate_intercepts:
        distances = euclidean_distance(df_wide['INITIAL_IMPAIRMENT'], df_wide['CHANGE_OBSERVED'],
                                         fixed_slope, intercept)
        errors_intercept.append(distances.sum())
    best_intercept = candidate_intercepts[np.argmin(errors_intercept)]
    
    # ---- Optimize slope with the best intercept fixed ----
    candidate_slopes = np.linspace(0.1, 2.0, 200)
    errors_slope = []
    for slope in candidate_slopes:
        distances = euclidean_distance(df_wide['INITIAL_IMPAIRMENT'], df_wide['CHANGE_OBSERVED'],
                                         slope, best_intercept)
        errors_slope.append(distances.sum())
    best_slope = candidate_slopes[np.argmin(errors_slope)]
    
    # Save the fit parameters for both fits (PRR and Best fit)
    fit_parameters_list.append({
        "assessment": assessment,
        "fit": "PRR",
        "slope": fixed_slope,
        "intercept": best_intercept
    })
    fit_parameters_list.append({
        "assessment": assessment,
        "fit": "Best fit",
        "slope": best_slope,
        "intercept": best_intercept
    })
    
    # Scatter plot of the observed data
    ax.scatter(df_wide['INITIAL_IMPAIRMENT'], df_wide['CHANGE_OBSERVED'],
               color='blue', alpha=0.6, s=15)
    
    # Generate x values for the line plots
    x_vals = np.linspace(df_wide['INITIAL_IMPAIRMENT'].min(), df_wide['INITIAL_IMPAIRMENT'].max(), 100)
    
    # PRR line: fixed slope (0.7) and optimized intercept
    y_prr = fixed_slope * x_vals + best_intercept
    ax.plot(x_vals, y_prr, linestyle='--', color='black', label="PRR (dashed)")
    
    # Best fit line: optimized slope and same intercept
    y_best = best_slope * x_vals + best_intercept
    ax.plot(x_vals, y_best, linestyle='-', color='black', label="Best fit (solid)")
    
    # --- Identify outliers based on Euclidean distance to the PRR line ---
    # (using the fixed slope and best intercept)
    distances = euclidean_distance(df_wide['INITIAL_IMPAIRMENT'], df_wide['CHANGE_OBSERVED'],
                                   fixed_slope, best_intercept)
    Q1 = np.percentile(distances, 25)
    Q3 = np.percentile(distances, 75)
    IQR = Q3 - Q1
    threshold = Q3 + 1.5 * IQR
    outlier_mask = distances > threshold
    # Record outlier information
    for rec_id in df_wide.loc[outlier_mask, 'record_id']:
        outliers_list.append({
            "assessment": assessment,
            "record_id": rec_id
        })
    # Mark outliers with red edge circles (no fill)
    ax.scatter(df_wide.loc[outlier_mask, 'INITIAL_IMPAIRMENT'],
               df_wide.loc[outlier_mask, 'CHANGE_OBSERVED'],
               color='red', s=40, marker="x", alpha=0.5)
    
    # Set title for subplot
    ax.set_title(assessment_map[assessment], fontsize=12)
    if i == 0:
        ax.set_ylabel('Change observed', fontsize=12)
    
    
    # Remove top and right spines
    sns.despine(ax=ax, top=True, right=True)
    
    # Harmonize tick numbers if the assessment is not MRS
    if assessment != 'MRS':
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))
    

plt.tight_layout(rect=[0, 0, 1, 1])
# Adjust layout to provide extra space at the bottom for the tables
plt.subplots_adjust(bottom=0.25)

plt.subplots_adjust(wspace=0.37) 

# Add a general x label for all subplots
fig.supxlabel('Initial impairment', fontsize=12)

fig_path = os.path.join(pppath, "all_assessments_prr_vs_best_fit.svg")
plt.savefig(fig_path)
plt.show()

# Save the fit parameters table with both fits (PRR and Best fit)
fit_parameters_df = pd.DataFrame(fit_parameters_list)
fit_parameters_csv_path = os.path.join(parent_dir, 'output', "figures", "fit_parameters.csv")
fit_parameters_df.to_csv(fit_parameters_csv_path, index=False)

# Save the outliers table
outliers_df = pd.DataFrame(outliers_list)
outliers_csv_path = os.path.join(parent_dir, 'output', "figures", "outliers.csv")
outliers_df.to_csv(outliers_csv_path, index=False)
