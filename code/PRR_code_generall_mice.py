import os
import pandas as pd
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
# === File paths ===
code_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(code_dir)
input_file_path = os.path.join(parent_dir, 'input', 'Mice_data.csv')
output_dir = os.path.join(parent_dir, 'output')
figure_dir = os.path.join(parent_dir, 'output', "figures", "prr_mice")
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
# (Using 0, 3, and 28 as specified in the first part of your instructions.)
desired_timepoints = [0, 3, 28]
df_clean = df_clean[df_clean['TimePointMerged'].isin(desired_timepoints)]

#% === Step 4: Convert measurement columns to numeric and drop rows with NaNs or non-numeric entries ===
measurement_cols = ['C_PawDragPercent', 'GW_FootFault', 'RB_HindlimbDrop']
df_clean[measurement_cols] = df_clean[measurement_cols].apply(pd.to_numeric, errors='coerce')
df_clean = df_clean.dropna(subset=measurement_cols)

# === Step 5: Keep only StudyIDs that have exactly the required timepoints ===
required_timepoints = set(desired_timepoints)
# Get the set of timepoints available for each StudyID
study_timepoints = df_clean.groupby('StudyID')['TimePointMerged'].apply(set)

# Identify StudyIDs to keep and those to discard
studyids_to_keep = study_timepoints[study_timepoints == required_timepoints].index
studyids_to_discard = study_timepoints[study_timepoints != required_timepoints].index

# Save the discarded StudyIDs to a CSV
discarded_df = pd.DataFrame({'StudyID': list(studyids_to_discard)})
#discarded_csv_path = os.path.join(output_dir, 'discarded_subjects.csv')
#discarded_df.to_csv(discarded_csv_path, index=False)

# Filter the DataFrame to only include StudyIDs with all required timepoints
df_final = df_clean[df_clean['StudyID'].isin(studyids_to_keep)].copy()

#%
# List of measurement columns (assessments) to analyze.
assessment_types = ['C_PawDragPercent', 'GW_FootFault', 'RB_HindlimbDrop']
assessment_mapping = {
    'C_PawDragPercent': 'Paw Drags',
    'GW_FootFault': 'Foot Faults',
    'RB_HindlimbDrop': 'Hindlimb Drops'
}
# Function to calculate Euclidean distance from a point (x, y) to a line (slope, intercept)
def euclidean_distance(x, y, slope, intercept):
    return np.abs(slope * x - y + intercept) / np.sqrt(slope**2 + 1)

# Set up the subplot grid: 1 row x 5 columns.
fig, axs = plt.subplots(1, 3, figsize=(18/2.54,5/2.54), dpi=300)
axs = axs.flatten()  # flatten for easier indexing

# Lists to collect fit parameters and outlier information (for later use)
fit_parameters_list = []
outliers_list = []

# Loop over each assessment type (measurement)
for i, assessment in enumerate(assessment_types):
    # Check that we have an available axis (if there are fewer assessments than subplots)
    if i >= len(axs):
        break
    ax = axs[i]
    
    # Extract data for the current assessment by selecting StudyID, TimePointMerged, and the measurement column.
    df_assess = df_final[['StudyID', 'TimePointMerged', assessment]].copy()
    
    # Rename the measurement column to 'value' to ease pivoting.
    df_assess = df_assess.rename(columns={assessment: 'value'})
    
    # Pivot the data to a wide format: rows are StudyIDs and columns are the timepoints (0, 3, 28).
    df_wide = df_assess.pivot(index='StudyID', columns='TimePointMerged', values='value')
    
    # Drop rows with any missing timepoints
    df_wide = df_wide.dropna(subset=[0, 3, 28])
    
    # Rename columns to include the assessment name (e.g., "C_PawDragPercent_tp0")
    df_wide.columns = [f"{assessment}_tp{int(col)}" for col in df_wide.columns]
    df_wide.reset_index(inplace=True)
    
    # Compute derived columns:
    # INITIAL_IMPAIRMENT = value at timepoint 0 minus value at timepoint 3.
    # CHANGE_OBSERVED    = value at timepoint 28 minus value at timepoint 3.
    df_wide['INITIAL_IMPAIRMENT'] = -df_wide[f"{assessment}_tp0"] + df_wide[f"{assessment}_tp3"]
    df_wide['CHANGE_OBSERVED'] = -df_wide[f"{assessment}_tp28"] + df_wide[f"{assessment}_tp3"]
    
    # ---- Optimize intercept with fixed slope of 0.7 (PRR) ----
    fixed_slope = 0.7
    candidate_intercepts = np.linspace(-100, 100, 200)
    errors_intercept = []
    for intercept in candidate_intercepts:
        distances = euclidean_distance(df_wide['INITIAL_IMPAIRMENT'], 
                                       df_wide['CHANGE_OBSERVED'],
                                       fixed_slope, intercept)
        errors_intercept.append(distances.sum())
    best_intercept = candidate_intercepts[np.argmin(errors_intercept)]
    
    # ---- Optimize slope with the best intercept fixed ----
    candidate_slopes = np.linspace(0.1, 2.0, 200)
    errors_slope = []
    for slope in candidate_slopes:
        distances = euclidean_distance(df_wide['INITIAL_IMPAIRMENT'], 
                                       df_wide['CHANGE_OBSERVED'],
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
    distances = euclidean_distance(df_wide['INITIAL_IMPAIRMENT'], df_wide['CHANGE_OBSERVED'],
                                   fixed_slope, best_intercept)
    Q1 = np.percentile(distances, 25)
    Q3 = np.percentile(distances, 75)
    IQR = Q3 - Q1
    threshold = Q3 + 1.5 * IQR
    outlier_mask = distances > threshold
    
    # Record outlier information using 'StudyID' as the identifier
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
plt.subplots_adjust(bottom=0.3)

# Add a general x label for all subplots
fig.supxlabel('Initial impairment', fontsize=12)

# Save and show the figure
fig_path = os.path.join(figure_dir, "all_assessments_prr_vs_best_fit_mice.svg")
plt.savefig(fig_path, dpi=300)
plt.show()

# Save the fit parameters to CSV
fit_parameters_df = pd.DataFrame(fit_parameters_list)
fit_parameters_csv_path = os.path.join(parent_dir, 'output', "fit_parameters_mice.csv")
fit_parameters_df.to_csv(fit_parameters_csv_path, index=False)

# Save the outliers information to CSV
outliers_df = pd.DataFrame(outliers_list)
outliers_csv_path = os.path.join(parent_dir, 'output', "outlier_mice.csv")
outliers_df.to_csv(outliers_csv_path, index=False)
