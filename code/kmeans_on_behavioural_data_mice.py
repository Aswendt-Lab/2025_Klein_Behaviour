import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns  # used for palette (if needed)

#######################################
# === Define Directories and File Paths ===
#######################################
code_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(code_dir)
input_file_path = os.path.join(parent_dir, 'input', 'Mice_data.csv')
output_dir = os.path.join(parent_dir, 'output')
figure_dir = os.path.join(parent_dir, 'output', "figures", "pythonFigs", "prr_mice")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(figure_dir, exist_ok=True)

#######################################
# === Read the CSV file ===
#######################################
df = pd.read_csv(input_file_path)

#######################################
# === Preprocessing: Clean and Subset the DataFrame ===
#######################################
cols_to_keep = ['StudyID', 'TimePointMerged', 'C_PawDragPercent', 'GW_FootFault', 'RB_HindlimbDrop', 'Group']
df_clean = df[cols_to_keep].copy()

# Filter rows for Group == "Stroke"
df_clean = df_clean[df_clean['Group'] == "Stroke"]
df_clean2 = df_clean[df_clean['Group'] == "Stroke"]


# Filter rows for the desired timepoints [3, 28]
desired_timepoints = [3, 28]
df_clean = df_clean[df_clean['TimePointMerged'].isin(desired_timepoints)]

# Convert measurement columns to numeric and drop rows with missing or non-numeric entries
measurement_cols = ['C_PawDragPercent', 'GW_FootFault', 'RB_HindlimbDrop']
df_clean[measurement_cols] = df_clean[measurement_cols].apply(pd.to_numeric, errors='coerce')
df_clean = df_clean.dropna(subset=measurement_cols)

# Keep only StudyIDs that have exactly the required timepoints
required_timepoints = set(desired_timepoints)
study_timepoints = df_clean.groupby('StudyID')['TimePointMerged'].apply(set)
studyids_to_keep = study_timepoints[study_timepoints == required_timepoints].index
# Optionally, save the discarded StudyIDs
studyids_to_discard = study_timepoints[study_timepoints != required_timepoints].index
discarded_df = pd.DataFrame({'StudyID': list(studyids_to_discard)})
# Uncomment to save discarded subjects:
# discarded_csv_path = os.path.join(output_dir, 'discarded_subjects.csv')
# discarded_df.to_csv(discarded_csv_path, index=False)

df_final = df_clean[df_clean['StudyID'].isin(studyids_to_keep)].copy()

desired_timepoints2 = [0,3, 28]
df_clean2 = df_clean2[df_clean2['TimePointMerged'].isin(desired_timepoints2)]
df_final2 = df_clean2[df_clean2['StudyID'].isin(studyids_to_keep)].copy()


#######################################
# === Setup for 3D Analysis and Plotting ===
#######################################
# List of assessment columns and friendly names mapping.
assessment_types = ['C_PawDragPercent', 'GW_FootFault', 'RB_HindlimbDrop']
assessment_mapping = {
    'C_PawDragPercent': 'Paw Drags',
    'GW_FootFault': 'Foot Faults',
    'RB_HindlimbDrop': 'Hindlimb Drops'
}

# Order of features corresponds to x, y, and z axes.
required_features = assessment_types

# Compute original axis limits (using maximum observed scores scaled by 15% margin)
max_scores = {feature: df_final[feature].max() for feature in required_features}
x_lim = max_scores[required_features[0]] * 1.15
y_lim = max_scores[required_features[1]] * 1.15
z_lim = max_scores[required_features[2]] * 1.15

#######################################
# === Standardize (normalize) values and Fixed Clustering Using Timepoint 3 in 3D ===
#######################################
fixed_tp = 3  # use timepoint 3 for clustering
df_fixed = df_final[df_final['TimePointMerged'] == fixed_tp].copy()
df_fixed.set_index('StudyID', inplace=True)

# Check that all required features exist at tp=3.
if not all(feature in df_fixed.columns for feature in required_features):
    raise ValueError(f"Timepoint {fixed_tp} does not have all required features: {required_features}.")

# Create a StandardScaler and fit it on the tp=3 data.
scaler = StandardScaler()
X_fixed = df_fixed[required_features].values
X_fixed_scaled = scaler.fit_transform(X_fixed)

# Run KMeans on the standardized data (2 clusters).
kmeans_fixed = KMeans(n_clusters=2, random_state=42)
fixed_labels = kmeans_fixed.fit_predict(X_fixed_scaled)
df_fixed['fixed_cluster'] = fixed_labels

# Determine the "good" cluster based on lower combined (mean) score (deficit values; lower is better).
cluster_means = df_fixed.groupby('fixed_cluster')[required_features].mean()
cluster_means['combined'] = cluster_means.mean(axis=1)
good_fixed_cluster = cluster_means['combined'].idxmin()  # lower is better

# Label clusters: "good" if it matches good_fixed_cluster, else "poor".
df_fixed['fixed_type'] = df_fixed['fixed_cluster'].apply(
    lambda x: 'good' if x == good_fixed_cluster else 'poor'
)
# Save the fixed assignment for merging.
fixed_df = df_fixed[['fixed_type']].copy()

# Compute the decision boundary plane in standardized space.
center0_scaled, center1_scaled = kmeans_fixed.cluster_centers_
midpoint_scaled = (center0_scaled + center1_scaled) / 2
normal_vector_scaled = center1_scaled - center0_scaled  # normal vector in standardized space
n_norm = np.linalg.norm(normal_vector_scaled)

# Create a meshgrid in standardized space.
# Determine grid limits by transforming the original plotting limits.
# For each feature, compute standardized min and max.
orig_limits = np.array([[0, x_lim],
                        [0, y_lim],
                        [0, z_lim]])
# scaler.mean_ and scaler.scale_ are arrays corresponding to the features (in order).
std_limits = np.empty_like(orig_limits, dtype=float)
for i in range(3):
    std_limits[i, 0] = (orig_limits[i, 0] - scaler.mean_[i]) / scaler.scale_[i]
    std_limits[i, 1] = (orig_limits[i, 1] - scaler.mean_[i]) / scaler.scale_[i]

# We choose the grid plane based on the standardized coordinate with the largest absolute normal component.
abs_norm = np.abs(normal_vector_scaled)
if abs_norm[2] >= abs_norm[0] and abs_norm[2] >= abs_norm[1]:
    # Use standardized x and y as the grid; solve for z.
    std_x = np.linspace(std_limits[0, 0], std_limits[0, 1], 20)
    std_y = np.linspace(std_limits[1, 0], std_limits[1, 1], 20)
    grid_x, grid_y = np.meshgrid(std_x, std_y)
    # Solve for z in standardized space using the plane equation:
    grid_z = (-normal_vector_scaled[0]*(grid_x - midpoint_scaled[0]) - 
              normal_vector_scaled[1]*(grid_y - midpoint_scaled[1])
             ) / normal_vector_scaled[2] + midpoint_scaled[2]
elif abs_norm[1] >= abs_norm[0]:
    # Use standardized x and z as grid; solve for y.
    std_x = np.linspace(std_limits[0, 0], std_limits[0, 1], 20)
    std_z = np.linspace(std_limits[2, 0], std_limits[2, 1], 20)
    grid_x, grid_z = np.meshgrid(std_x, std_z)
    grid_y = (-normal_vector_scaled[0]*(grid_x - midpoint_scaled[0]) - 
              normal_vector_scaled[2]*(grid_z - midpoint_scaled[2])
             ) / normal_vector_scaled[1] + midpoint_scaled[1]
else:
    # Use standardized y and z as grid; solve for x.
    std_y = np.linspace(std_limits[1, 0], std_limits[1, 1], 20)
    std_z = np.linspace(std_limits[2, 0], std_limits[2, 1], 20)
    grid_y, grid_z = np.meshgrid(std_y, std_z)
    grid_x = (-normal_vector_scaled[1]*(grid_y - midpoint_scaled[1]) - 
              normal_vector_scaled[2]*(grid_z - midpoint_scaled[2])
             ) / normal_vector_scaled[0] + midpoint_scaled[0]

# Transform the standardized grid back to the original scale for plotting.
grid_shape = grid_x.shape
grid_flat = np.c_[grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]
grid_orig = scaler.inverse_transform(grid_flat)
xx_orig = grid_orig[:, 0].reshape(grid_shape)
yy_orig = grid_orig[:, 1].reshape(grid_shape)
zz_orig = grid_orig[:, 2].reshape(grid_shape)

#######################################
# === Plotting and Distance Calculation for Each Timepoint ===
#######################################
timepoints = sorted(df_final['TimePointMerged'].unique())

# Define a palette for coloring ("good" in orange, "poor" in purple).
palette_new = {'good': '#E69F00', 'poor': '#CC79A7'}

# List to accumulate computed distances.
all_distances = []

# Set up a figure with one 3D subplot per timepoint.
cm_factor = 1 / 2.52  # scaling factor for figure size if needed
fig = plt.figure(figsize=(18 * cm_factor, 9 * cm_factor), dpi=300)
axes = []
for i in range(len(timepoints)):
    ax = fig.add_subplot(1, len(timepoints), i + 1, projection='3d')
    axes.append(ax)

for ax, tp in zip(axes, timepoints):
    # Subset current timepoint data and set index to StudyID.
    df_tp = df_final[df_final['TimePointMerged'] == tp].copy()
    df_tp.set_index('StudyID', inplace=True)
    if not all(feature in df_tp.columns for feature in required_features):
        print(f"Timepoint {tp} missing required features; skipping.")
        continue

    # Merge the fixed clustering assignment (from timepoint 3) with current data.
    merged = df_tp.merge(fixed_df, left_index=True, right_index=True, how='left')
    
    # Standardize the current data using the scaler fitted at tp=3.
    X_tp = merged[required_features].values
    X_tp_scaled = scaler.transform(X_tp)
    
    # Compute the signed perpendicular distance in standardized space.
    raw_distance = np.dot(X_tp_scaled - midpoint_scaled, normal_vector_scaled) / n_norm
    # Determine sign based on proximity to the origin in standardized space.
    point_norms = np.linalg.norm(X_tp_scaled, axis=1)
    midpoint_norm = np.linalg.norm(midpoint_scaled)
    signed_distance = np.where(point_norms < midpoint_norm, -np.abs(raw_distance), np.abs(raw_distance))
    merged['distance_to_boundary'] = signed_distance
    merged.reset_index(inplace=True)  # bring back 'StudyID'
    all_distances.append(merged)

    # Plot the decision boundary plane (in original units).
    ax.plot_surface(xx_orig, yy_orig, zz_orig, color='grey', alpha=0.3,
                    rstride=1, cstride=1, edgecolor='none')
    
    # Scatter plot: use original (unstandardized) values.
    ax.scatter(
        merged[required_features[0]],
        merged[required_features[1]],
        merged[required_features[2]],
        c=merged['fixed_type'].map(palette_new),
        s=15, edgecolor="black"
    )
    
    # Label axes using friendly names.
    ax.set_xlabel(assessment_mapping[required_features[0]])
    ax.set_ylabel(assessment_mapping[required_features[1]])
    ax.set_zlabel(assessment_mapping[required_features[2]])
    
    # Set axis limits using the original data scale.
    ax.set_xlim(0, x_lim)
    ax.set_ylim(0, y_lim)
    ax.set_zlim(0, z_lim)
    
    # Set subplot title.
    ax.set_title(f"Timepoint {tp}", fontsize=5)

plt.tight_layout(rect=[0, 0, 1, 1])
plt.subplots_adjust(bottom=0.05, wspace=0.17)

# Save the combined 3D scatter figure.
fig.savefig(os.path.join(figure_dir, 'timepoints_scatter_3D_normalized.svg'))
fig.savefig(os.path.join(figure_dir, 'timepoints_scatter_3D_normalized.png'))
plt.show()

# Combine distance measurements across all timepoints and save to CSV.
distance_df = pd.concat(all_distances, ignore_index=True)
distance_csv_file = os.path.join(output_dir, 'behavioral_data_with_distance_3D_normalized.csv')
distance_df.to_csv(distance_csv_file, index=False)

#######################################
# === Create Additional Outputs: CSV with Cluster Classification ===
#######################################
# Merge the fixed clustering assignment onto the final dataframe.
# This creates a CSV similar to the original 'Mice_data.csv' but with an additional column ("fixed_type")
# indicating the "good" or "poor" cluster assignment based on timepoint 3.
df_classified = df_final2.merge(fixed_df, left_on='StudyID', right_index=True, how='left')
classified_csv_file = os.path.join(output_dir, 'Mice_data_classified.csv')
df_classified.to_csv(classified_csv_file, index=False)

#######################################
# === Write a Text File Reporting Mouse Count and Cluster Breakdown ===
#######################################
# Use the fixed clustering DataFrame from timepoint 3 (each StudyID appears only once here).
cluster_counts = fixed_df['fixed_type'].value_counts()
total_mice = cluster_counts.sum()
good_count = cluster_counts.get('good', 0)
poor_count = cluster_counts.get('poor', 0)

# Create the text to save.
text_lines = [
    f"Total mice: {total_mice}",
    f"Good cluster: {good_count}",
    f"Poor cluster: {poor_count}"
]
txt_file = os.path.join(output_dir, "mice_count_cluster.txt")
with open(txt_file, "w") as f:
    f.write("\n".join(text_lines))
