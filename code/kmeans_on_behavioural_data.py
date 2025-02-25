import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches  # for custom legend handles

###############################
# Setup directories
###############################
code_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(code_dir)
output_dir = os.path.join(parent_dir, "output")
figures_dir = os.path.join(output_dir, "figures")
os.makedirs(figures_dir, exist_ok=True)

csv_file = os.path.join(output_dir, 'behavioral_data_cleaned.csv')
df = pd.read_csv(csv_file)

# Define the required score locations
required_locations = ['uex', 'lex']

###############################
# Step 1: Fixed clustering (for dot colors and decision boundary)
# Use tp == 0 as the fixed reference
###############################
fixed_tp = 0

df_fixed = df[df['tp'] == fixed_tp].copy()
pivot_fixed = df_fixed.pivot(index='record_id', columns='position', values='score')
if not set(required_locations).issubset(pivot_fixed.columns):
    raise ValueError(f"Timepoint {fixed_tp} does not have all required locations: {required_locations}.")
pivot_fixed = pivot_fixed.dropna(subset=required_locations)
X_fixed = pivot_fixed[required_locations].values

# Run KMeans (k=2) on fixed_tp data
kmeans_fixed = KMeans(n_clusters=2, random_state=42)
fixed_labels = kmeans_fixed.fit_predict(X_fixed)
pivot_fixed['fixed_cluster'] = fixed_labels

# Determine which fixed cluster has the higher combined (mean) score
fixed_cluster_means = pivot_fixed.groupby('fixed_cluster')[['lex', 'uex']].mean()
fixed_cluster_means['combined'] = fixed_cluster_means.mean(axis=1)
good_fixed_cluster = fixed_cluster_means['combined'].idxmax()

# Label the fixed clusters: “good” if it equals the good_fixed_cluster, else “bad”
pivot_fixed['fixed_type'] = pivot_fixed['fixed_cluster'].apply(
    lambda x: 'good' if x == good_fixed_cluster else 'bad'
)
# Save the fixed assignment for later merging (dots always use these colors)
fixed_df = pivot_fixed[['fixed_type']].copy()

# Compute the fixed decision boundary (perpendicular bisector of the two centers)
center0, center1 = kmeans_fixed.cluster_centers_
midpoint = (center0 + center1) / 2
normal_vector = center1 - center0  # vector perpendicular to the boundary (up to a sign)
n_norm = np.linalg.norm(normal_vector)

# Compute a mesh grid for the fixed boundary (for plotting the gray dashed line)
xx_fixed, yy_fixed = np.meshgrid(np.linspace(0, 150, 200),
                                 np.linspace(0, 150, 200))
grid_points_fixed = np.c_[xx_fixed.ravel(), yy_fixed.ravel()]
Z_fixed = kmeans_fixed.predict(grid_points_fixed)
Z_fixed = Z_fixed.reshape(xx_fixed.shape)
# Map predictions: 1 if equals the good_fixed_cluster, else 0
Z_fixed_mapped = np.where(Z_fixed == good_fixed_cluster, 1, 0)

###############################
# Step 2: For each timepoint, plot using fixed clustering and compute signed distances
###############################
timepoints = sorted(df['tp'].unique())
cm = 1 / 2.52  # scaling factor for figure size

# Define the palette: green for "good" and red for "bad"
palette_new = {'good': 'green', 'bad': 'red'}

# List to store computed distances for each subject at each timepoint
all_distances = []

for tp in timepoints:
    # Subset and pivot the current timepoint's data
    df_tp = df[df['tp'] == tp].copy()
    pivot_tp = df_tp.pivot(index='record_id', columns='position', values='score')
    if not set(required_locations).issubset(pivot_tp.columns):
        print(f"Timepoint {tp} missing required locations; skipping.")
        continue
    pivot_tp = pivot_tp.dropna(subset=required_locations)
    pivot_tp['tp'] = tp  # add the timepoint info
    
    # Merge the fixed clustering assignment (for dot colors)
    merged_for_dots = pivot_tp.merge(fixed_df, left_index=True, right_index=True, how='left')
    
    # Calculate the raw perpendicular distance from each point P ([uex, lex]) to the fixed boundary:
    # raw_distance = ((P - midpoint) dot normal_vector) / ||normal_vector||
    points = merged_for_dots[required_locations].values
    raw_distance = (points - midpoint).dot(normal_vector) / n_norm

    # Define the sign based on proximity to (0,0): if a point is closer to (0,0)
    # than the midpoint is, assign a negative distance; otherwise, positive.
    point_norms = np.linalg.norm(points, axis=1)
    midpoint_norm = np.linalg.norm(midpoint)
    signed_distance = np.where(point_norms < midpoint_norm, -np.abs(raw_distance), np.abs(raw_distance))
    
    merged_for_dots['distance_to_boundary'] = signed_distance
    
    # Save the computed distances along with record_id and timepoint
    merged_for_dots['record_id'] = merged_for_dots.index
    all_distances.append(merged_for_dots.reset_index(drop=True))
    
    # Plotting
    plt.figure(figsize=(9 * cm, 9 * cm), dpi=300)
    
    # Plot the fixed decision boundary as a dashed gray line
    plt.contour(xx_fixed, yy_fixed, Z_fixed_mapped, levels=[0.5],
                colors='grey', linestyles='dashed', linewidths=1)
    
    # Plot the subject dots colored by the fixed clustering assignment
    sns.scatterplot(
        data=merged_for_dots,
        x=required_locations[0],
        y=required_locations[1],
        hue='fixed_type',
        palette=palette_new,
        s=30,
        edgecolor="black",
        alpha=0.8
    )
    
    plt.title(f'Timepoint {tp}\nDots: Fixed clustering from tp=={fixed_tp} (good=green, bad=red)', fontsize=5)
    plt.xlabel(f'{required_locations[0]} Score',fontsize=12)
    plt.ylabel(f'{required_locations[1]} Score',fontsize=12)
    plt.xlim(0, 150)
    plt.ylim(0, 150)
    
    # Add legend for the fixed clusters (dots)
    legend_fixed = plt.legend(title='Fixed Cluster', bbox_to_anchor=(1.05, 1), loc='upper left',fontsize=12)
    plt.gca().add_artist(legend_fixed)
    
    plt.tight_layout()
    
    # Save the figure in SVG and PNG formats in the figures folder
    plt.savefig(os.path.join(figures_dir, f'timepoint_{tp}_scatter.svg'))
    plt.savefig(os.path.join(figures_dir, f'timepoint_{tp}_scatter.png'))
    
    plt.show()

# Combine the distance measurements across all timepoints into a single DataFrame
distance_df = pd.concat(all_distances, ignore_index=True)

# Save the updated data (with signed distance) to a new CSV file
distance_csv_file = os.path.join(output_dir, 'behavioral_data_with_distance.csv')
distance_df.to_csv(distance_csv_file, index=False)
