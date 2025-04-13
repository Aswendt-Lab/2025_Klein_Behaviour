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

csv_file = os.path.join(output_dir, 'behavioral_data_cleaned_FM.csv')
df = pd.read_csv(csv_file)

# Group the DataFrame by 'record_id' and 'position'
groups = df.groupby(['record_id', 'position'])

# Filter groups with exactly 3 rows
df_valid = groups.filter(lambda x: len(x) == 3)

# Filter groups with not exactly 3 rows
df_removed = groups.filter(lambda x: len(x) != 3)

# Save the removed groups to a temporary CSV file
temp_csv_path = os.path.join(output_dir, "removed_groups_temp.csv")
df_removed.to_csv(temp_csv_path, index=False)

# If you want to update df to be only the valid rows:
df = df_valid

# Define the required score locations (order is important for later calculations)
required_locations = ['uex', 'lex']

###############################
# Define maximum scores based on:
# "Hi, der Uex hat 126 Punkte und der Lex 86 Punkte im Gesamtscore"
###############################
max_uex_score = 126
max_lex_score = 86

# For tick labels, the max tick is the true max, but the axis limits are set 15% higher.
x_lim = max_lex_score * 1.15  # x-axis limit is 15% more than the max lex score09118294973.
y_lim = max_uex_score * 1.15  # y-axis limit is 15% more than the max uex score.

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

# Label the fixed clusters: “good” if it equals the good_fixed_cluster, else “poor”
pivot_fixed['fixed_type'] = pivot_fixed['fixed_cluster'].apply(
    lambda x: 'good' if x == good_fixed_cluster else 'poor'
)
# Save the fixed assignment for later merging (dots always use these colors)
fixed_df = pivot_fixed[['fixed_type']].copy()

# Compute the fixed decision boundary (perpendicular bisector of the two centers)
center0, center1 = kmeans_fixed.cluster_centers_
midpoint = (center0 + center1) / 2
normal_vector = center1 - center0  # vector perpendicular to the boundary (up to a sign)
n_norm = np.linalg.norm(normal_vector)

# Compute a mesh grid for the fixed boundary (for plotting the gray dashed line)
# For the model, the order is [uex, lex], so we create a grid with:
#   x corresponding to lex and y corresponding to uex, then swap.
xx_fixed, yy_fixed = np.meshgrid(np.linspace(0, x_lim, 200),
                                 np.linspace(0, y_lim, 200))
# Supply points as [uex, lex]
grid_points_fixed = np.c_[yy_fixed.ravel(), xx_fixed.ravel()]
Z_fixed = kmeans_fixed.predict(grid_points_fixed)
Z_fixed = Z_fixed.reshape(xx_fixed.shape)
# Map predictions: 1 if equals the good_fixed_cluster, else 0
Z_fixed_mapped = np.where(Z_fixed == good_fixed_cluster, 1, 0)

###############################
# Step 2: For each timepoint, plot using fixed clustering and compute signed distances
###############################
timepoints = sorted(df['tp'].unique())
cm = 1 / 2.52  # scaling factor for figure size

# Define the palette: green for "good" and red for "poor"
palette_new = {'good': '#E69F00', 'poor': '#CC79A7'}
# List to store computed distances for each subject at each timepoint
all_distances = []

# Define custom ticks (same for all subplots)
num_ticks = 5
x_ticks = np.linspace(0, max_lex_score, num=num_ticks)
x_tick_labels = [f"{int(tick)}" for tick in x_ticks]
x_tick_labels[-1] = f"     {int(x_ticks[-1])}$_{{max}}$"

y_ticks = np.linspace(0, max_uex_score, num=num_ticks)
y_tick_labels = [f"{int(tick)}" for tick in y_ticks]
y_tick_labels[-1] = f"{int(y_ticks[-1])}$_{{max}}$"

# Create a single figure with 1 row and 3 columns of subplots and shared y-axis
fig, axes = plt.subplots(1, len(timepoints), sharey=True, figsize=(18 * cm , 6 * cm), dpi=300)

for ax, tp in zip(axes, timepoints):
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
    points = merged_for_dots[required_locations].values
    raw_distance = (points - midpoint).dot(normal_vector) / n_norm

    # Define the sign based on proximity to (0,0): if a point is closer to (0,0)
    # than the midpoint is, assign a negative distance; otherwise, positive.
    point_norms = np.linalg.norm(points, axis=1)
    midpoint_norm = np.linalg.norm(midpoint)
    signed_distance = np.where(point_norms < midpoint_norm, -np.abs(raw_distance), np.abs(raw_distance))
    
    merged_for_dots['distance_to_boundary'] = signed_distance
    merged_for_dots['record_id'] = merged_for_dots.index
    all_distances.append(merged_for_dots.reset_index(drop=True))
    
    # Plot the fixed decision boundary as a dashed gray line on the current axis
    ax.contour(xx_fixed, yy_fixed, Z_fixed_mapped, levels=[0.5],
               colors='grey', linestyles='dashed', linewidths=1)
    
    # Plot the subject dots colored by the fixed clustering assignment (swap axes: x=lex, y=uex)
    ax = sns.scatterplot(
        data=merged_for_dots,
        x='lex',
        y='uex',
        hue='fixed_type',
        palette=palette_new,
        s=15,
        edgecolor="black",
        alpha=1,
        ax=ax
    )
    
    sns.despine(ax=ax, top=True, right=True)
    # Set new axis labels: x-axis becomes "FM-LE" and y-axis "FM-UE"
    ax.set_xlabel('FM-LE', fontsize=12)
    if ax == axes[0]:
        ax.set_ylabel('FM-UE', fontsize=12)
    else:
        ax.set_ylabel('')
        
        
    ax.set_xlim(0, x_lim)
    ax.set_ylim(0, y_lim)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels)
    ax.legend().set_visible(False)

plt.tight_layout()
# Save the combined subplots figure
plt.savefig(os.path.join(figures_dir, 'timepoints_scatter.svg'))
plt.savefig(os.path.join(figures_dir, 'timepoints_scatter.png'))
plt.show()

# Combine the distance measurements across all timepoints into a single DataFrame
distance_df = pd.concat(all_distances, ignore_index=True)

# Save the updated data (with signed distance) to a new CSV file
distance_csv_file = os.path.join(output_dir, 'behavioral_data_with_distance_FM.csv')
distance_df.to_csv(distance_csv_file, index=False)
