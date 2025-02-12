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
# Step 1: Fixed clustering (for dot colors)
# Use tp == 0 as the fixed reference (i.e. "first cluster")
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

# Compute the fixed mesh (decision boundary) for tp==0 using a fixed domain (0 to 150)
xx_fixed, yy_fixed = np.meshgrid(np.linspace(0, 150, 200),
                                 np.linspace(0, 150, 200))
grid_points_fixed = np.c_[xx_fixed.ravel(), yy_fixed.ravel()]
Z_fixed = kmeans_fixed.predict(grid_points_fixed)
Z_fixed = Z_fixed.reshape(xx_fixed.shape)
# Map predictions: 1 if equals the good_fixed_cluster, else 0
Z_fixed_mapped = np.where(Z_fixed == good_fixed_cluster, 1, 0)

###############################
# Step 2: For each timepoint, run new clustering (for mesh) 
# and plot using:
#   - Background mesh: computed from new clustering on that timepoint (curved using bilinear interpolation)
#   - Dots: colored by the fixed clustering from tp==0
###############################
timepoints = sorted(df['tp'].unique())
cm = 1 / 2.52  # scaling factor for figure size

# Define the palette: use green for "good" and red for "bad"
palette_new = {'good': 'green', 'bad': 'red'}
# For the mesh, set a colormap accordingly: 0 (bad) -> red, 1 (good) -> green
mesh_cmap = ListedColormap(['red', 'green'])

for tp in timepoints:
    # Subset and pivot the current timepoint's data
    df_tp = df[df['tp'] == tp].copy()
    pivot_tp = df_tp.pivot(index='record_id', columns='position', values='score')
    if not set(required_locations).issubset(pivot_tp.columns):
        print(f"Timepoint {tp} missing required locations; skipping.")
        continue
    pivot_tp = pivot_tp.dropna(subset=required_locations)
    X_current = pivot_tp[required_locations].values

    # Run new KMeans clustering on current timepoint data
    new_kmeans = KMeans(n_clusters=2, random_state=42)
    new_labels = new_kmeans.fit_predict(X_current)
    pivot_tp['new_cluster'] = new_labels

    # Determine which new cluster has the higher combined score
    new_cluster_means = pivot_tp.groupby('new_cluster')[required_locations].mean()
    new_cluster_means['combined'] = new_cluster_means.mean(axis=1)
    good_new_cluster = new_cluster_means['combined'].idxmax()
    pivot_tp['new_type'] = pivot_tp['new_cluster'].apply(lambda x: 'good' if x == good_new_cluster else 'bad')
    
    # Compute mesh boundaries using a fixed domain: 0 to 150
    x_min_new, x_max_new = 0, 150
    y_min_new, y_max_new = 0, 150

    xx_new, yy_new = np.meshgrid(np.linspace(x_min_new, x_max_new, 200),
                                 np.linspace(y_min_new, y_max_new, 200))
    grid_points_new = np.c_[xx_new.ravel(), yy_new.ravel()]
    Z_mesh_new = new_kmeans.predict(grid_points_new)
    Z_mesh_new = Z_mesh_new.reshape(xx_new.shape)
    # Map the new predictions to 0/1: 1 if equals the good_new_cluster, else 0
    Z_mapped_new = np.where(Z_mesh_new == good_new_cluster, 1, 0)

    # Create a figure with dpi=300
    plt.figure(figsize=(12 * cm, 9 * cm), dpi=300)
    
    # Plot the dynamic background mesh using imshow with bilinear interpolation for a curved effect.
    plt.imshow(Z_mapped_new, extent=(x_min_new, x_max_new, y_min_new, y_max_new),
               origin='lower', cmap=mesh_cmap, alpha=0.1, interpolation='bilinear')
    
    # Overlay the fixed (tp==0) boundary as a dashed grey line
    plt.contour(xx_fixed, yy_fixed, Z_fixed_mapped, levels=[0.5],
                colors='grey', linestyles='dashed', linewidths=1)
    
    # For the dots, merge the current timepoint data with the fixed assignment (from tp==0)
    merged_for_dots = pivot_tp.merge(fixed_df, left_index=True, right_index=True, how='left')
    
    # Plot dots colored by the fixed clustering (so they remain constant over time)
    sns.scatterplot(
        data=merged_for_dots,
        x=required_locations[0],
        y=required_locations[1],
        hue='fixed_type',
        palette=palette_new,
        s=50,
        edgecolor="black",
        alpha=0.8
    )
    
    plt.title(f'Timepoint {tp}\nDots: Fixed clustering from tp=={fixed_tp} (good=green, bad=red)\n'
              f'Mesh: New clustering boundaries', fontsize=8)
    plt.xlabel(f'{required_locations[0]} Score', fontsize=8)
    plt.ylabel(f'{required_locations[1]} Score', fontsize=8)
    
    # Set fixed axis limits: 0 to 150 for both x and y
    plt.xlim(0, 150)
    plt.ylim(0, 150)
    
    # First, add the legend for the fixed clusters (dots)
    legend_fixed = plt.legend(title='Fixed Cluster', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.gca().add_artist(legend_fixed)
    
    # Now, add a separate legend for the dynamic mesh (dynamic cluster)
    dynamic_handles = [
        mpatches.Patch(color=palette_new['good'], alpha=0.1, label='Dynamic Cluster: Good'),
        mpatches.Patch(color=palette_new['bad'], alpha=0.1, label='Dynamic Cluster: Bad')
    ]
    plt.legend(handles=dynamic_handles, title='Dynamic Cluster', bbox_to_anchor=(1.05, 0.5), loc='center left', fontsize=8)
    
    plt.tight_layout()
    
    # Save the plot in SVG and PNG formats in the figures folder
    plt.savefig(os.path.join(figures_dir, f'timepoint_{tp}_scatter.svg'))
    plt.savefig(os.path.join(figures_dir, f'timepoint_{tp}_scatter.png'))
    
    plt.show()
