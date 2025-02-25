import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import MultiComparison

# -----------------------------
# Setup directories and file paths
# -----------------------------
code_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(code_dir)
output_dir = os.path.join(parent_dir, "output")
figures_dir = os.path.join(output_dir, "figures")
os.makedirs(figures_dir, exist_ok=True)

# -----------------------------
# Read in the CSV file with distance data
# -----------------------------
csv_file = os.path.join(output_dir, 'behavioral_data_with_distance.csv')
df = pd.read_csv(csv_file)
df = df.dropna().reset_index(drop=True)

# Ensure variables have the correct type
df['tp'] = df['tp'].astype('category')
df['fixed_type'] = df['fixed_type'].astype('category')
df['record_id'] = df['record_id'].astype('string')

# Force an ordering on timepoints based on their sorted order
timepoints = sorted(df['tp'].unique())
df['tp'] = pd.Categorical(df['tp'], categories=timepoints, ordered=True)

# -----------------------------
# Filter: Only include subjects with complete data (i.e. available in all timepoints)
# -----------------------------
complete_subjects = df.groupby('record_id')['tp'].nunique()
complete_subjects = complete_subjects[complete_subjects == len(timepoints)].index
df = df[df['record_id'].isin(complete_subjects)]

# -----------------------------
# Plotting: Boxplots by Fixed_Type with Time on x-axis and connected subject trajectories (each with a unique color)
# -----------------------------
# Define color palettes for the boxplot background
box_palette = {'good': 'green', 'bad': 'red'}

measure = 'distance_to_boundary'
ylabel = 'Distance to Boundary'

# Get unique fixed_type groups (e.g., ['good', 'bad'])
fixed_types = list(df['fixed_type'].cat.categories)
n_fixed = len(fixed_types)

# Calculate overall min and max for the measure and extend by ±15%
overall_min = df[measure].min()
overall_max = df[measure].max()
overall_range = overall_max - overall_min
y_lim_lower = overall_min - 0.15 * overall_range
y_lim_upper = overall_max + 0.15 * overall_range

# Convert cm to inches (1 cm ≈ 0.3937 inches)
cm_to_inch = 0.3937
# Allocate 10 cm width per subplot and 18 cm height overall
fig_width = 6 * n_fixed * cm_to_inch  
fig_height = 10 * cm_to_inch  

fig, axes = plt.subplots(1, n_fixed, figsize=(fig_width, fig_height), sharey=True, dpi=300)
if n_fixed == 1:
    axes = [axes]

for ax, ft in zip(axes, fixed_types):
    # Filter data for the current fixed_type group
    df_ft = df[df['fixed_type'] == ft]
    
    # Draw the background boxplot using a common color for each group
    sns.boxplot(
        x='tp',
        y=measure,
        data=df_ft,
        color=box_palette[ft],
        ax=ax,
        showfliers=False,
        width=0.6
    )
    
    # Generate a unique color palette for subjects in this fixed_type group.
    unique_subjects = df_ft['record_id'].unique()
    subject_palette = sns.color_palette("husl", len(unique_subjects))
    subject_color_map = {subject: color for subject, color in zip(unique_subjects, subject_palette)}
    
    # For each subject, connect their points across timepoints with a line and marker dots.
    for subject, subject_df in df_ft.groupby('record_id'):
        # Since we have filtered for complete subjects, only those with all timepoints remain.
        subject_df = subject_df.sort_values('tp')
        x_vals = subject_df['tp'].cat.codes  # Convert categorical timepoints to numeric codes (0, 1, 2, ...)
        y_vals = subject_df[measure]
        ax.plot(x_vals, y_vals, marker='o', markersize=2, linewidth=1.2, linestyle='-',
                color=subject_color_map[subject], alpha=0.8)
    
    ax.set_title(f"{ft.capitalize()} group", fontsize=9)
    ax.set_xlabel("Timepoint", fontsize=12)
    ax.tick_params(labelsize=8)
    sns.despine(ax=ax, top=True, right=True)
    ax.set_ylim(y_lim_lower, y_lim_upper)

# Label the y-axis on the leftmost subplot only
axes[0].set_ylabel(ylabel, fontsize=12)
for ax in axes[1:]:
    ax.set_ylabel("")

plt.tight_layout()

# Save the figure with dpi=300
fig_svg = os.path.join(figures_dir, 'distance_boxplots_by_fixedtype_only_in_all_timepoints.svg')
fig_png = os.path.join(figures_dir, 'distance_boxplots_by_fixedtype_only_in_all_timepoints.png')
plt.savefig(fig_svg, dpi=300)
plt.savefig(fig_png, dpi=300)
plt.show()

# -----------------------------
# Mixed-Model Analysis
# -----------------------------
model = smf.mixedlm("distance_to_boundary ~ tp + tp:fixed_type", data=df, groups=df["record_id"])
try:
    mixed_model = model.fit()
except Exception as e:
    print(f"Mixed model failed: {e}")
    raise e

model_summary = mixed_model.summary()
analysis_filename = 'mixed_model_analysis.txt'
analysis_output_path = os.path.join(output_dir, analysis_filename)
with open(analysis_output_path, 'w') as f:
    f.write(str(model_summary))
print(f'Mixed model analysis saved at: {analysis_output_path}')

# -----------------------------
# Posthoc Comparisons using MultiComparison (Tukey HSD)
# -----------------------------
df['Group_Timepoint'] = df['fixed_type'].astype(str) + '_' + df['tp'].astype(str)
mc = MultiComparison(df['distance_to_boundary'], df['Group_Timepoint'])
posthoc_result = mc.tukeyhsd()
posthoc_filename = 'posthoc_comparisons.csv'
posthoc_output_path = os.path.join(output_dir, posthoc_filename)
summary = posthoc_result.summary()
df_posthoc = pd.DataFrame(summary.data[1:], columns=summary.data[0])
df_posthoc.to_csv(posthoc_output_path, index=False)
print(f'Posthoc comparisons saved at: {posthoc_output_path}')
