import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Setup directories
#not that important script, didnt saw the effect I was hoping for. 
# -----------------------------
code_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(code_dir)
output_dir = os.path.join(parent_dir, "output")
figures_dir = os.path.join(output_dir, "figures")
os.makedirs(figures_dir, exist_ok=True)

# -----------------------------
# Read in the CSV with recovery types
# -----------------------------
csv_file = os.path.join(output_dir, "behavioral_data_with_distance_with_recovery_types.csv")
df = pd.read_csv(csv_file)

# -----------------------------
# Define visual settings
# -----------------------------
# Recovery types and fixed palette from Seaborn's Set1 palette
recovery_types = [
    "Steady recovery",
    "Steady decline",
    "Early recovery with chronic decline",
    "Late recovery with acute decline"
]
palette = dict(zip(recovery_types, sns.color_palette("Set1", n_colors=4)))

# -----------------------------
# Separate data by timepoint and create square subplots
# -----------------------------
timepoints = sorted(df['tp'].unique())
n_tp = len(timepoints)

# Create one row of subplots with shared axes and square aspect ratio.
fig, axes = plt.subplots(1, n_tp, figsize=(9, 9), dpi=300, sharex=True, sharey=True)
if n_tp == 1:
    axes = [axes]

for ax, tp in zip(axes, timepoints):
    df_tp = df[df['tp'] == tp]
    
    # Create scatter plot without legend by passing legend=False
    sns.scatterplot(
        data=df_tp,
        x="uex",
        y="lex",
        hue="recovery_type",
        palette=palette,
        s=50,
        edgecolor="black",
        alpha=0.8,
        legend=False,
        ax=ax
    )
    
    ax.set_title(f"Timepoint {tp}", fontsize=12)
    ax.set_xlabel("UEX Score", fontsize=12)
    ax.set_ylabel("LEX Score", fontsize=12)
    ax.set_xlim(0, 150)
    ax.set_ylim(0, 150)
    ax.tick_params(labelsize=12)
    # Force each subplot to be square
    ax.set_aspect('equal', adjustable='box')

plt.tight_layout()

# -----------------------------
# Save the figure
# -----------------------------
png_file = os.path.join(figures_dir, "recovery_types_scatter_by_timepoint_no_legend.png")
svg_file = os.path.join(figures_dir, "recovery_types_scatter_by_timepoint_no_legend.svg")
plt.savefig(png_file, dpi=300, bbox_inches="tight")
plt.savefig(svg_file, dpi=300, bbox_inches="tight")
plt.show()
