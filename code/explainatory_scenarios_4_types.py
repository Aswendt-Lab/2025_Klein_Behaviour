import os
import matplotlib.pyplot as plt
import seaborn as sns

# Determine the directory where this script is located, then go one level up for output
code_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(code_dir)
output_dir = os.path.join(parent_dir, "output")
figures_dir = os.path.join(output_dir, "figures")
os.makedirs(figures_dir, exist_ok=True)

# Define example data for each recovery scenario
recovery_data = {
    "Steady recovery": [10, 20, 30],                # Increasing: 10 -> 20 -> 30
    "Steady decline": [30, 20, 10],                 # Decreasing: 30 -> 20 -> 10
    "Early recovery with chronic decline": [10, 25, 15],  # Increase then decrease
    "Late recovery with acute decline": [30, 15, 25]      # Decrease then increase
}

# Define line styles for each recovery type
line_style_map = {
    "Steady recovery": "-",
    "Steady decline": "--",
    "Early recovery with chronic decline": ":",
    "Late recovery with acute decline": "-."
}

# Use a seaborn palette ("Set1") to assign fixed colors to each recovery type
recovery_types = list(recovery_data.keys())
colors = sns.color_palette("Set1", n_colors=4)
recovery_color_map = dict(zip(recovery_types, colors))

# Define x-axis values (we will drop the labels)
x_vals = [0, 1, 2]

# Create a 2x2 grid of subplots with size in inches (converted from cm)
fig, axes = plt.subplots(2, 2, figsize=(5/2.53, 5/2.53), dpi=300)
axes = axes.flatten()

# Loop over each recovery type and plot its example trajectory
for ax, rtype in zip(axes, recovery_types):
    y_vals = recovery_data[rtype]
    ax.plot(x_vals, y_vals, marker='o', markersize=2, linewidth=1,
            linestyle=line_style_map[rtype], color=recovery_color_map[rtype])
    #ax.set_title(rtype, fontsize=10)
    # Remove axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    # Remove grid lines
    ax.grid(False)

plt.tight_layout()

# Save the figure in the output/figures folder with the given name
png_file = os.path.join(figures_dir, "explainatory_scenarios_4_types.png")
svg_file = os.path.join(figures_dir, "explainatory_scenarios_4_types.svg")
plt.savefig(png_file, dpi=300, bbox_inches="tight")
plt.savefig(svg_file, dpi=300, bbox_inches="tight")
plt.show()
