import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Determine the directory where this script is located, then go one level up for output
code_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(code_dir)
output_dir = os.path.join(parent_dir, "output")
figures_dir = os.path.join(output_dir, "figures")
os.makedirs(figures_dir, exist_ok=True)

# Define example data for each recovery scenario
recovery_data = {
    "Steady recovery": [10, 20, 30],                # Increasing: 10 -> 20 -> 30
    "Steady decline": [30, 20, 10],                  # Decreasing: 30 -> 20 -> 10
    "Early recovery with chronic decline": [10, 25, 15],  # Increase then decrease
    "Late recovery with acute decline": [30, 15, 25]       # Decrease then increase
}

# Define line styles for each recovery type
line_style_map = {
    "Steady recovery": "-",
    "Steady decline": "--",
    "Early recovery with chronic decline": ":",
    "Late recovery with acute decline": "-."
}

# Define a color mapping using colors from the Okabe-Ito palette:
recovery_color_map = {
    "Steady recovery": "#009E73",                  # Bluish green for growth
    "Steady decline": "#D55E00",                   # Vermilion red for decline
    "Early recovery with chronic decline": "#E69F00", # Orange for early improvement but caution
    "Late recovery with acute decline": "#0072B2"    # Deep blue for late recovery after an acute drop
}

# Define x-axis values (we will drop the labels)
x_vals = [0, 1, 2]

# Create a 2x2 grid of subplots with size in inches (converted from cm)
fig, axes = plt.subplots(2, 2, figsize=(5/2.53, 5/2.53), dpi=300)
axes = axes.flatten()

# Loop over each recovery type and plot its example trajectory
for ax, rtype in zip(axes, recovery_data.keys()):
    y_vals = recovery_data[rtype]
    ax.plot(x_vals, y_vals, marker='o', markersize=2, linewidth=1,
            linestyle=line_style_map[rtype], color=recovery_color_map[rtype])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(False)

# Create custom legend handles
legend_elements = [
    Line2D([0], [0], color=recovery_color_map[rtype], lw=2, linestyle=line_style_map[rtype],
           marker='o', markersize=5, label=rtype)
    for rtype in recovery_data.keys()
]

# Adjust layout to leave space for the legend on the right
fig.subplots_adjust(right=0.75)  # Shrink plot area to make space on the right

# Calculate the width fraction for 18 cm in the legend.
# The figure width in inches is (5/2.53). Convert 18 cm to inches: 18/2.54.
legend_width_in = 18 / 2.54
# Now compute the fraction relative to the figure width.
legend_width_fraction = legend_width_in / (5/2.53)  # Approximately 3.59

# Place legend outside the figure to the right.
# Here we specify bbox_to_anchor as (x0, y0, width, height) in the normalized coordinate system.
fig.legend(handles=legend_elements, loc='lower left',
           bbox_to_anchor=(0.78, 0.5, legend_width_fraction, 0.2),
           ncol=4, prop={'size': 10, 'family': 'Calibri'}, frameon=False)

# Save the figure
png_file = os.path.join(figures_dir, "explanatory_scenarios_4_types.png")
svg_file = os.path.join(figures_dir, "explanatory_scenarios_4_types.svg")
plt.savefig(png_file, dpi=300, bbox_inches="tight")
plt.savefig(svg_file, dpi=300, bbox_inches="tight")
plt.show()
