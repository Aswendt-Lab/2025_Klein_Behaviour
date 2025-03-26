# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 14:35:49 2025

@author: arefk
"""

import os
import pandas as pd
from scipy.stats import linregress
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, r2_score  # Removed duplicate import
import matplotlib.pyplot as plt 
import seaborn as sns

# Function to calculate Euclidean distance from a point to a line
def euclidean_distance(x, y, slope, intercept):
    return np.abs(slope * x - y + intercept) / np.sqrt(slope**2 + 1)

# Get the directory where the code file is located
code_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory of the code directory
parent_dir = os.path.dirname(code_dir)
# Specify the input file path relative to the code file
input_file_path = os.path.join(parent_dir, 'output', 'behavioral_data_with_distance_with_recovery_types.csv')
final_csv = os.path.join(parent_dir, 'output', 'behavioral_data_with_distance_with_recovery_types_with_PRR_HRR.csv')
outfile_path = os.path.join(parent_dir, 'output')
pppath = os.path.join(outfile_path, "figures", "prr&hrr_Euclidian")
os.makedirs(pppath, exist_ok=True)

# Read the CSV file into a Pandas DataFrame
df = pd.read_csv(input_file_path)
#df = df[df["fixed_type"]=="bad"]
# Filter out subjects with less than 3 records and NaN values
subject_counts = df["record_id"].value_counts()
neglected_subjects = subject_counts[subject_counts < 3].index.tolist()
df_filtered = df[~df["record_id"].isin(neglected_subjects)]
df_filtered.reset_index(drop=True, inplace=True)


# Normalize each score to a 0-1 range and compute the average
df_filtered["MOTOR_SCORE"] = ((df_filtered["lex"] / 86) + (df_filtered["uex"] / 126)) / 2
#df_filtered["MOTOR_SCORE"] = df_filtered["lex"]


# Compute z-scores for 'lex' and 'uex'
#df_filtered["lex_z"] = (df_filtered["lex"] - df_filtered["lex"].mean()) / df_filtered["lex"].std()
#df_filtered["uex_z"] = (df_filtered["uex"] - df_filtered["uex"].mean()) / df_filtered["uex"].std()

# Average the z-scores to get MOTOR_SCORE
#df_filtered["MOTOR_SCORE"] = (df_filtered["lex_z"] + df_filtered["uex_z"]) / 2



# Plot motor score over time
motor_over_time = df_filtered.groupby('tp')['MOTOR_SCORE'].mean().reset_index()
plt.figure(figsize=(18/2.53, 10/2.53))
plt.plot(motor_over_time['tp'], motor_over_time['MOTOR_SCORE'], marker='o')
plt.xlabel('Time Point')
plt.ylabel('Average Motor Score')
plt.title('Motor Score Over Time')
plt.tight_layout()
plt.show()
#%
# Pivot the DataFrame for time-dependent variables only.
constant_cols = ['record_id', 'fixed_type', 'recovery_type']
df_constants = df_filtered[constant_cols].drop_duplicates()
time_dependent_cols = [col for col in df_filtered.columns if col not in constant_cols + ['tp']]
df_time = df_filtered.pivot(index='record_id', columns='tp', values=time_dependent_cols)
df_time.columns = [f"{col[0]}_tp{col[1]}" for col in df_time.columns]
df_time.reset_index(inplace=True)
df_wide = pd.merge(df_time, df_constants, on='record_id')

# Calculate INITIAL_IMPAIRMENT and CHANGE_OBSERVED
df_wide['INITIAL_IMPAIRMENT'] = 1 - df_wide['MOTOR_SCORE_tp0']
df_wide['CHANGE_OBSERVED'] =  -df_wide['MOTOR_SCORE_tp0'] +df_wide['MOTOR_SCORE_tp2']  
error = -0.8
df_wide['CHANGE_PREDICTED'] = 0.7 * df_wide['INITIAL_IMPAIRMENT'] + error
#plt.scatter(df_wide['INITIAL_IMPAIRMENT'], df_wide['CHANGE_OBSERVED'], marker='o')
#%
# Initialize 'LINE_CLUSTER' to 'HRR'
df_wide['LINE_CLUSTER'] = "HRR"


# Initial linear regression on the whole dataset (if needed)
x_initial = df_wide['INITIAL_IMPAIRMENT']
y_initial = df_wide['CHANGE_OBSERVED']
slope_initial, intercept_initial, r_value_initial, p_value_initial, std_err_initial = linregress(x_initial, y_initial)
regression_line_initial = slope_initial * x_initial + intercept_initial
residuals_initial = y_initial - regression_line_initial


#%
# Define number of iterations
R = 10 # You can adjust this value

# Loop for iterative refinement
for cc in range(R-1):
    # Perform linear regression on HRR points
    HRR_points = df_wide[df_wide["LINE_CLUSTER"] == "HRR"]
    x_HRR = HRR_points['INITIAL_IMPAIRMENT']
    y_HRR = HRR_points['CHANGE_OBSERVED']
    
    if len(x_HRR) < 2:
        print(f"Not enough points to perform linear regression for HRR at iteration {cc}.")
        slope, intercept = 0, 0
    else:
        slope, intercept, r_value, p_value, std_err = linregress(x_HRR, y_HRR)
    

    intercept = round(intercept, 2)
        
    print(intercept)
    # Use a consistent PRR slope of -0.7
    prr_slope =0.7
    
    # Calculate Euclidean distances for all points
    dist_to_regression = euclidean_distance(df_wide['INITIAL_IMPAIRMENT'], 
                                            df_wide['CHANGE_OBSERVED'], 
                                            slope, intercept)
    dist_to_PRR = euclidean_distance(df_wide['INITIAL_IMPAIRMENT'], 
                                     df_wide['CHANGE_OBSERVED'], 
                                     prr_slope, intercept)
    
    # Assign points to the closest line (HRR vs. PRR)
    df_wide['LINE_CLUSTER'] = np.where(dist_to_regression < dist_to_PRR, 'HRR', 'PRR')
    
    # In final iteration, calculate outliers using IQR on PRR residuals
    if cc == (R - 1):
        final_slope = prr_slope
        final_intercept = intercept  # already rounded
        prr_points = df_wide[df_wide["LINE_CLUSTER"] == "PRR"]
        
        if not prr_points.empty:
            prr_line_vals = final_slope * prr_points['INITIAL_IMPAIRMENT'] + final_intercept
            final_residuals = prr_points['CHANGE_OBSERVED'] - prr_line_vals
    
            Q1_final = np.percentile(final_residuals, 25)
            Q3_final = np.percentile(final_residuals, 75)
            IQR_final = Q3_final - Q1_final
            threshold_final = 1.5 * IQR_final
    
            # Identify outliers based on the IQR rule
            outliers = prr_points[
                (final_residuals < Q1_final - threshold_final) | 
                (final_residuals > Q3_final + threshold_final)
            ]
            # Update cluster label for outliers
            df_wide.loc[outliers.index, 'LINE_CLUSTER'] = "outlier"
        else:
            print("No PRR points available for outlier detection.")
            outliers = pd.DataFrame()
        
        current_intercept = final_intercept

    else:
        outliers = pd.DataFrame()  # no outliers in intermediate iterations
        current_intercept = intercept  # use current HRR regression intercept
    
    # Plotting
    cm = 1 / 2.54  # conversion factor for centimeters
    if cc == (R - 1):
        plt.figure(figsize=(9 * cm, 9 * cm), dpi=300)
        s = 50
        linewidth = 1
    else:
        plt.figure(figsize=(9 * cm, 9 * cm), dpi=300)
        s = 15
        linewidth = 0.5

    # Scatterplot with hue indicating the cluster; outliers are colored red
    sns.scatterplot(x='INITIAL_IMPAIRMENT', y='CHANGE_OBSERVED', hue='LINE_CLUSTER', data=df_wide,
                    palette={'HRR': "g", 'PRR': "b", 'outlier': "red"}, marker='o', s=25, alpha=0.7)
    
    # Define an x-range for plotting the lines
    x_range = np.linspace(df_wide['INITIAL_IMPAIRMENT'].min(), df_wide['INITIAL_IMPAIRMENT'].max(), 100)
    
    # Plot HRR regression line
    plt.plot(x_range, slope * x_range + intercept, color='black', linewidth=linewidth,
             label=f'Linear fit (HRR): y = {slope:.2f}x + {intercept:.2f}')
    
    # Plot PRR line using the consistent slope and current intercept
    plt.plot(x_range, prr_slope * x_range + current_intercept, color='black', linestyle="--", linewidth=linewidth,
             label=f'PRR Line: y = {prr_slope}x + {current_intercept}')
    
    # Set labels and styling
    plt.xlabel('INITIAL_IMPAIRMENT (Max - Acute)', fontsize=8)
    plt.ylabel('CHANGE_OBSERVED (Chronic - Acute)', fontsize=8)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    #plt.ylim([-2, 2])
    #plt.xlim([-0.5, 1])
    
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False, fontsize=10, markerscale=0.5)
    plt.tight_layout()  # Adjust layout to accommodate the legend outside
    plt.savefig(os.path.join(pppath, f"{cc}_ppr_rule.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(pppath, f"{cc}_ppr_rule.svg"), bbox_inches='tight')
    plt.show()
    
    
    # In final iteration, calculate goodness of fit and save outputs
    if cc == (R - 1):
        # HRR line goodness of fit
        HRR_data = df_wide[df_wide["LINE_CLUSTER"] == "HRR"]
        y_true_HRR = HRR_data['CHANGE_OBSERVED']
        x_HRR = HRR_data['INITIAL_IMPAIRMENT']
        y_pred_HRR = slope * x_HRR + intercept
        r_squared_HRR = r2_score(y_true_HRR, y_pred_HRR)
        
        # PRR line goodness of fit (excluding outliers)
        PRR_data = df_wide[df_wide["LINE_CLUSTER"] == "PRR"]
        y_true_PRR = PRR_data['CHANGE_OBSERVED']
        x_PRR = PRR_data['INITIAL_IMPAIRMENT']
        y_pred_PRR = final_slope * x_PRR + final_intercept
        r_squared_PRR = r2_score(y_true_PRR, y_pred_PRR)
        
        print(f"R-squared for HRR fit: {r_squared_HRR:.4f}")
        print(f"R-squared for PRR fit: {r_squared_PRR:.4f}")
        
        # Save the final DataFrame to CSV
        df_wide.to_csv(os.path.join(outfile_path, 'final_plot_ppr_HRR_changeObserved_initialImpairment.csv'), index=False)
        
        # Save R-squared values to a CSV file
        r_squared_values = {"HRR": r_squared_HRR, "PRR": r_squared_PRR}
        r_squared_df = pd.DataFrame(list(r_squared_values.items()), columns=['Cluster', 'R_squared'])
        r_squared_df.to_csv(os.path.join(outfile_path, 'r_squared_values_prr_fitters_nonfitters.csv'), index=False)
        print("\nGoodness of Fit (R-squared) values saved to 'r_squared_values.csv'.")

#%%

plt.scatter(df_wide['INITIAL_IMPAIRMENT'], df_wide['CHANGE_OBSERVED'], marker='o')
#plt.plot(x_initial, regression_line_initial)
plt.scatter(df_wide['INITIAL_IMPAIRMENT'], df_wide['CHANGE_PREDICTED'] )
#%%
# Extract the data from the DataFrame
x = df_wide['CHANGE_OBSERVED']
y = df_wide['CHANGE_PREDICTED']

# Create the scatter plot
plt.scatter(x, y, label='Data Points')

# Fit a linear regression line (first degree polynomial)
m, b = np.polyfit(x, y, 1)

# Plot the fitted line
plt.plot(x, m * x + b, color='red', label='Fitted Line',marker="o",alpha=0.5)

# Add labels and title
plt.xlabel('Observed Change')
plt.ylabel('Predicted Change')
plt.title('Observed vs. Predicted Change with Fitted Line')
plt.legend()

# Display the plot
plt.show()
