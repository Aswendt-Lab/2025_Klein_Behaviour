# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 20:23:18 2025

@author: arefk
"""

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

def compare_regression_lines(df, x_col, y_col, group_col):
    """
    Compare two regression lines using an ANCOVA approach.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data.
    x_col : str
        Name of the continuous predictor column.
    y_col : str
        Name of the dependent variable column.
    group_col : str
        Name of the categorical group column (with two levels).

    Returns:
    --------
    model : statsmodels regression results object
        Fitted OLS regression model.
    anova_table : pandas.DataFrame
        ANOVA table from the fitted model.

    The function fits a model of the form:
        y ~ x * group
    where the interaction term (x:group) tests whether the slopes differ
    between the groups. If the p-value for the interaction term is significant,
    the slopes of the two lines are statistically different.
    """
    
    # Create the formula for the model including interaction
    formula = f"{y_col} ~ {x_col} * {group_col}"
    
    # Fit the OLS regression model
    model = smf.ols(formula, data=df).fit()
    
    # Perform an ANOVA on the model to see the effect of predictors and the interaction term
    anova_table = sm.stats.anova_lm(model, typ=2)
    print("ANOVA Table:")
    print(anova_table)
    print("\nModel Summary:")
    print(model.summary())
    
    return model, anova_table

# Example usage:
if __name__ == "__main__":
    # Create an example dataset
    import numpy as np
    
    np.random.seed(42)
    size = 50
    df_example = pd.DataFrame({
        'x': np.concatenate([np.linspace(0, 10, size), np.linspace(0, 10, size)]),
        'group': ['A'] * size + ['B'] * size,
    })
    # Generate y with different slopes and intercepts for groups A and B.
    df_example['y'] = np.where(
        df_example['group'] == 'A',
        2 + 1.5 * df_example['x'] + np.random.normal(0, 1, size*2),
        1 + 2.5 * df_example['x'] + np.random.normal(0, 1, size*2)
    )
    
    # Compare the regression lines for the two groups
    model, anova_table = compare_regression_lines(df_example, x_col='x', y_col='y', group_col='group')
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)
