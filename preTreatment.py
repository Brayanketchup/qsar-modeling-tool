
import pandas as pd
import numpy as np




def filter_features(data, variance_cutoff, inter_r2_cutoff):
    """
    Filters features based on variance cut-off and inter-feature R² cut-off.
    
    Parameters:
    - data: pd.DataFrame - DataFrame containing features as columns.
    - variance_cutoff: float - Minimum variance required to keep a feature.
    - inter_r2_cutoff: float - Maximum allowable R² (squared correlation) between features.
    
    Returns:
    - pd.DataFrame - Filtered DataFrame with features meeting the variance and inter-R² criteria.
    """
    # Step 1: Variance Filter
    high_variance_features = data.loc[:, data.var() > variance_cutoff]
    
    # Step 2: Correlation Filter
    corr_matrix = high_variance_features.corr().abs() ** 2  # Inter-feature R² values
    
    # Select features to keep
    to_drop = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > inter_r2_cutoff:
                # Add the second feature to drop list (remove redundancy)
                to_drop.add(corr_matrix.columns[j])
                
    filtered_features = high_variance_features.drop(columns=list(to_drop))
    
    print("filter data: ---------------------------------------")
    print(filtered_features)
    print("original data: ---------------------------------------")
    print(data)
    
    return filtered_features