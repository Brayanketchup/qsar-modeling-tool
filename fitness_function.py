import numpy as np
import pandas as pd
from numpy.linalg import det, LinAlgError

from sklearn.linear_model import LinearRegression
from scipy.stats import sem, t
# from statsmodels.multivariate.manova import MANOVA
# from sklearn.discriminant_analysis import MANOVA
# from statsmodels.multivariate.manova import MANOVA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


# def compute_wilks_lambda(x,y,labels):
#     print('ignore this is a label for developing purposes')
#     return 1   

# # def calculate_wilks_lambda(matrixX, classY, y_labels):
# import numpy as np

# def calculate_wilks_lambda(X, y, reg=1e-5):
#     # Identify unique class labels
#     classes = np.unique(y)
#     if len(classes) != 2:
#         raise ValueError("This function only supports datasets with two unique labels (binary classification).")
    
#     # Separate data by class
#     class_1_data = X[y == classes[0]]
#     class_2_data = X[y == classes[1]]
    
#     # Calculate means for each class and the overall mean
#     mean_class_1 = np.mean(class_1_data, axis=0)
#     mean_class_2 = np.mean(class_2_data, axis=0)
#     overall_mean = np.mean(X, axis=0)
    
#     # Within-Class Scatter Matrix (S_W)
#     S_W = np.zeros((X.shape[1], X.shape[1]))
#     for data in [class_1_data, class_2_data]:
#         mean_vector = np.mean(data, axis=0)
#         scatter_matrix = np.dot((data - mean_vector).T, (data - mean_vector))
#         S_W += scatter_matrix
    
#     # Add regularization to prevent singular matrix issues
#     S_W += reg * np.identity(S_W.shape[0])
    
#     # Between-Class Scatter Matrix (S_B)
#     mean_diff_class_1 = (mean_class_1 - overall_mean).reshape(-1, 1)
#     mean_diff_class_2 = (mean_class_2 - overall_mean).reshape(-1, 1)
#     S_B = class_1_data.shape[0] * np.dot(mean_diff_class_1, mean_diff_class_1.T) + \
#           class_2_data.shape[0] * np.dot(mean_diff_class_2, mean_diff_class_2.T)
    
#     # Calculate determinants of S_W and (S_W + S_B)
#     det_S_W = np.linalg.det(S_W)
#     det_S_W_plus_S_B = np.linalg.det(S_W + S_B)
    
#     # Calculate Wilks' Lambda
#     wilks_lambda = det_S_W / det_S_W_plus_S_B
#     wilks_lambda = max(min(wilks_lambda, 1), 0)
    
#     # return wilks_lambda
#     return wilks_lambda




# def calculate_wilks_lambda(matrixX, classY, y_labels):
#     """
#     Compute Wilks' Lambda for a binary classification task.

#     Parameters:
#     matrixX : numpy.ndarray
#         Feature matrix.
#     classY : numpy.ndarray
#         Target labels corresponding to matrixX.
#     y_labels : numpy.ndarray
#         Full set of original target labels to determine unique classes.

#     Returns:
#     float
#         Wilks' Lambda statistic, measuring class separability.
#     """
#     unique_classes = np.unique(y_labels)
#     if len(unique_classes) != 2:
#         raise ValueError("The label vector must have exactly 2 distinct classes. Found: {}".format(unique_classes))
    
#     # Separate class labels
#     inactiveId, activeId = unique_classes[0], unique_classes[1]

#     # Initializing the within-class and between-class scatter matrices
#     n_features = matrixX.shape[1]
#     S_W = np.zeros((n_features, n_features))
#     S_B = np.zeros((n_features, n_features))

#     # Overall mean of the data
#     mean_overall = np.mean(matrixX, axis=0)

#     # Calculate within-class and between-class scatters
#     for cls in [inactiveId, activeId]:
#         class_data = matrixX[classY == cls]
#         class_mean = np.mean(class_data, axis=0)
#         n_cls = class_data.shape[0]
        
#         # Within-class scatter for the current class
#         S_W += np.dot((class_data - class_mean).T, (class_data - class_mean))

#         # Between-class scatter for the current class
#         mean_diff = (class_mean - mean_overall).reshape(n_features, 1)
#         S_B += n_cls * np.dot(mean_diff, mean_diff.T)

#     # Regularization to avoid singularity
#     reg_term = 1e-6 * np.eye(n_features)
#     S_W_reg = S_W + reg_term

#     # Calculate Wilks' Lambda using determinants
#     try:
#         lambda_ = det(S_W_reg) / det(S_W_reg + S_B)
#     except np.linalg.LinAlgError:
#         lambda_ = 1  # Fall back to the least discriminative scenario

#     return max(min(lambda_, 1), 0)  # Ensure lambda is within the [0, 1] range

# ///////// //////////

# def calculate_wilks_lambda(y, X):
#     """
#     Calculate Wilks' Lambda for a dataset.
    
#     Parameters:
#     - y: array-like, shape (n_samples,), labels or classes for each sample.
#     - X: array-like, shape (n_samples, n_features), feature data.

#     Returns:
#     - float: Wilks' Lambda value (between 0 and 1).
#     """
#     # Convert inputs to numpy arrays
#     y = np.array(y)
#     X = np.array(X)

#     # Calculate the overall mean of the dataset
#     overall_mean = np.mean(X, axis=0)

#     # Initialize the within-group scatter and total scatter matrices
#     within_group_scatter = np.zeros((X.shape[1], X.shape[1]))
#     total_scatter = np.zeros((X.shape[1], X.shape[1]))

#     # Calculate within-group scatter matrix
#     for label in np.unique(y):
#         group_data = X[y == label]
#         group_mean = np.mean(group_data, axis=0)
#         within_group_scatter += np.dot((group_data - group_mean).T, (group_data - group_mean))

#     # Calculate total scatter matrix
#     total_scatter = np.dot((X - overall_mean).T, (X - overall_mean))

#     # Calculate determinants
#     try:
#         det_within = np.linalg.det(within_group_scatter)
#         det_total = np.linalg.det(total_scatter)
#         if det_total == 0:
#             return 1  # Avoid division by zero
#         # Calculate Wilks' Lambda
#         wilks_lambda_value = det_within / det_total
#     except np.linalg.LinAlgError:
#         # Handle cases where determinants cannot be computed
#         wilks_lambda_value = 1

#     # Ensure result is between 0 and 1
#     return max(0, min(wilks_lambda_value, 1))






def calculate_fitness_score_maeBased(MAE, SD, range_train):
    
    
    low_threshold_ratio = 0.15
    high_threshold_ratio = 0.25

    low_threshold = low_threshold_ratio * range_train
    high_threshold = high_threshold_ratio * range_train

    first_component = max(0, (low_threshold - MAE) / low_threshold)
    second_component = max(0, (high_threshold - (MAE + 3 * SD)) / high_threshold)

    fitness_score = first_component + second_component
    return fitness_score


def leave_one_out_cross_validation(matrixX, classY, leverages):

    # Initializing parameters
    n_samples = matrixX.shape[0]
    loo_errors = []
    predictions = []
    actuals = []
    

    # hat_matrix = matrixX @ np.linalg.inv(matrixX.T @ matrixX) @ matrixX.T
    # leverages = np.diagonal(hat_matrix)
    
    for i in range(n_samples):
        # Leave one sample out
        X_train = np.delete(matrixX, i, axis=0)
        y_train = np.delete(classY, i, axis=0)
        X_test = matrixX[i].reshape(1, -1)
        y_test = classY[i]
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)[0]
        
        # Record predictions and actuals
        predictions.append(y_pred)
        actuals.append(y_test)

        loo_errors.append(np.abs(y_pred - y_test))
    
    
    # calculating Q2(LOO)
    actuals = np.array(actuals)
    predictions = np.array(predictions)
    mean_actuals = np.mean(actuals)
    ss_res = np.sum((actuals - predictions) ** 2)
    ss_tot = np.sum((actuals - mean_actuals) ** 2)
    Q2_LOO = 1 - (ss_res / ss_tot)
    
    # Calculating(Avg_Rm2(LOO))
    # Avg_Rm2_LOO = 1 - (ss_res / ss_tot)
    Avg_Rm2_LOO = 1 - np.sum((actuals - predictions)**2 / (1 - leverages)) / ss_tot
    # Avg_Rm2_LOO = 1 - np.sum((y_test - y_pred)**2 / (1 - leverages)) / ss_tot


    mae = np.mean(loo_errors)

    # Calculate standard error of the MAE
    std_err = sem(loo_errors)

    # Determine the t-critical value for 95% confidence
    t_critical = t.ppf(0.975, df=n_samples - 1)  # two-sided t-test for 95% CI
    
    # Calculate 95% confidence interval
    mae95 = mae + t_critical * std_err
    sd95 = t_critical * std_err
    
    
    return Q2_LOO, Avg_Rm2_LOO, mae95, sd95

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def get_rm2_test(y_true, y_pred, min_y, max_y):
    # Normalize data inline
    scale_y_true = (y_true - min_y) / (max_y - min_y)
    scale_y_pred = (y_pred - min_y) / (max_y - min_y)

    # Calculate Pearson correlation coefficient
    pearson_r = np.corrcoef(scale_y_true, scale_y_pred)[0, 1]

    # Calculate scaled mean squared error using numpy
    scale_mse = np.mean((scale_y_true - scale_y_pred) ** 2)
    scale_var_y_true = np.var(scale_y_true)

    # Scaled RM2 calculations
    scale_rm2 = pearson_r * (1 - np.sqrt(abs(pearson_r - (1 - scale_mse / scale_var_y_true))))
    scale_rm2_dash = scale_rm2  # Assuming RM2 dash is similar for demonstration purposes

    scale_avg_rm2 = (scale_rm2 + scale_rm2_dash) / 2
    
    return scale_avg_rm2

def fitness_mae_based(range_train, mae95_train_loo, sd95_train_loo):
    """
    Calculates the fitness score based on MAE criteria.

    Parameters:
    range_train (float): The range of observed training values (Y).
    mae95_train_loo (float): The MAE at 95% confidence level from LOO cross-validation.
    sd95_train_loo (float): The standard deviation of absolute errors at 95% confidence level from LOO cross-validation.

    Returns:
    float: The calculated fitness score based on MAE criteria.
    """
    factor1 = 0.15 * range_train
    factor2 = 0.25 * range_train
    score_part1 = (factor1 - mae95_train_loo) / factor1
    score_part2 = (factor2 - (mae95_train_loo + 3 * sd95_train_loo)) / factor2

    # Calculate total fitness score by summing the two parts
    fitness_score = score_part1 + score_part2

    return fitness_score




def calculate_prediction_quality_test(mae):
    
    if mae < 0.1:
        return 'GOOD'
    elif mae < 0.5:
        return 'MODERATE'
    else:
        return 'BAD'
    
    
    
# def fitness_MCC_WilksLambda_based(mcc, wilks):
#     if mcc > wilks:
#         fitness_score = abs(mcc - wilks)
#     else:
#         fitness_score = 0
#     return fitness_score