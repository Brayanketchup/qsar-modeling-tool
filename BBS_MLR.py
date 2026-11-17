

# import numpy as np
# import random
# import streamlit as st
# import pandas as pd
# from classification_division import kennard_stone_sampling, euclidean_sampling
# from sklearn.model_selection import train_test_split
# from scipy.stats import sem
# from fitness_function import  calculate_prediction_quality_test, fitness_mae_based, leave_one_out_cross_validation, get_rm2_test
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_absolute_error, r2_score
# import matplotlib.pyplot as plt
# import seaborn as sns
# # import shap
    
    
# def evaluate_individual(individual, X, y, feature_names, division_method, test_size, fitness_method):
#     # Select features based on the individual's genes.
#     selected_features = [i for i, gene in enumerate(individual) if gene == 1]
   
#     if not selected_features:
#         return 0  # No features selected, worst case.
    
#     X_selected = np.array(X)[:, selected_features]
#     if division_method == "random division":
#         X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=test_size, random_state=42)
#     elif division_method == "Kennard stone":
#         X_train, X_test, y_train, y_test = kennard_stone_sampling(X_selected, y, test_size=test_size)
#     elif division_method == "Euclidean":
#         X_train, X_test, y_train, y_test = euclidean_sampling(X_selected, y, test_size=test_size)
 
 
 
#     hat_matrix_train = X_train @ np.linalg.inv(X_train.T @ X_train) @ X_train.T
#     leverages_train = np.diagonal(hat_matrix_train)
    
#     hat_matrix_test = X_test @ np.linalg.inv(X_test.T @ X_test) @ X_test.T
#     leverages_test = np.diagonal(hat_matrix_test)
    
#     model = LinearRegression()
#     model.fit(X_train, y_train)
    
#     # predictions that will be used for further analisys
#     y_pred_train = model.predict(X_train)
#     y_pred_test = model.predict(X_test)
    
#     residuals_test = y_test - y_pred_test
#     residuals_train = y_train - y_pred_train
    
#     standardized_residuals_test = residuals_test / np.std(residuals_test)
#     standardized_residuals_train = residuals_train / np.std(residuals_train)
    
#     # Scale observed and predicted data (as seen in Java code)
#     min_y_train = np.min(y_train)
#     max_y_train = np.max(y_train)
    
#     y_train_scaled = (y_train - min_y_train) / (max_y_train - min_y_train)
#     y_test_scaled = (y_test - min_y_train) / (max_y_train - min_y_train)
#     y_pred_scaled = (y_pred_test - min_y_train) / (max_y_train - min_y_train)
    
#     # Compute sum of squares for scaled and original data
#     SS_tot_train_scaled = np.sum((y_train_scaled - np.mean(y_train_scaled)) ** 2)
#     SS_res_scaled = np.sum((y_test_scaled - y_pred_scaled) ** 2)
    
#     # values to return
#     r2 = r2_score(y_test, y_pred_test)  
#     mae_training = mean_absolute_error(y_train, y_pred_train)
#     mae_test = mean_absolute_error(y_test, y_pred_test)
#     prediction_quality_train = calculate_prediction_quality_test(mae_training)
#     prediction_quality_test = calculate_prediction_quality_test(mae_test)
#     Q2F1 = 1 - SS_res_scaled / SS_tot_train_scaled
#     Q2F2 = 1 - SS_res_scaled / np.sum((y_test_scaled - np.mean(y_test_scaled)) ** 2)
    
#         # min_y_test max_y_test
        
#     min_y_train = np.min(y_train)
#     max_y_train = np.min(y_train)
#     min_y_test = np.min(y_test)
#     max_y_test = np.max(y_test)
    
#     avg_rm2_test = get_rm2_test(y_test, y_pred_test, min_y_test, max_y_test)

#     # composing the final equation
#     coefficients = model.coef_
#     intercept = model.intercept_
#     standard_errors = sem(X_train, axis=0)
#     selected_feature_names = [feature_names[i] for i in selected_features]

#     equation_parts = [
#         f"{coeff:.3f} (+/- {se:.3f}) * {name}" for coeff, se, name in zip(coefficients, standard_errors, selected_feature_names)
#     ]
#     equation = " + ".join(equation_parts)
#     final_equation = f"Score = {intercept:.3f} + {equation}"
        
        
#     Q2_LOO, Avg_Rm2_LOO, mae95, sd95 = leave_one_out_cross_validation(X_train, y_train, leverages_train)
#     range_train = max(y) - min(y) 

#     if  fitness_method == "LOO":
#         fitness_score = Q2_LOO
#         # return final_equation, r2, Q2_LOO, Avg_Rm2_LOO, mae_training, fitness_score, prediction_quality_train, Q2F1, Q2F2, avg_rm2_test, mae_test, prediction_quality_test
#     elif fitness_method == "MAE":
#         fitness_score = fitness_mae_based(range_train, mae95, sd95)
#         # return final_equation, r2, Q2_LOO, Avg_Rm2_LOO, mae_training, fitness_score, prediction_quality_train, Q2F1, Q2F2, avg_rm2_test, mae_test, prediction_quality_test
#     else:
#         print("not valid")
        
#     individual = {   
#         "features" : individual,          
#         "R2": r2, 
#         "Q2_LOO": Q2_LOO, 
#         "Avg_Rm2_LOO": Avg_Rm2_LOO, 
#         "mae_training": mae_training, 
#         "fitness_score": fitness_score, 
#         "prediction_quality_train": prediction_quality_train, 
#         "Q2F1": Q2F1, 
#         "Q2F2": Q2F2, 
#         "avg_rm2_test": avg_rm2_test, 
#         "mae_test": mae_test, 
#         "prediction_quality_test": prediction_quality_test,
#         "final_equation": final_equation,
#         "leverages_test": leverages_test,
#         "leverages_train": leverages_train,
#         "standardized_residuals_test": standardized_residuals_test,
#         "standardized_residuals_train": standardized_residuals_train,
#         "y_pred_train": y_pred_train,
#         "y_pred_test": y_pred_test,
#         'X_train': X_train, 
#         'X_test': X_test, 
#         'y_train': y_train, 
#         'y_test': y_test

#     }
    
    
    
#     return  individual

# # def BSS_LDA(arr, length):
# def BSS_MLR(X, y, division_method, feature_names, test_size, fitness_method, length):
    
#     dataset_length = X.shape[1]
#     results = []
#     n = dataset_length
            
#     # Create an index list to represent the current combination
#     indices = list(range(length))
    
    

#     while True:
#         current_individual = np.zeros(dataset_length, dtype=int)
#         for i in indices:
#             current_individual[i] = 1
        
        
        
#         # equation, R2, Q2_LOO, Avg_Rm2_LOO, mae_training, fitness_score, prediction_quality_train, Q2F1, Q2F2, avg_rm2_test, mae_test, prediction_quality_test = evaluate_individual(current_individual, X, y, feature_names, division_method, test_size, fitness_method)
#         individual = evaluate_individual(current_individual, X, y, feature_names, division_method, test_size, fitness_method)
        
#         # individual = {  
#         # "equation": equation,            
#         # "R2": R2, 
#         # "Q2_LOO": Q2_LOO, 
#         # "Avg_Rm2_LOO": Avg_Rm2_LOO, 
#         # "mae_training": mae_training, 
#         # "fitness_score": fitness_score, 
#         # "prediction_quality_train": prediction_quality_train, 
#         # "Q2F1": Q2F1, 
#         # "Q2F2": Q2F2, 
#         # "avg_rm2_test": avg_rm2_test, 
#         # "mae_test": mae_test, 
#         # "prediction_quality_test": prediction_quality_test
#         # }
              
#         results.append(individual)
        
#         # Find the rightmost index that can be incremented
#         for i in reversed(range(length)):
#             if indices[i] != i + n - length:
#                 break
#         else:
#             # If no index can be incremented, we're done
#             break
        
#         # Increment the current index and reset following indices
#         indices[i] += 1
#         for j in range(i + 1, length):
#             indices[j] = indices[j - 1] + 1
            
    
#     if  fitness_method == "LOO":
#         # q2 closer to 1 is better
#         results.sort(key=lambda ind: ind['fitness_score'], reverse=True)
#     elif  fitness_method == "MAE":
#         # mae closer to 0 is better 
#         results.sort(key=lambda ind: ind['fitness_score'], reverse=False)
            
#     return results

    
    
    
# def BBS_MLR_main(X, y, feature_names, equation_length, division_method, test_size, fitness_method, random_seed):
    
    
#     if random_seed is not None:
#         random.seed(random_seed)  
#         np.random.seed(random_seed)
        
    

#     # Initialize the population
#     Individuals = BSS_MLR(X, y, division_method, feature_names, test_size, fitness_method, equation_length)
    
    
#     df_individuals = pd.DataFrame(Individuals)
    
#     best_individual = Individuals[0]
#     best_individual
    
    
#     # equation = best_individual["equation"]
#     R2 = best_individual["R2"]
#     Q2_LOO = best_individual["Q2_LOO"]
#     Avg_Rm2_LOO = best_individual["Avg_Rm2_LOO"]
#     mae_training = best_individual["mae_training"]
#     fitness_score = best_individual["fitness_score"]
#     prediction_quality_train = best_individual["prediction_quality_train"]
#     Q2F1 = best_individual["Q2F1"]
#     Q2F2 = best_individual["Q2F2"]
#     avg_rm2_test = best_individual["avg_rm2_test"]
#     mae_test = best_individual["mae_test"]
#     prediction_quality_test = best_individual["prediction_quality_test"]
#     final_equation = best_individual["final_equation"]
#     leverages_test = best_individual["leverages_test"]
#     leverages_train = best_individual["leverages_train"]
#     standardized_residuals_test = best_individual["standardized_residuals_test"]
#     standardized_residuals_train = best_individual["standardized_residuals_train"]
#     y_pred_train = best_individual["y_pred_train"]
#     y_pred_test = best_individual["y_pred_test"]
#     y_train = best_individual["y_train"]
#     y_test = best_individual["y_test"]
#     X_train = best_individual["X_train"]
#     X_test = best_individual["X_test"]
    
    
#     st.title('Model Performance Overview')

#     # Display the model equation
#     st.subheader('Model Equation')
#     # st.write(f"**Equation:** {best_individual['equation']}")
#     st.write(f"*Final Equation:* {best_individual['final_equation']}")

#     # Metrics presentation
#     st.subheader('Model Metrics')
#     metrics = {
#         "R² (Coefficient of Determination)": best_individual["R2"],
#         "Q² LOO (Leave-One-Out Cross Validation)": best_individual["Q2_LOO"],
#         "Average RM² LOO": best_individual["Avg_Rm2_LOO"],
#         "MAE Training": best_individual["mae_training"],
#         "Fitness Score": best_individual["fitness_score"],
#         "Prediction Quality (Training)": best_individual["prediction_quality_train"],
#         "Q²F1 (Model Quality Metric 1)": best_individual["Q2F1"],
#         "Q²F2 (Model Quality Metric 2)": best_individual["Q2F2"],
#         "Average RM² Test": best_individual["avg_rm2_test"],
#         "MAE Test": best_individual["mae_test"],
#         "Prediction Quality (Test)": best_individual["prediction_quality_test"]
#     }
#     st.table(metrics)


#     st.write("X_train")
#     st.write(X_train)
    
#     st.write("y_train")
#     st.write(y_train)
    
#     plt.figure(figsize=(10, 6))
#     sns.scatterplot(x=leverages_test, y=standardized_residuals_test)
#     plt.axhline(y=0, color='red', linestyle='--')
#     plt.xlabel('Leverage')
#     plt.ylabel('Standardized Residuals')
#     plt.title('Williams Plot (test Data)')
#     st.subheader('Williams Plot (test Data)')
#     st.pyplot(plt)
    
    
#     plt.clf()

    
#     plt.figure(figsize=(10, 6))
#     sns.scatterplot(x=leverages_train, y=standardized_residuals_train)
#     plt.axhline(y=0, color='red', linestyle='--')
#     plt.xlabel('Leverage')
#     plt.ylabel('Standardized Residuals')
#     plt.title('Williams Plot (training Data)')
#     st.subheader('Williams Plot (training Data)')
#     st.pyplot(plt)
    
    
#     # Scatter plot of observed vs predicted
#     plt.figure(figsize=(10, 6))
#     sns.scatterplot(x=y_test, y=y_pred_test)
#     plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
#     plt.xlabel('Observed')
#     plt.ylabel('Predicted')
#     plt.title('Scatter Plot of Observed vs Predicted (Test Data)')
#     st.subheader('Scatter Plot of Observed vs Predicted (Test Data)')
#     st.pyplot(plt) 
    
#     plt.clf()

#     # Scatter plot of observed vs predicted
#     plt.figure(figsize=(10, 6))
#     sns.scatterplot(x=y_train, y=y_pred_train)
#     plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
#     plt.xlabel('Observed')
#     plt.ylabel('Predicted')
#     plt.title('Scatter Plot of Observed vs Predicted (Training Data)')
#     st.subheader('Scatter Plot of Observed vs Predicted (Training Data)')
#     st.pyplot(plt)
    
#     # print('y_pred_train', len(y_pred_train), 'y_train', len(y_train))
    
#     st.write("Best Individuals")
#     st.write(df_individuals.head())
    
#     # ///////////////////////////////
#     #  write the mlr shap plot code and then upload ga_mlr to work as BBS_MLR
    
    
    
#     # st.subheader('Confusion Matrix')
#     # st.pyplot(fig)


import numpy as np
import random
import streamlit as st
import pandas as pd
from division_methods import kennard_stone_sampling, euclidean_sampling
from sklearn.model_selection import train_test_split
from scipy.stats import sem
from fitness_function import  calculate_prediction_quality_test, fitness_mae_based, leave_one_out_cross_validation, get_rm2_test
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
# import shap
    
    
def evaluate_individual(individual, X, y, feature_names, division_method, test_size, fitness_method):
    # Select features based on the individual's genes.
    selected_features = [i for i, gene in enumerate(individual) if gene == 1]
   
    if not selected_features:
        return 0  # No features selected, worst case.
    
    X_selected = np.array(X)[:, selected_features]
    if division_method == "random division":
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=test_size, random_state=42)
    elif division_method == "Kennard stone":
        X_train, X_test, y_train, y_test = kennard_stone_sampling(X_selected, y, test_size=test_size)
    elif division_method == "Euclidean":
        X_train, X_test, y_train, y_test = euclidean_sampling(X_selected, y, test_size=test_size)
 
 
 
    hat_matrix_train = X_train @ np.linalg.inv(X_train.T @ X_train) @ X_train.T
    leverages_train = np.diagonal(hat_matrix_train)
    
    hat_matrix_test = X_test @ np.linalg.inv(X_test.T @ X_test) @ X_test.T
    leverages_test = np.diagonal(hat_matrix_test)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # predictions that will be used for further analisys
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    residuals_test = y_test - y_pred_test
    residuals_train = y_train - y_pred_train
    
    standardized_residuals_test = residuals_test / np.std(residuals_test)
    standardized_residuals_train = residuals_train / np.std(residuals_train)
    
    # Scale observed and predicted data (as seen in Java code)
    min_y_train = np.min(y_train)
    max_y_train = np.max(y_train)
    
    y_train_scaled = (y_train - min_y_train) / (max_y_train - min_y_train)
    y_test_scaled = (y_test - min_y_train) / (max_y_train - min_y_train)
    y_pred_scaled = (y_pred_test - min_y_train) / (max_y_train - min_y_train)
    
    # Compute sum of squares for scaled and original data
    SS_tot_train_scaled = np.sum((y_train_scaled - np.mean(y_train_scaled)) ** 2)
    SS_res_scaled = np.sum((y_test_scaled - y_pred_scaled) ** 2)
    
    # values to return
    r2 = r2_score(y_test, y_pred_test)  
    mae_training = mean_absolute_error(y_train, y_pred_train)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    prediction_quality_train = calculate_prediction_quality_test(mae_training)
    prediction_quality_test = calculate_prediction_quality_test(mae_test)
    Q2F1 = 1 - SS_res_scaled / SS_tot_train_scaled
    Q2F2 = 1 - SS_res_scaled / np.sum((y_test_scaled - np.mean(y_test_scaled)) ** 2)
    
        # min_y_test max_y_test
        
    min_y_train = np.min(y_train)
    max_y_train = np.min(y_train)
    min_y_test = np.min(y_test)
    max_y_test = np.max(y_test)
    
    avg_rm2_test = get_rm2_test(y_test, y_pred_test, min_y_test, max_y_test)

    # composing the final equation
    coefficients = model.coef_
    intercept = model.intercept_
    standard_errors = sem(X_train, axis=0)
    selected_feature_names = [feature_names[i] for i in selected_features]

    equation_parts = [
        f"{coeff:.3f} (+/- {se:.3f}) * {name}" for coeff, se, name in zip(coefficients, standard_errors, selected_feature_names)
    ]
    equation = " + ".join(equation_parts)
    final_equation = f"Score = {intercept:.3f} + {equation}"
        
        
    Q2_LOO, Avg_Rm2_LOO, mae95, sd95 = leave_one_out_cross_validation(X_train, y_train, leverages_train)
    range_train = max(y) - min(y) 

    if  fitness_method == "LOO":
        fitness_score = Q2_LOO
        # return final_equation, r2, Q2_LOO, Avg_Rm2_LOO, mae_training, fitness_score, prediction_quality_train, Q2F1, Q2F2, avg_rm2_test, mae_test, prediction_quality_test
    elif fitness_method == "MAE":
        fitness_score = fitness_mae_based(range_train, mae95, sd95)
        # return final_equation, r2, Q2_LOO, Avg_Rm2_LOO, mae_training, fitness_score, prediction_quality_train, Q2F1, Q2F2, avg_rm2_test, mae_test, prediction_quality_test
    else:
        print("not valid")
        
    individual = {
        "features": individual,          
        "R2": r2, 
        "Q2_LOO": Q2_LOO, 
        "Avg_Rm2_LOO": Avg_Rm2_LOO, 
        "mae_training": mae_training, 
        "fitness_score": fitness_score, 
        "prediction_quality_train": prediction_quality_train, 
        "Q2F1": Q2F1, 
        "Q2F2": Q2F2, 
        "avg_rm2_test": avg_rm2_test, 
        "mae_test": mae_test, 
        "prediction_quality_test": prediction_quality_test,
        "final_equation": final_equation,
        "leverages_test": leverages_test,
        "leverages_train": leverages_train,
        "standardized_residuals_test": standardized_residuals_test,
        "standardized_residuals_train": standardized_residuals_train,
        "y_pred_train": y_pred_train,
        "y_pred_test": y_pred_test,
        'X_train': X_train, 
        'X_test': X_test, 
        'y_train': y_train, 
        'y_test': y_test
    }
    
    
    
    return  individual

# def BSS_LDA(arr, length):
def BSS_MLR(X, y, division_method, feature_names, test_size, fitness_method, length):
    
    dataset_length = X.shape[1]
    results = []
    n = dataset_length
            
    # Create an index list to represent the current combination
    indices = list(range(length))
    
    

    while True:
        current_individual = np.zeros(dataset_length, dtype=int)
        for i in indices:
            current_individual[i] = 1
        
        
        
        # equation, R2, Q2_LOO, Avg_Rm2_LOO, mae_training, fitness_score, prediction_quality_train, Q2F1, Q2F2, avg_rm2_test, mae_test, prediction_quality_test = evaluate_individual(current_individual, X, y, feature_names, division_method, test_size, fitness_method)
        individual = evaluate_individual(current_individual, X, y, feature_names, division_method, test_size, fitness_method)
        
        # individual = {  
        # "equation": equation,            
        # "R2": R2, 
        # "Q2_LOO": Q2_LOO, 
        # "Avg_Rm2_LOO": Avg_Rm2_LOO, 
        # "mae_training": mae_training, 
        # "fitness_score": fitness_score, 
        # "prediction_quality_train": prediction_quality_train, 
        # "Q2F1": Q2F1, 
        # "Q2F2": Q2F2, 
        # "avg_rm2_test": avg_rm2_test, 
        # "mae_test": mae_test, 
        # "prediction_quality_test": prediction_quality_test
        # }
              
        results.append(individual)
        
        # Find the rightmost index that can be incremented
        for i in reversed(range(length)):
            if indices[i] != i + n - length:
                break
        else:
            # If no index can be incremented, we're done
            break
        
        # Increment the current index and reset following indices
        indices[i] += 1
        for j in range(i + 1, length):
            indices[j] = indices[j - 1] + 1
            
    
    if  fitness_method == "LOO":
        # q2 closer to 1 is better
        results.sort(key=lambda ind: ind['fitness_score'], reverse=True)
    elif  fitness_method == "MAE":
        # mae closer to 0 is better 
        results.sort(key=lambda ind: ind['fitness_score'], reverse=False)
            
    return results

    
    
    
def BBS_MLR_main(X, y, feature_names, equation_length, division_method, test_size, fitness_method, random_seed):
    
    
    if random_seed is not None:
        random.seed(random_seed)  
        np.random.seed(random_seed)
        
    

    # Initialize the population
    Individuals = BSS_MLR(X, y, division_method, feature_names, test_size, fitness_method, equation_length)
    
    
    df_individuals = pd.DataFrame(Individuals)
    
    best_individual = Individuals[0]
    best_individual
    
    
    # equation = best_individual["equation"]
    R2 = best_individual["R2"]
    Q2_LOO = best_individual["Q2_LOO"]
    Avg_Rm2_LOO = best_individual["Avg_Rm2_LOO"]
    mae_training = best_individual["mae_training"]
    fitness_score = best_individual["fitness_score"]
    prediction_quality_train = best_individual["prediction_quality_train"]
    Q2F1 = best_individual["Q2F1"]
    Q2F2 = best_individual["Q2F2"]
    avg_rm2_test = best_individual["avg_rm2_test"]
    mae_test = best_individual["mae_test"]
    prediction_quality_test = best_individual["prediction_quality_test"]
    final_equation = best_individual["final_equation"]
    leverages_test = best_individual["leverages_test"]
    leverages_train = best_individual["leverages_train"]
    standardized_residuals_test = best_individual["standardized_residuals_test"]
    standardized_residuals_train = best_individual["standardized_residuals_train"]
    y_pred_train = best_individual["y_pred_train"]
    y_pred_test = best_individual["y_pred_test"]
    y_train = best_individual["y_train"]
    y_test = best_individual["y_test"]
    X_train = best_individual["X_train"]
    X_test = best_individual["X_test"]
    
    
    st.title('Model Performance Overview')

    # Display the model equation
    st.subheader('Model Equation')
    # st.write(f"**Equation:** {best_individual['equation']}")
    st.write(f"*Final Equation:* {best_individual['final_equation']}")

    # Metrics presentation
    st.subheader('Model Metrics')
    metrics = {
        "R² (Coefficient of Determination)": best_individual["R2"],
        "Q² LOO (Leave-One-Out Cross Validation)": best_individual["Q2_LOO"],
        "Average RM² LOO": best_individual["Avg_Rm2_LOO"],
        "MAE Training": best_individual["mae_training"],
        "Fitness Score": best_individual["fitness_score"],
        "Prediction Quality (Training)": best_individual["prediction_quality_train"],
        "Q²F1 (Model Quality Metric 1)": best_individual["Q2F1"],
        "Q²F2 (Model Quality Metric 2)": best_individual["Q2F2"],
        "Average RM² Test": best_individual["avg_rm2_test"],
        "MAE Test": best_individual["mae_test"],
        "Prediction Quality (Test)": best_individual["prediction_quality_test"]
    }
    st.table(metrics)


    st.write("X_train")
    st.write(X_train)
    
    st.write("y_train")
    st.write(y_train)
    
    # plt.figure(figsize=(10, 6))
    # sns.scatterplot(x=leverages_test, y=standardized_residuals_test)
    # plt.axhline(y=0, color='red', linestyle='--')
    # plt.xlabel('Leverage')
    # plt.ylabel('Standardized Residuals')
    # plt.title('Williams Plot (test Data)')
    # st.subheader('Williams Plot (test Data)')
    # st.pyplot(plt)
    
    
    

    
    # plt.figure(figsize=(10, 6))
    # sns.scatterplot(x=leverages_train, y=standardized_residuals_train)
    # plt.axhline(y=0, color='red', linestyle='--')
    # plt.xlabel('Leverage')
    # plt.ylabel('Standardized Residuals')
    # plt.title('Williams Plot (training Data)')
    # st.subheader('Williams Plot (training Data)')
    # st.pyplot(plt)
    
    plt.figure(figsize=(10, 6))

    # Plot test data with circles
    sns.scatterplot(x=leverages_test, y=standardized_residuals_test, marker='o', label='Test Data')

    # Plot training data with triangles
    sns.scatterplot(x=leverages_train, y=standardized_residuals_train, marker='^', label='Training Data')

    # Add a horizontal line at y=0
    plt.axhline(y=0, color='red', linestyle='--')

    # Set labels and title
    plt.xlabel('Leverage')
    plt.ylabel('Standardized Residuals')
    plt.title('Williams Plot (Training and Test Data)')

    # Display legend
    plt.legend()

    # Show the plot in Streamlit
    st.subheader('Williams Plot (Training and Test Data)')
    st.pyplot(plt)

# /////////////////
    
    
    plt.clf()
    
    # Scatter plot of observed vs predicted
    # plt.figure(figsize=(10, 6))
    # sns.scatterplot(x=y_test, y=y_pred_test)
    # plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    # plt.xlabel('Observed')
    # plt.ylabel('Predicted')
    # plt.title('Scatter Plot of Observed vs Predicted (Test Data)')
    # st.subheader('Scatter Plot of Observed vs Predicted (Test Data)')
    # st.pyplot(plt) 
    
    # plt.clf()

    # # Scatter plot of observed vs predicted
    # plt.figure(figsize=(10, 6))
    # sns.scatterplot(x=y_train, y=y_pred_train)
    # plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
    # plt.xlabel('Observed')
    # plt.ylabel('Predicted')
    # plt.title('Scatter Plot of Observed vs Predicted (Training Data)')
    # st.subheader('Scatter Plot of Observed vs Predicted (Training Data)')
    # st.pyplot(plt)
    # Create a single figure for both training and test data
    plt.figure(figsize=(10, 6))
    
    # Scatter plot for test data with circles
    sns.scatterplot(x=y_test, y=y_pred_test, marker='o', label='Test Data')
    
    # Scatter plot for training data with triangles
    sns.scatterplot(x=y_train, y=y_pred_train, marker='^', label='Training Data')
    
    # Plot the y=x line to indicate the ideal prediction
    plt.plot([min(y_train.min(), y_test.min()), max(y_train.max(), y_test.max())], 
             [min(y_train.min(), y_test.min()), max(y_train.max(), y_test.max())], 
             'r--')
    
    # Set labels, title, and legend
    plt.xlabel('Observed')
    plt.ylabel('Predicted')
    plt.title('Scatter Plot of Observed vs Predicted (Training and Test Data)')
    plt.legend()
    
    # Show the plot in Streamlit
    st.subheader('Scatter Plot of Observed vs Predicted (Training and Test Data)')
    st.pyplot(plt)

    
    # print('y_pred_train', len(y_pred_train), 'y_train', len(y_train))
    
    st.write("Best Individuals")
    # st.write(df_individuals.head()) error
    
    # ///////////////////////////////
    #  write the mlr shap plot code and then upload ga_mlr to work as BBS_MLR
    
    
    
    # st.subheader('Confusion Matrix')
    # st.pyplot(fig)