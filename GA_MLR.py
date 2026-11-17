# import numpy as np
# from deap import creator, base, tools, algorithms
# import random
# import pygad
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# import pygad
# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.metrics import accuracy_score, confusion_matrix
# import seaborn as sns
# import matplotlib.pyplot as plt
# from deap import base, creator, tools, algorithms
# import random
# from classification_division import kennard_stone_sampling, euclidean_sampling
# from copy import deepcopy
# from sklearn.preprocessing import LabelEncoder
# import numpy as np
# import random
# from sklearn.metrics import matthews_corrcoef
# from fitness_function import compute_wilks_lambda, leave_one_out_cross_validation
# from sklearn.metrics import (accuracy_score, precision_score, recall_score,
#                              f1_score, confusion_matrix, roc_curve, auc)
# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.ensemble import RandomForestClassifier
# # from sklearn.metrics import accuracy_score, confusion_matrix
# from sklearn.metrics import (accuracy_score, precision_score, recall_score,
#                              f1_score, confusion_matrix, roc_curve, auc)
# import seaborn as sns
# import matplotlib.pyplot as plt
# from classification_division import kennard_stone_sampling, euclidean_sampling
# # from Classification_GA import initialize_ga
# from fitness_function import compute_wilks_lambda
# import random
# from sklearn.metrics import matthews_corrcoef
# from scipy.stats import sem

# from sklearn.linear_model import LinearRegression

# import numpy as np
# import pandas as pd
# import streamlit as st
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.linear_model import LinearRegression
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# import scipy.stats as stats
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

    
# # init individuals functions
# def init_individual(dataset_length, equation_length):
#     """
#     dataset_length = amount of features in the dataset

#     num_ones = amount of active features per individual
#     """
#     # Check if the number of 1s is greater than the array length
#     if equation_length > dataset_length:
#         raise ValueError("Number of 1s cannot exceed the length of the array.")
    
#     # Create an array of zeros with the specified length
#     individual = np.zeros(dataset_length, dtype=int)
    
#     # Randomly choose positions to place 1s
#     ones_indices = np.random.choice(dataset_length, equation_length, replace=False)
    
#     # Set those positions to 1
#     individual[ones_indices] = 1
    
#     return individual
   
   
   
# # Generate generation 0 initialization of the first generation
# def init_population(pop_size, X, y, feature_names, division_method, test_size, dataset_length, equation_length, fitness_method):
#     initial_population =  []
#     for i in range(pop_size):
#         new_individual = init_individual(dataset_length, equation_length)
        
#         equation, R2, Q2_LOO, Avg_Rm2_LOO, mae_training, fitness_score, prediction_quality_train, Q2F1, Q2F2, avg_rm2_test, mae_test, prediction_quality_test = evaluate_individual(new_individual, X, y, feature_names, division_method, test_size, fitness_method)

#         individual = {
#             "features" : new_individual,
#             "equation": equation,
#             "R2": R2,
#             "Q2_LOO": Q2_LOO,
#             "Avg_Rm2_LOO": Avg_Rm2_LOO,
#             "mae_training": mae_training,
#             "fitness_score": fitness_score,
#             "prediction_quality_train": prediction_quality_train,
#             "Q2F1": Q2F1,
#             "Q2F2": Q2F2,
#             "avg_rm2_test": avg_rm2_test,
#             "mae_test": mae_test,
#             "prediction_quality_test": prediction_quality_test,
#         }
        
        
#         initial_population.append(individual)
        
#         # //
#         # initial_population.append(individual)
#         if  fitness_method == "LOO":
#             # q2 closer to 1 is better
#             initial_population.sort(key=lambda ind: ind['fitness_score'], reverse=True)
#         elif  fitness_method == "MAE":
#             # mae closer to 0 the better 
#             initial_population.sort(key=lambda ind: ind['fitness_score'], reverse=False)
        

#     return initial_population

# def crossover(ind1, ind2, cx_prob):
#     random_value = random.random()
    
#     ind1 = pd.Series(ind1)
#     ind2 = pd.Series(ind2)
    
#     if(random_value < cx_prob):

#         ind1_copy = ind1[:]
#         ind2_copy = ind2[:]

#         ind1_idx_list = ind1_copy[ind1_copy == 1].index
#         ind2_idx_list = ind2_copy[ind2_copy == 1].index
#         # Randomly pick one of these indices

#         ind1_idx = random.choice(ind1_idx_list) if not ind1_idx_list.empty else None
#         ind2_idx = random.choice(ind2_idx_list) if not ind2_idx_list.empty else None

#         if ind1_idx == ind2_idx and len(ind1_idx_list) > 0:
#             ind1_idx = random.choice(ind1_idx_list)


#         if ind1_idx is not None and ind2_idx is not None:
#             if ind2_copy[ind1_idx] == 0:
#                 ind2_copy[ind1_idx] = 1
#                 ind2_copy[ind2_idx] = 0

#             if ind1_copy[ind2_idx] == 0:
#                 ind1_copy[ind2_idx] = 1
#                 ind1_copy[ind1_idx] = 0
                        
#         return ind1_copy, ind2_copy
#     else:
#         return ind1, ind2

# # mutate crossed childs to integrate new changes and posivilities
# def mutation(ind, mut_prob):
    
#     random_value = random.random()
#     # print(random_value)
#     if(random_value < mut_prob):
    
#         ind_copy = ind[:]

#         # one_idx_list = ind_copy[ind_copy == 1].index
#         one_idx_list = ind_copy[ind_copy == 1].index

#         # zero_idx_list = ind_copy[ind_copy == 0].index
#         zero_idx_list = ind_copy[ind_copy == 0].index


#         flip_one_to_zero = random.choice(one_idx_list) if not one_idx_list.empty else None
#         flip_zero_to_one = random.choice(zero_idx_list) if not one_idx_list.empty else None

#         ind_copy[flip_one_to_zero]=0
#         ind_copy[flip_zero_to_one]=1
    

#         return ind_copy
#     else :
#         return ind

# # evaluate each individual in the population and give them a fitness score
# # def evaluate_individual(individual, X, y, division_method, test_size, fitness_method):
# #     # Select features based on the individual's genes.
# #     selected_features = [i for i, gene in enumerate(individual) if gene == 1]
   
# #     if not selected_features:
# #         return 0  # No features selected, worst case.
    
# #     X_selected = np.array(X)[:, selected_features]
# #     if division_method == "random division":
# #         X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=test_size, random_state=42)
# #     elif division_method == "Kennard stone":
# #         X_train, X_test, y_train, y_test = kennard_stone_sampling(X_selected, y, test_size=test_size)
# #     elif division_method == "Euclidean":
# #         X_train, X_test, y_train, y_test = euclidean_sampling(X_selected, y, test_size=test_size)
        
        
        
# #     MAE, Q2, R2, LOO_Mean, LOO_Error, Predictive_Quality = leave_one_out_cross_validation(X_train, y_train, y)


    
# #     # return the selected fitness score
# #     if  fitness_method == "LOO":
# #         if(Q2 < 0 or Q2 > 1):
# #             fitness_score = 0
# #             return fitness_score, MAE, Q2, R2, LOO_Mean, LOO_Error, Predictive_Quality
# #         else:
# #             fitness_score = Q2
# #             return fitness_score, MAE, Q2, R2, LOO_Mean, LOO_Error, Predictive_Quality
        
# #     elif fitness_method == "MAE":
            
# #         fitness_score = MAE
# #         return fitness_score, MAE, Q2, R2, LOO_Mean, LOO_Error, Predictive_Quality

# #         # if(MAE < 0 or MAE > 1 ):
# #         #     fitness_score = 1
# #         #     return fitness_score, MAE, Q2, R2, LOO_Mean, LOO_Error, Predictive_Quality
# #         # else:
# #         #     fitness_score = MAE
# #         #     return fitness_score, MAE, Q2, R2, LOO_Mean, LOO_Error, Predictive_Quality

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
 
 
 
#     hat_matrix = X_train @ np.linalg.inv(X_train.T @ X_train) @ X_train.T
#     leverages = np.diagonal(hat_matrix)       
    
#     model = LinearRegression()
#     model.fit(X_train, y_train)
    
#     # predictions that will be used for further analisys
#     y_pred_train = model.predict(X_train)
#     y_pred_test = model.predict(X_test)
    

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
        
        
#     Q2_LOO, Avg_Rm2_LOO, mae95, sd95 = leave_one_out_cross_validation(X_train, y_train, leverages)
#     range_train = max(y) - min(y) 

#     if  fitness_method == "LOO":
#         fitness_score = Q2_LOO
#         return final_equation, r2, Q2_LOO, Avg_Rm2_LOO, mae_training, fitness_score, prediction_quality_train, Q2F1, Q2F2, avg_rm2_test, mae_test, prediction_quality_test
        
#     elif fitness_method == "MAE":
#         fitness_score = fitness_mae_based(range_train, mae95, sd95)
#         return final_equation, r2, Q2_LOO, Avg_Rm2_LOO, mae_training, fitness_score, prediction_quality_train, Q2F1, Q2F2, avg_rm2_test, mae_test, prediction_quality_test
#     else:
#         print("something is wrong")        

 
# def initialize_ga(X, y, feature_names, num_generations, pop_size, mut_prob, equation_length, division_method, test_size, fitness_method, cx_prob, random_seed):
#     """
#     Initializes and evolves a population for a genetic algorithm.

#     Parameters:
#     X : array-like, feature dataset.
#     y : array-like, target dataset.
#     dataset_length : int, length of an individual's chromosome.
#     num_generations : int, number of generations to evolve.
#     pop_size : int, number of individuals in each generation.
#     mut_prob : float, probability of mutation.
#     equation_length : int, length of the equation/problem solution.
#     division_method : function, method to split dataset for training/testing.
#     test_size : float, proportion of dataset to be used as test set.
#     cx_prob : float, probability of crossover.

#     Evolves a population through selection, crossover, mutation, and introduces new individuals,
#     continuously selecting the best performers for the next generation.
#     """
  
#     dataset_length = X.shape[1]
#     # initialize the global randome seed
#     if random_seed is not None:
#         random.seed(random_seed)  
#         np.random.seed(random_seed)
    
#     # not counting generation 0, generate 100 new generations
#     num_generations = num_generations+1
#     generations = [None] * num_generations
    
#     # init the first generation 'gen_0'  with random individuals
#     generations[0] = init_population(pop_size, X, y, feature_names, division_method, test_size, dataset_length, equation_length, fitness_method)

#     # we will use 50% of the best  individuals to reproduce for the next generation
#     best_features_index = round(pop_size/2)
    
 
#     # repeat the same process per each generation (use the top 50% of previous generation to get the best features, cross, mutate, and generate 30% of the new population from randoms)
#     for gen in range(1, num_generations):
#         # per each generation
#         new_gen =  []
#         best_individual = generations[gen-1][0]
        
#         if (best_individual["fitness_score"] > 0 ):
#             new_gen.append(best_individual)
#             new_gen_parents =  []
#             for individual in generations[gen-1][:best_features_index]:
#                 if (individual["fitness_score"] <= 0):
#                     break
#                 else:
#                     new_gen_parents.append(individual)
                  
#             if(len(new_gen_parents) > 1):
#                 cross_amount = len(new_gen_parents)
#                 selected_individuals = np.array([individual["features"] for individual in new_gen_parents])    
#             else:
#                 cross_amount = 0
                        
#                 # selected_individuals = np.array([individual["features"] for individual in new_gen_parents])
#         else:
#             cross_amount = 0
        
        
        
#         # amount of crossed individuals added so far
#         number_of_cross_added = 0
#         while(number_of_cross_added < cross_amount):
#             # print("cross_amount",cross_amount)
#             # print('select ', parent1, " select 2", parent2)
#             parent1 = selected_individuals[random.randint(0, len(selected_individuals) - 1)]
#             parent2 = selected_individuals[random.randint(0, len(selected_individuals) - 1)]
            
#             child1, child2 = crossover(parent1, parent2, cx_prob)
            
#             new_individual = mutation(child1, mut_prob)
#             new_individual2 = mutation(child2, mut_prob)
            
#             Child_1_equation, Child_1_R2, Child_1_Q2_LOO, Child_1_Avg_Rm2_LOO, Child_1_mae_training, Child_1_fitness_score, Child_1_prediction_quality_train, Child_1_Q2F1, Child_1_Q2F2, Child_1_avg_rm2_test, Child_1_mae_test, Child_1_prediction_quality_test = evaluate_individual(new_individual, X, y, feature_names, division_method, test_size, fitness_method)
#             Child_2_equation, Child_2_R2, Child_2_Q2_LOO, Child_2_Avg_Rm2_LOO, Child_2_mae_training, Child_2_fitness_score, Child_2_prediction_quality_train, Child_2_Q2F1, Child_2_Q2F2, Child_2_avg_rm2_test, Child_2_mae_test, Child_2_prediction_quality_test = evaluate_individual(new_individual2, X, y, feature_names, division_method, test_size, fitness_method)
#             # Child_1_fitness_score, Child_1_MAE, Child_1_Q2, Child_1_R2, Child_1_LOO_Mean, Child_1_LOO_Error, Child_1_Predictive_Quality = evaluate_individual(new_individual, X, y, division_method, test_size, fitness_method)
#             # Child_2_fitness_score, Child_2_MAE, Child_2_Q2, Child_2_R2, Child_2_LOO_Mean, Child_2_LOO_Error, Child_2_Predictive_Quality = evaluate_individual(new_individual2, X, y, division_method, test_size, fitness_method)
            
            

#             individual = {
#                 "features" : new_individual,
#                 "equation": Child_1_equation,
#                 "R2": Child_1_R2,
#                 "Q2_LOO": Child_1_Q2_LOO,
#                 "Avg_Rm2_LOO": Child_1_Avg_Rm2_LOO,
#                 "mae_training": Child_1_mae_training,
#                 "fitness_score": Child_1_fitness_score,
#                 "prediction_quality_train": Child_1_prediction_quality_train,
#                 "Q2F1": Child_1_Q2F1,
#                 "Q2F2": Child_1_Q2F2,
#                 "avg_rm2_test": Child_1_avg_rm2_test,
#                 "mae_test": Child_1_mae_test,
#                 "prediction_quality_test": Child_1_prediction_quality_test,
#             }
#             individual2 = {
#                 "features" : new_individual2,
#                 "equation": Child_2_equation,
#                 "R2": Child_2_R2,
#                 "Q2_LOO": Child_2_Q2_LOO,
#                 "Avg_Rm2_LOO": Child_2_Avg_Rm2_LOO,
#                 "mae_training": Child_2_mae_training,
#                 "fitness_score": Child_2_fitness_score,
#                 "prediction_quality_train": Child_2_prediction_quality_train,
#                 "Q2F1": Child_2_Q2F1,
#                 "Q2F2": Child_2_Q2F2,
#                 "avg_rm2_test": Child_2_avg_rm2_test,
#                 "mae_test": Child_2_mae_test,
#                 "prediction_quality_test": Child_2_prediction_quality_test,
#             }
            
#             new_gen.append(individual)
#             new_gen.append(individual2)
#             number_of_cross_added += 2
            
#         # generation of new blood per generation
#         while len(new_gen) < pop_size:
#             """
#             create remaining population of random new individual after the crossing over 
#             """
            
#             new_individual = init_individual(dataset_length, equation_length)
            
                
#             equation, R2, Q2_LOO, Avg_Rm2_LOO, mae_training, fitness_score, prediction_quality_train, Q2F1, Q2F2, avg_rm2_test, mae_test, prediction_quality_test = evaluate_individual(new_individual, X, y, feature_names, division_method, test_size, fitness_method)

#             individual = {
#                 "features" : new_individual,
#                 "equation": equation,
#                 "R2": R2,
#                 "Q2_LOO": Q2_LOO,
#                 "Avg_Rm2_LOO": Avg_Rm2_LOO,
#                 "mae_training": mae_training,
#                 "fitness_score": fitness_score,
#                 "prediction_quality_train": prediction_quality_train,
#                 "Q2F1": Q2F1,
#                 "Q2F2": Q2F2,
#                 "avg_rm2_test": avg_rm2_test,
#                 "mae_test": mae_test,
#                 "prediction_quality_test": prediction_quality_test,
#             }
                        
#             new_gen.append(individual)
            

#             if len(new_gen) >= pop_size:
#                 break
        
        
#         if  fitness_method == "LOO":
#             # q2 closer to 1 is better
#             new_gen.sort(key=lambda ind: ind['fitness_score'], reverse=True)
#         elif  fitness_method == "MAE":
#             # mae closer to 0 is better 
#             new_gen.sort(key=lambda ind: ind['fitness_score'], reverse=False)
       
            
    
            
#         # print( "fitness_score", fitness_score, "MAE", MAE, "Q2", Q2, "R2", R2, "LOO_Mean", LOO_Mean, "LOO_Error", LOO_Error, "Predictive_Quality", Predictive_Quality)
#         # print("new_ind_arr", new_ind_arr, "fitness_score", fitness_score, "MAE", MAE, "Q2", Q2, "R2", R2, "LOO_Mean", LOO_Mean, "LOO_Error", LOO_Error, "Predictive_Quality", Predictive_Quality)
        
#         print("best individual in gen", gen, ' was fitness_score ', new_gen[0]["fitness_score"],  "with MAE test", new_gen[0]["mae_test"], " and Q2_LOO", new_gen[0]["Q2_LOO"])
#         # print("best individual in gen", gen, ' was', new_gen[0]["score"],  "with wilks", new_gen[0]["wilks"], " and MCC", new_gen[0]["mcc"])
#         # print("best individual in gen", gen, ' was', new_gen[0]["score"])
            
#         generations[gen] = new_gen
        
    
    
#     # best_ind_arr = generations[num_generations-1][0]["features"]
#     # best_ind_score = generations[num_generations-1][0]["score"]
    
    
#     # print('len is: ',len(best_ind_arr))
    
#     # return best_ind_arr, best_ind_score
#     return generations[num_generations-1][0]
    


# def GA_MLR_main(X, y, feature_names, num_generations, pop_size, mut_prob, equation_length, division_method, test_size, fitness_method, cx_prob, random_seed):
    
            
#     # Split the dataset
#     # if division_method == "random division":
#     #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)
#     # elif division_method == "Kennard stone":
#     #     X_train, X_test, y_train, y_test = kennard_stone_sampling(X, y, test_size=test_size)
#     # elif division_method == "Euclidean":
#     #     X_train, X_test, y_train, y_test = euclidean_sampling(X, y, test_size=test_size)
#     # else:
#     #     st.error("Invalid division method.")
#     #     return       
            
#     # GA_test_size = test_size/(1-test_size)
#     # Run GA
    
#     best_ind = initialize_ga(X, y,feature_names, num_generations, pop_size, mut_prob, equation_length, division_method, test_size, fitness_method, cx_prob, random_seed)


#     features = best_ind["features"]
#     equation = best_ind["equation"]
#     R2 = best_ind["R2"]
#     Q2_LOO = best_ind["Q2_LOO"]
#     Avg_Rm2_LOO = best_ind["Avg_Rm2_LOO"]
#     mae_training = best_ind["mae_training"]
#     fitness_score = best_ind["fitness_score"]
#     prediction_quality_train = best_ind["prediction_quality_train"]
#     Q2F1 = best_ind["Q2F1"]
#     Q2F2 = best_ind["Q2F2"]
#     avg_rm2_test = best_ind["avg_rm2_test"]
#     mae_test = best_ind["mae_test"]
#     prediction_quality_test = best_ind["prediction_quality_test"]  
    
#     # best_ind_arr = best_ind["features"]
#     # training_score = best_ind["score"] 
#     # training_MAE = best_ind["MAE"] 
#     # training_Q2 = best_ind["Q2"] 
#     # training_R2 = best_ind["R2"] 
#     # training_LOO_Mean = best_ind["LOO_Mean"] 
#     # training_LOO_Error = best_ind["LOO_Error"] 
#     # training_Predictive_Quality = best_ind["Predictive_Quality"]    
    

                
#     # selected_features = [i for i, gene in enumerate(best_ind_arr) if gene == 1]
#     # X_train_best_features = X_train[:, selected_features]
#     # X_test_best_features = X_test[:, selected_features]
                
#                 # retrain the model on the original train  set with the selected features
#     # model = LinearRegression()
#     # lda = LinearDiscriminantAnalysis()
#     # lda.fit(X_train_best_features, y_train)
#     # model.fit(X_train_best_features, y_train)
                
#     # y_pred = model.predict(X_test_best_features)
     
#     # test_MAE, test_Q2, test_R2, test_LOO_Mean, test_LOO_Error, test_Predictive_Quality = leave_one_out_cross_validation(X_train_best_features, y_train, y)

#     # if  fitness_method == "LOO":
#     #     if(test_Q2 < 0 or test_Q2 > 1):
#     #         test_fitness_score = 0
#     #     else:
#     #         test_fitness_score = test_Q2

#     # elif fitness_method == "MAE":

#     #     if(test_MAE < 0 or test_MAE > 1 ):
#     #         test_fitness_score = 100
#     #     else:
#     #         test_fitness_score = test_MAE
            
    

#     # Compute ROC curve and AUC
#     # fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test_best_features)[:, 1])
#     # roc_auc = auc(fpr, tpr)

#     # Display results using Streamlit
#     # st.title('MLR Model Evaluation')
#     # st.subheader('Training Model Metrics')
    
#     # st.text(f'Training score: {training_score}')
#     # st.text(f'Training MAE: {training_MAE}')
#     # st.text(f'Training Q2: {training_Q2}')
#     # st.text(f'Training R2: {training_R2}')
#     # st.text(f'Training LOO_Mean: {training_LOO_Mean}')
#     # st.text(f'Training LOO_Error: {training_LOO_Error}')
#     # st.text(f'Training score: {training_score:.2f}')
#     # st.text(f'Training MAE (In percent): {training_MAE:.2f}')
#     # st.text(f'Training Q2: {training_Q2:.2f}')
#     # st.text(f'Training R2: {training_R2:.2f}')
#     # st.text(f'Training LOO_Mean: {training_LOO_Mean:.2f}')
#     # st.text(f'Training LOO_Error: {training_LOO_Error:.2f}')
#     # st.text(f'Training Predictive_Quality: {training_Predictive_Quality}')

#     # st.subheader('Test Model Metrics')
    
#     # st.text(f'Test score: {test_fitness_score}')
#     # st.text(f'Test MAE: {test_MAE}')
#     # st.text(f'Test Q2: {test_Q2}')
#     # st.text(f'Test R2: {test_R2}')
#     # st.text(f'Test LOO_Mean: {test_LOO_Mean}')
#     # st.text(f'Test LOO_Error: {test_LOO_Error}')
#     # st.text(f'Test score: {test_fitness_score:.2f}')
#     # st.text(f'Test MAE (In percent): {test_MAE:.2f}')
#     # st.text(f'Test Q2: {test_Q2:.2f}')
#     # st.text(f'Test R2: {test_R2:.2f}')
#     # st.text(f'Test LOO_Mean: {test_LOO_Mean:.2f}')
#     # st.text(f'Test LOO_Error: {test_LOO_Error:.2f}')
#     # st.text(f'Test Predictive_Quality: {test_Predictive_Quality}')
    
#     # graphs
#     # Generate predictions
    
    
#     # Calculate coefficients and equation
#     # coefficients = model.coef_
#     # intercept = model.intercept_
#     # standard_errors = sem(X_train_best_features, axis=0)
#     # selected_feature_names = [feature_names[i] for i in selected_features]

#     # equation_parts = [
#     #     f"{coeff:.3f} (+/- {se:.3f}) * {name}" for coeff, se, name in zip(coefficients, standard_errors, selected_feature_names)
#     # ]
#     # equation = " + ".join(equation_parts)
#     # final_equation = f"Score = {intercept:.3f} + {equation}"

#     # Display final equation
#     # st.write("Final Equation:")
#     # st.write(final_equation)

#     # find the best graphs
#     # residuals = y_test - y_pred

#     # # Plotting the residuals
#     # plt.figure(figsize=(10, 6))
#     # sns.residplot(x=y_pred, y=residuals, lowess=True, line_kws={'color': 'red', 'lw': 1})
#     # plt.title('Residual Plot')
#     # plt.xlabel('Predicted Values')
#     # plt.ylabel('Residuals')
#     # plt.axhline(y=0, color='blue', linestyle='--')
#     # plt.grid(True)
#     # # plt.show()

#     # st.pyplot(plt)
    
    
    
    
#     # Display the Q-Q Plot for residuals
#     # plot_qq(residuals)

    
    


#     # # ROC Curve
#     # st.subheader('ROC Curve')
#     # plt.figure()
#     # plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
#     # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#     # plt.xlim([0.0, 1.0])
#     # plt.ylim([0.0, 1.05])
#     # plt.xlabel('False Positive Rate')
#     # plt.ylabel('True Positive Rate')
#     # plt.title('Receiver Operating Characteristic')
#     # plt.legend(loc="lower right")
#     # st.pyplot(plt)

#     # # Confusion Matrix
#     # conf_m = confusion_matrix(y_test, y_pred)
#     # fig, ax = plt.subplots()
#     # sns.heatmap(conf_m, annot=True, fmt="d", cmap="Blues", cbar=False, square=True, ax=ax)
#     # ax.set_xlabel("Predicted")
#     # ax.set_ylabel("True")
#     # ax.set_title("Confusion Matrix")
#     # st.subheader('Confusion Matrix')
#     # st.pyplot(fig)

#     # # Selected features
#     # selected_feature_names = [feature_names[i] for i in selected_features]
#     # st.write("Selected Features:", selected_feature_names)

#     # # Final equation
#     # coefficients = lda.coef_[0]
#     # intercept = lda.intercept_[0]
#     # standard_errors = sem(X_train_best_features, axis=0)

#     # equation_parts = [
#     #     f"{coeff:.3f} (+/- {se:.3f}) * {name}" for coeff, se, name in zip(coefficients, standard_errors, selected_feature_names)
#     # ]
#     # equation = " + ".join(equation_parts)
#     # final_equation = f"Score = {intercept:.3f} + {equation}"

#     # st.write("Final Equation:")
#     # st.write(final_equation)

import numpy as np
# from deap import creator, base, tools, algorithms
import random
# import pygad
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# import pygad
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
# from deap import base, creator, tools, algorithms
import random
from division_methods import kennard_stone_sampling, euclidean_sampling
from copy import deepcopy
from sklearn.preprocessing import LabelEncoder
import numpy as np
import random
from sklearn.metrics import matthews_corrcoef
from fitness_function import  leave_one_out_cross_validation
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc)
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc)
import seaborn as sns
import matplotlib.pyplot as plt
from division_methods import kennard_stone_sampling, euclidean_sampling
# from Classification_GA import initialize_ga
# from fitness_function import compute_wilks_lambda
import random
from sklearn.metrics import matthews_corrcoef
from scipy.stats import sem

from sklearn.linear_model import LinearRegression

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
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

    
# init individuals functions
def init_individual(dataset_length, equation_length):
    """
    dataset_length = amount of features in the dataset

    num_ones = amount of active features per individual
    """
    # Check if the number of 1s is greater than the array length
    if equation_length > dataset_length:
        raise ValueError("Number of 1s cannot exceed the length of the array.")
    
    # Create an array of zeros with the specified length
    individual = np.zeros(dataset_length, dtype=int)
    
    # Randomly choose positions to place 1s
    ones_indices = np.random.choice(dataset_length, equation_length, replace=False)
    
    # Set those positions to 1
    individual[ones_indices] = 1
    
    return individual
   
   
   
# Generate generation 0 initialization of the first generation
def init_population(pop_size, X, y, feature_names, division_method, test_size, dataset_length, equation_length, fitness_method):
    initial_population =  []
    for i in range(pop_size):
        new_individual = init_individual(dataset_length, equation_length)
        
        # equation, R2, Q2_LOO, Avg_Rm2_LOO, mae_training, fitness_score, prediction_quality_train, Q2F1, Q2F2, avg_rm2_test, mae_test, prediction_quality_test = evaluate_individual(new_individual, X, y, feature_names, division_method, test_size, fitness_method)
        individual = evaluate_individual(new_individual, X, y, feature_names, division_method, test_size, fitness_method)

        # individual = {
        #     "features" : new_individual,
        #     "equation": equation,
        #     "R2": R2,
        #     "Q2_LOO": Q2_LOO,
        #     "Avg_Rm2_LOO": Avg_Rm2_LOO,
        #     "mae_training": mae_training,
        #     "fitness_score": fitness_score,
        #     "prediction_quality_train": prediction_quality_train,
        #     "Q2F1": Q2F1,
        #     "Q2F2": Q2F2,
        #     "avg_rm2_test": avg_rm2_test,
        #     "mae_test": mae_test,
        #     "prediction_quality_test": prediction_quality_test,
        # }
        
        
        initial_population.append(individual)
        
        # //
        # initial_population.append(individual)
        if  fitness_method == "LOO":
            # q2 closer to 1 is better
            initial_population.sort(key=lambda ind: ind['fitness_score'], reverse=True)
        elif  fitness_method == "MAE":
            # mae closer to 0 the better 
            initial_population.sort(key=lambda ind: ind['fitness_score'], reverse=False)
        

    return initial_population

def crossover(ind1, ind2, cx_prob):
    random_value = random.random()
    
    ind1 = pd.Series(ind1)
    ind2 = pd.Series(ind2)
    
    if(random_value < cx_prob):

        ind1_copy = ind1[:]
        ind2_copy = ind2[:]

        ind1_idx_list = ind1_copy[ind1_copy == 1].index
        ind2_idx_list = ind2_copy[ind2_copy == 1].index
        # Randomly pick one of these indices

        ind1_idx = random.choice(ind1_idx_list) if not ind1_idx_list.empty else None
        ind2_idx = random.choice(ind2_idx_list) if not ind2_idx_list.empty else None

        if ind1_idx == ind2_idx and len(ind1_idx_list) > 0:
            ind1_idx = random.choice(ind1_idx_list)


        if ind1_idx is not None and ind2_idx is not None:
            if ind2_copy[ind1_idx] == 0:
                ind2_copy[ind1_idx] = 1
                ind2_copy[ind2_idx] = 0

            if ind1_copy[ind2_idx] == 0:
                ind1_copy[ind2_idx] = 1
                ind1_copy[ind1_idx] = 0
                        
        return ind1_copy, ind2_copy
    else:
        return ind1, ind2

# mutate crossed childs to integrate new changes and posivilities
def mutation(ind, mut_prob):
    
    random_value = random.random()
    # print(random_value)
    if(random_value < mut_prob):
    
        ind_copy = ind[:]

        # one_idx_list = ind_copy[ind_copy == 1].index
        one_idx_list = ind_copy[ind_copy == 1].index

        # zero_idx_list = ind_copy[ind_copy == 0].index
        zero_idx_list = ind_copy[ind_copy == 0].index


        flip_one_to_zero = random.choice(one_idx_list) if not one_idx_list.empty else None
        flip_zero_to_one = random.choice(zero_idx_list) if not one_idx_list.empty else None

        ind_copy[flip_one_to_zero]=0
        ind_copy[flip_zero_to_one]=1
    

        return ind_copy
    else :
        return ind

# evaluate each individual in the population and give them a fitness score
# def evaluate_individual(individual, X, y, division_method, test_size, fitness_method):
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
        
        
        
#     MAE, Q2, R2, LOO_Mean, LOO_Error, Predictive_Quality = leave_one_out_cross_validation(X_train, y_train, y)


    
#     # return the selected fitness score
#     if  fitness_method == "LOO":
#         if(Q2 < 0 or Q2 > 1):
#             fitness_score = 0
#             return fitness_score, MAE, Q2, R2, LOO_Mean, LOO_Error, Predictive_Quality
#         else:
#             fitness_score = Q2
#             return fitness_score, MAE, Q2, R2, LOO_Mean, LOO_Error, Predictive_Quality
        
#     elif fitness_method == "MAE":
            
#         fitness_score = MAE
#         return fitness_score, MAE, Q2, R2, LOO_Mean, LOO_Error, Predictive_Quality

#         # if(MAE < 0 or MAE > 1 ):
#         #     fitness_score = 1
#         #     return fitness_score, MAE, Q2, R2, LOO_Mean, LOO_Error, Predictive_Quality
#         # else:
#         #     fitness_score = MAE
#         #     return fitness_score, MAE, Q2, R2, LOO_Mean, LOO_Error, Predictive_Quality

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
 
    # hat_matrix = X_train @ np.linalg.inv(X_train.T @ X_train) @ X_train.T
    # leverages = np.diagonal(hat_matrix)       
    
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
        "features" : individual,
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
        "y_train": y_train,
        "y_test": y_test,

    }
    
    return  individual

 
def initialize_ga(X, y, feature_names, num_generations, pop_size, mut_prob, equation_length, division_method, test_size, fitness_method, cx_prob, random_seed):
    """
    Initializes and evolves a population for a genetic algorithm.

    Parameters:
    X : array-like, feature dataset.
    y : array-like, target dataset.
    dataset_length : int, length of an individual's chromosome.
    num_generations : int, number of generations to evolve.
    pop_size : int, number of individuals in each generation.
    mut_prob : float, probability of mutation.
    equation_length : int, length of the equation/problem solution.
    division_method : function, method to split dataset for training/testing.
    test_size : float, proportion of dataset to be used as test set.
    cx_prob : float, probability of crossover.

    Evolves a population through selection, crossover, mutation, and introduces new individuals,
    continuously selecting the best performers for the next generation.
    """
  
    dataset_length = X.shape[1]
    # initialize the global randome seed
    if random_seed is not None:
        random.seed(random_seed)  
        np.random.seed(random_seed)
    
    # not counting generation 0, generate 100 new generations
    num_generations = num_generations+1
    generations = [None] * num_generations
    
    # init the first generation 'gen_0'  with random individuals
    generations[0] = init_population(pop_size, X, y, feature_names, division_method, test_size, dataset_length, equation_length, fitness_method)

    # we will use 50% of the best  individuals to reproduce for the next generation
    best_features_index = round(pop_size/2)
    
 
    # repeat the same process per each generation (use the top 50% of previous generation to get the best features, cross, mutate, and generate 30% of the new population from randoms)
    for gen in range(1, num_generations):
        # per each generation
        new_gen =  []
        best_individual = generations[gen-1][0]
        
        if (best_individual["fitness_score"] > 0 ):
            new_gen.append(best_individual)
            new_gen_parents =  []
            for individual in generations[gen-1][:best_features_index]:
                if (individual["fitness_score"] <= 0):
                    break
                else:
                    new_gen_parents.append(individual)
                  
            if(len(new_gen_parents) > 1):
                cross_amount = len(new_gen_parents)
                selected_individuals = np.array([individual["features"] for individual in new_gen_parents])    
            else:
                cross_amount = 0
                        
        else:
            cross_amount = 0
        
        
        
        # amount of crossed individuals added so far
        number_of_cross_added = 0
        while(number_of_cross_added < cross_amount):
            
            parent1 = selected_individuals[random.randint(0, len(selected_individuals) - 1)]
            parent2 = selected_individuals[random.randint(0, len(selected_individuals) - 1)]
            
            child1, child2 = crossover(parent1, parent2, cx_prob)
            
            new_individual = mutation(child1, mut_prob)
            new_individual2 = mutation(child2, mut_prob)
            
            individual = evaluate_individual(new_individual, X, y, feature_names, division_method, test_size, fitness_method)
            individual2 = evaluate_individual(new_individual2, X, y, feature_names, division_method, test_size, fitness_method)

            
            new_gen.append(individual)
            new_gen.append(individual2)
            number_of_cross_added += 2
            
        # generation of new blood per generation
        while len(new_gen) < pop_size:
            """
            create remaining population of random new individual after the crossing over 
            """
            
            new_individual = init_individual(dataset_length, equation_length)
            
                
            # equation, R2, Q2_LOO, Avg_Rm2_LOO, mae_training, fitness_score, prediction_quality_train, Q2F1, Q2F2, avg_rm2_test, mae_test, prediction_quality_test = evaluate_individual(new_individual, X, y, feature_names, division_method, test_size, fitness_method)
            individual = evaluate_individual(new_individual, X, y, feature_names, division_method, test_size, fitness_method)
                        
            new_gen.append(individual)
            

            if len(new_gen) >= pop_size:
                break
        
        
        if  fitness_method == "LOO":
            # q2 closer to 1 is better
            new_gen.sort(key=lambda ind: ind['fitness_score'], reverse=True)
        elif  fitness_method == "MAE":
            # mae closer to 0 is better 
            new_gen.sort(key=lambda ind: ind['fitness_score'], reverse=False)
       
            
    
      
        print("best individual in gen", gen, ' was fitness_score ', new_gen[0]["fitness_score"],  "with MAE test", new_gen[0]["mae_test"], " and Q2_LOO", new_gen[0]["Q2_LOO"])
          
        generations[gen] = new_gen
        
    
    return generations[num_generations-1][0]
    


def GA_MLR_main(X, y, feature_names, num_generations, pop_size, mut_prob, equation_length, division_method, test_size, fitness_method, cx_prob, random_seed):


    
    best_individual = initialize_ga(X, y,feature_names, num_generations, pop_size, mut_prob, equation_length, division_method, test_size, fitness_method, cx_prob, random_seed)
    
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


    
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=leverages_test, y=standardized_residuals_test)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Leverage')
    plt.ylabel('Standardized Residuals')
    plt.title('Williams Plot (test Data)')
    st.subheader('Williams Plot (test Data)')
    st.pyplot(plt)
    
    
    plt.clf()

    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=leverages_train, y=standardized_residuals_train)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Leverage')
    plt.ylabel('Standardized Residuals')
    plt.title('Williams Plot (training Data)')
    st.subheader('Williams Plot (training Data)')
    st.pyplot(plt)
    
    # Scatter plot of observed vs predicted
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred_test)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Observed')
    plt.ylabel('Predicted')
    plt.title('Scatter Plot of Observed vs Predicted (Test Data)')
    st.subheader('Scatter Plot of Observed vs Predicted (Test Data)')
    st.pyplot(plt) 
    
    plt.clf()

    # Scatter plot of observed vs predicted
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_train, y=y_pred_train)
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
    plt.xlabel('Observed')
    plt.ylabel('Predicted')
    plt.title('Scatter Plot of Observed vs Predicted (Training Data)')
    st.subheader('Scatter Plot of Observed vs Predicted (Training Data)')
    st.pyplot(plt)
    
    # print('y_pred_train', len(y_pred_train), 'y_train', len(y_train))
    
    # st.write("Best Individuals")
    # st.write(df_individuals.head())
    
    # features = best_ind["features"]
    # equation = best_ind["equation"]
    # R2 = best_ind["R2"]
    # Q2_LOO = best_ind["Q2_LOO"]
    # Avg_Rm2_LOO = best_ind["Avg_Rm2_LOO"]
    # mae_training = best_ind["mae_training"]
    # fitness_score = best_ind["fitness_score"]
    # prediction_quality_train = best_ind["prediction_quality_train"]
    # Q2F1 = best_ind["Q2F1"]
    # Q2F2 = best_ind["Q2F2"]
    # avg_rm2_test = best_ind["avg_rm2_test"]
    # mae_test = best_ind["mae_test"]
    # prediction_quality_test = best_ind["prediction_quality_test"]  
    
    # best_ind_arr = best_ind["features"]
    # training_score = best_ind["score"] 
    # training_MAE = best_ind["MAE"] 
    # training_Q2 = best_ind["Q2"] 
    # training_R2 = best_ind["R2"] 
    # training_LOO_Mean = best_ind["LOO_Mean"] 
    # training_LOO_Error = best_ind["LOO_Error"] 
    # training_Predictive_Quality = best_ind["Predictive_Quality"]    
    

                
    # selected_features = [i for i, gene in enumerate(best_ind_arr) if gene == 1]
    # X_train_best_features = X_train[:, selected_features]
    # X_test_best_features = X_test[:, selected_features]
                
                # retrain the model on the original train  set with the selected features
    # model = LinearRegression()
    # lda = LinearDiscriminantAnalysis()
    # lda.fit(X_train_best_features, y_train)
    # model.fit(X_train_best_features, y_train)
                
    # y_pred = model.predict(X_test_best_features)
     
    # test_MAE, test_Q2, test_R2, test_LOO_Mean, test_LOO_Error, test_Predictive_Quality = leave_one_out_cross_validation(X_train_best_features, y_train, y)

    # if  fitness_method == "LOO":
    #     if(test_Q2 < 0 or test_Q2 > 1):
    #         test_fitness_score = 0
    #     else:
    #         test_fitness_score = test_Q2

    # elif fitness_method == "MAE":

    #     if(test_MAE < 0 or test_MAE > 1 ):
    #         test_fitness_score = 100
    #     else:
    #         test_fitness_score = test_MAE
            
    

    # Compute ROC curve and AUC
    # fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test_best_features)[:, 1])
    # roc_auc = auc(fpr, tpr)

    # Display results using Streamlit
    # st.title('MLR Model Evaluation')
    # st.subheader('Training Model Metrics')
    
    # st.text(f'Training score: {training_score}')
    # st.text(f'Training MAE: {training_MAE}')
    # st.text(f'Training Q2: {training_Q2}')
    # st.text(f'Training R2: {training_R2}')
    # st.text(f'Training LOO_Mean: {training_LOO_Mean}')
    # st.text(f'Training LOO_Error: {training_LOO_Error}')
    # st.text(f'Training score: {training_score:.2f}')
    # st.text(f'Training MAE (In percent): {training_MAE:.2f}')
    # st.text(f'Training Q2: {training_Q2:.2f}')
    # st.text(f'Training R2: {training_R2:.2f}')
    # st.text(f'Training LOO_Mean: {training_LOO_Mean:.2f}')
    # st.text(f'Training LOO_Error: {training_LOO_Error:.2f}')
    # st.text(f'Training Predictive_Quality: {training_Predictive_Quality}')

    # st.subheader('Test Model Metrics')
    
    # st.text(f'Test score: {test_fitness_score}')
    # st.text(f'Test MAE: {test_MAE}')
    # st.text(f'Test Q2: {test_Q2}')
    # st.text(f'Test R2: {test_R2}')
    # st.text(f'Test LOO_Mean: {test_LOO_Mean}')
    # st.text(f'Test LOO_Error: {test_LOO_Error}')
    # st.text(f'Test score: {test_fitness_score:.2f}')
    # st.text(f'Test MAE (In percent): {test_MAE:.2f}')
    # st.text(f'Test Q2: {test_Q2:.2f}')
    # st.text(f'Test R2: {test_R2:.2f}')
    # st.text(f'Test LOO_Mean: {test_LOO_Mean:.2f}')
    # st.text(f'Test LOO_Error: {test_LOO_Error:.2f}')
    # st.text(f'Test Predictive_Quality: {test_Predictive_Quality}')
    
    # graphs
    # Generate predictions
    
    
    # Calculate coefficients and equation
    # coefficients = model.coef_
    # intercept = model.intercept_
    # standard_errors = sem(X_train_best_features, axis=0)
    # selected_feature_names = [feature_names[i] for i in selected_features]

    # equation_parts = [
    #     f"{coeff:.3f} (+/- {se:.3f}) * {name}" for coeff, se, name in zip(coefficients, standard_errors, selected_feature_names)
    # ]
    # equation = " + ".join(equation_parts)
    # final_equation = f"Score = {intercept:.3f} + {equation}"

    # Display final equation
    # st.write("Final Equation:")
    # st.write(final_equation)

    # find the best graphs
    # residuals = y_test - y_pred

    # # Plotting the residuals
    # plt.figure(figsize=(10, 6))
    # sns.residplot(x=y_pred, y=residuals, lowess=True, line_kws={'color': 'red', 'lw': 1})
    # plt.title('Residual Plot')
    # plt.xlabel('Predicted Values')
    # plt.ylabel('Residuals')
    # plt.axhline(y=0, color='blue', linestyle='--')
    # plt.grid(True)
    # # plt.show()

    # st.pyplot(plt)
    
    
    
    
    # Display the Q-Q Plot for residuals
    # plot_qq(residuals)

    
    


    # # ROC Curve
    # st.subheader('ROC Curve')
    # plt.figure()
    # plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic')
    # plt.legend(loc="lower right")
    # st.pyplot(plt)

    # # Confusion Matrix
    # conf_m = confusion_matrix(y_test, y_pred)
    # fig, ax = plt.subplots()
    # sns.heatmap(conf_m, annot=True, fmt="d", cmap="Blues", cbar=False, square=True, ax=ax)
    # ax.set_xlabel("Predicted")
    # ax.set_ylabel("True")
    # ax.set_title("Confusion Matrix")
    # st.subheader('Confusion Matrix')
    # st.pyplot(fig)

    # # Selected features
    # selected_feature_names = [feature_names[i] for i in selected_features]
    # st.write("Selected Features:", selected_feature_names)

    # # Final equation
    # coefficients = lda.coef_[0]
    # intercept = lda.intercept_[0]
    # standard_errors = sem(X_train_best_features, axis=0)

    # equation_parts = [
    #     f"{coeff:.3f} (+/- {se:.3f}) * {name}" for coeff, se, name in zip(coefficients, standard_errors, selected_feature_names)
    # ]
    # equation = " + ".join(equation_parts)
    # final_equation = f"Score = {intercept:.3f} + {equation}"

    # st.write("Final Equation:")
    # st.write(final_equation)