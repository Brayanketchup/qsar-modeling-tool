import numpy as np
import random
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from division_methods import kennard_stone_sampling, euclidean_sampling
from sklearn.metrics import matthews_corrcoef
# from fitness_function import compute_wilks_lambda, compute_wilks_lambdaa, calculate_wilks_lambda

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc)


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
    
    individual[ones_indices] = 1
    
    return individual
   
   
   
# Generate generation 0 initialization of the first generation
def init_population(pop_size, X, y, feature_names, division_method, test_size, dataset_length, equation_length):
    initial_population =  []
    for i in range(pop_size):
        new_ind = init_individual(dataset_length, equation_length)

        individual = evaluate_individual(new_ind, X, y, feature_names, division_method, test_size)

        initial_population.append(individual)
        
       
        
    # if  fitness_method == "Wilks Lambda":
        # initial_population.sort(key=lambda ind: ind['score'], reverse=False)
        # 
    # elif fitness_method == "MCC + Wilks Lambda":
        # initial_population.sort(key=lambda ind: ind['mcc_test'], reverse=True)
    initial_population.sort(key=lambda ind: ind['mcc_train'], reverse=True)

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
def evaluate_individual(individual, X, y, feature_names, division_method, test_size):
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
    

    
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    
    selected_feature_names = [feature_names[i] for i in selected_features]
    
    coefficients = lda.coef_[0]
    # Calculate standard errors
    y_pred = lda.predict(X_train)
    residuals = y_train - y_pred
    residual_var = np.var(residuals, ddof=len(coefficients))
    equation_parts = []
    
    try:
        matrix_to_invert = np.dot(X_train.T, X_train)
        inv_hessian = np.linalg.pinv(matrix_to_invert)
        # inv_hessian = np.linalg.inv(matrix_to_invert)
        standard_errors = np.sqrt(np.diagonal(residual_var * inv_hessian))
        equation_parts = [
            f"{coeff:.3f} (+/- {se:.3f}) * {name}" for coeff, se, name in zip(coefficients, standard_errors, selected_feature_names)
        ]
        
    except np.linalg.LinAlgError as e:
        print("Failed to invert matrix due to singularity. Here is the matrix being inverted:")
        # print(matrix_to_invert)
        print("Error message:", e)
        

    
    final_equation = " + ".join(equation_parts)
    
    
    fpr, tpr, thresholds = roc_curve(y_test, lda.predict_proba(X_test)[:, 1])
    
    # print('printing')
    # print('thresholds', thresholds)

    # //predict for test
    y_pred_test = lda.predict(X_test)
    
    # predict for train
    y_pred_train = lda.predict(X_train)
    
    
    # wilks_train = calculate_wilks_lambda(X_train, y_train, y)
    # wilks_train = calculate_wilks_lambda(X_train, y_train)
    
    # wilks_test = calculate_wilks_lambda(X_test, y_test)
    # wilks_test = calculate_wilks_lambda(X_test, y_test)
    
    # print('selected_feature_names', selected_feature_names)
    # print('final_equation', final_equation)
    # print('wilks_test', wilks_test)
    
    
    selected_feature_names = [feature_names[i] for i in selected_features]
    
    # Call to compute_wilks_lambda function
    # Assuming 'data' should be a DataFrame containing the training data and the group labels
    # train_data = pd.DataFrame(X_train, columns=selected_feature_names)
    # train_data['Group'] = y_train  # Add the target variable as 'Group'
    
    accuracy_test = accuracy_score(y_test, y_pred_test)
    accuracy_train = accuracy_score(y_train, y_pred_train)
    
    accuracy_percentage_test = accuracy_test * 100
    accuracy_percentage_train = accuracy_train * 100
    
    precision_percentage_test = precision_score(y_test, y_pred_test, average='binary', zero_division=0) * 100
    precision_percentage_train = precision_score(y_train, y_pred_train, average='binary', zero_division=0) * 100
    
    recall_sensitivity_percentage_test = recall_score(y_test, y_pred_test, average='binary') * 100
    recall_sensitivity_percentage_train = recall_score(y_train, y_pred_train, average='binary') * 100

    tn_test, fp_test, fn_test, tp_test = confusion_matrix(y_test, y_pred_test).ravel()
    tn_train, fp_train, fn_train, tp_train = confusion_matrix(y_train, y_pred_train).ravel()

    specificity_percentage_test = (tn_test / (tn_test + fp_test)) * 100
    specificity_percentage_train = (tn_train / (tn_train + fp_train)) * 100

    f1_test = f1_score(y_test, y_pred_test, average='binary')
    f1_train = f1_score(y_train, y_pred_train, average='binary')
    
  
    
    # my own implementation of mcc
    numerator_test = (tp_test * tn_test) - (fp_test * fn_test)
    denominator_test = np.sqrt((tp_test + fp_test) * (tp_test + fn_test) * (tn_test + fp_test) * (tn_test + fn_test))
    mcc_test = numerator_test / denominator_test if denominator_test != 0 else 0
    
    mcc_test2 = matthews_corrcoef(y_test, y_pred_test)
    mcc_train2 = matthews_corrcoef(y_train, y_pred_train)
    
    # /////////////////// mcc calculations //////////////////
    numerator_train = (tp_train * tn_train) - (fp_train * fn_train)
    denominator_train = np.sqrt((tp_train + fp_train) * (tp_train + fn_train) * (tn_train + fp_train) * (tn_train + fn_train))
    mcc_train = numerator_train / denominator_train if denominator_train != 0 else 0
    
            
    
    individual = {
    'features': individual,
    'y_pred_test': y_pred_test,
    'y_pred_train': y_pred_train,
    'accuracy_percentage_test': accuracy_percentage_test,
    'accuracy_percentage_train': accuracy_percentage_train,
    'precision_percentage_test': precision_percentage_test,
    'precision_percentage_train': precision_percentage_train,
    'recall_sensitivity_percentage_test': recall_sensitivity_percentage_test,
    'recall_sensitivity_percentage_train': recall_sensitivity_percentage_train,
    'tn_test': tn_test, 
    'fp_test': fp_test, 
    'fn_test': fn_test, 
    'tp_test': tp_test,
    'tn_train': tn_train, 
    'fp_train': fp_train, 
    'fn_train': fn_train, 
    'tp_train': tp_train,
    'specificity_percentage_test': specificity_percentage_test,
    'specificity_percentage_train': specificity_percentage_train,
    'f1_test': f1_test,
    'f1_train': f1_train,
    'mcc_test': mcc_test,
    'mcc_train': mcc_train,
    'final_equation': final_equation,
    'fpr': fpr,
    'tpr': tpr,
    'X_train' : X_train, 
    'X_test' : X_test, 
    'y_train' : y_train, 
    'y_test' : y_test,
    'thresholds': thresholds,
    }
    
    return individual
    

def initialize_ga(X, y, num_generations, feature_names, pop_size, mut_prob, equation_length, division_method, test_size, cx_prob, random_seed):
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
    generations[0] = init_population(pop_size, X, y, feature_names, division_method, test_size, dataset_length, equation_length)

    # we will use 50% of the best  individuals to reproduce for the next generation
    best_features_index = round(pop_size/2)
    
 
    # repeat the same process per each generation (use the top 50% of previous generation to get the best features, cross, mutate, and generate 30% of the new population from randoms)
    for gen in range(1, num_generations):
        # per each generation
        new_gen =  []
        best_individual = generations[gen-1][0]
                    
        new_gen.append(best_individual)
        new_gen_parents =  []
        
        # print("this is the lengthaaasdasd: ", len(new_gen_parents))
        for individual in generations[gen-1][:best_features_index]:
                new_gen_parents.append(individual)
                
        if(len(new_gen_parents) > 1):
            cross_amount = len(new_gen_parents)
            # print('cross_amount should be greater than 1', cross_amount)
            selected_individuals = np.array([individual["features"] for individual in new_gen_parents])
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
            
            individual = evaluate_individual(new_individual, X, y, feature_names, division_method, test_size)
            individual2 = evaluate_individual(new_individual2, X, y, feature_names, division_method, test_size)
            
            new_gen.append(individual)
            new_gen.append(individual2)
            number_of_cross_added += 2
            
        # generation of new blood per generation
        while len(new_gen) < pop_size:
            """
            create remaining population of random new individual after the crossing over 
            """
            
            new_ind_arr = init_individual(dataset_length, equation_length)
            individual = evaluate_individual(new_ind_arr, X, y, feature_names, division_method, test_size)

            new_gen.append(individual)
            

            if len(new_gen) >= pop_size:
                break
        
        new_gen.sort(key=lambda ind: ind["mcc_train"], reverse=True)
        # if  fitness_method == "Wilks Lambda":
        #     new_gen.sort(key=lambda ind: ind['score'], reverse=False)
            
        # elif fitness_method == "MCC + Wilks Lambda":
        #     new_gen.sort(key=lambda ind: ind['score'], reverse=True)
        #     if(new_gen[0]["score"] == 0):
        #         new_gen.sort(key=lambda ind: ind["mcc_test"], reverse=True)
            
        generations[gen] = new_gen  
    
    return generations[num_generations-1][0]
    


def GA_LDA_main(X, y, feature_names, num_generations, pop_size, mut_prob, equation_length, division_method, test_size, cx_prob, random_seed):
    
    
    best_ind = initialize_ga(X, y, num_generations, feature_names, pop_size, mut_prob, equation_length, division_method, test_size, cx_prob, random_seed)
    
    # if fitness_method == "Wilks Lambda" and best_ind['score'] == 1 or fitness_method == "MCC + Wilks Lambda" and best_ind['score'] == 0:
    # if best_ind['score or fitness_method == "MCC + Wilks Lambda" and best_ind['score'] == 0:
        # st.error('No valid models were found with acceptable performance. Please check your data or model parameters.')      
    # else:    
   
    features = best_ind['features']
    y_pred_test = best_ind['y_pred_test']
    y_pred_train = best_ind['y_pred_train']
    accuracy_percentage_test = best_ind['accuracy_percentage_test']
    accuracy_percentage_train = best_ind['accuracy_percentage_train']
    precision_percentage_test = best_ind['precision_percentage_test']
    precision_percentage_train = best_ind['precision_percentage_train']
    recall_sensitivity_percentage_test = best_ind['recall_sensitivity_percentage_test']
    recall_sensitivity_percentage_train = best_ind['recall_sensitivity_percentage_train']
    tn_test = best_ind['tn_test']
    fp_test = best_ind['fp_test']
    fn_test = best_ind['fn_test']
    tp_test = best_ind['tp_test']
    tn_train = best_ind['tn_train']
    fp_train = best_ind['fp_train']
    fn_train = best_ind['fn_train']
    tp_train = best_ind['tp_train']
    specificity_percentage_test = best_ind['specificity_percentage_test']
    specificity_percentage_train = best_ind['specificity_percentage_train']
    f1_test = best_ind['f1_test']
    f1_train = best_ind['f1_train']
    mcc_test = best_ind['mcc_test']
    mcc_train = best_ind['mcc_train']
    final_equation = best_ind['final_equation']
    fpr = best_ind['fpr']
    tpr = best_ind['tpr']
    X_train = best_ind['X_train']
    X_test = best_ind['X_test']
    y_train = best_ind['y_train']
    y_test = best_ind['y_test']
    thresholds = best_ind['thresholds']
    
    
    
    
    # roc_auc = auc(fpr, tpr)
    
    conf_m = [[tn_test, fp_test],[fn_test, tp_test]]
    
    mcc_test_my_calc = ((tp_test * tn_test) - (fp_test * fn_test)) / np.sqrt((tp_test + fp_test) * (tp_test + fn_test) * (tn_test + fp_test) * (tn_test + fn_test))
    
    TPR = tp_test / (tp_test + fn_test)
    FPR = fp_test / (fp_test + tn_test)
    
    
    
    
    st.title('Model Performance Overview')
    st.write('### Final Equation')
    st.write(final_equation)
    
    

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Training Performance")
        st.write(f"Accuracy: {accuracy_percentage_train:.2f}%")
        st.write(f"Precision: {precision_percentage_train:.2f}%")
        st.write(f"Recall: {recall_sensitivity_percentage_train:.2f}%")
        st.write(f"Specificity: {specificity_percentage_train:.2f}%")
        st.write(f"F1 Score: {f1_train:.2f}")
        st.write(f"MCC: {mcc_train:.2f}")
        # st.write(f"Wilks' Lambda: {wilks_train:.2f}")
        st.write('Train length: ', len(y_train))
        st.write("tn_train: ", tn_train)
        st.write("fp_train: ", fp_train)
        st.write("fn_train: ", fn_train)
        st.write("tp_train: ", tp_train)
    with col2:
        st.subheader("Test Performance")
        st.write(f"Accuracy: {accuracy_percentage_test:.2f}%")
        st.write(f"Precision: {precision_percentage_test:.2f}%")
        st.write(f"Recall: {recall_sensitivity_percentage_test:.2f}%")
        st.write(f"Specificity: {specificity_percentage_test:.2f}%")
        st.write(f"F1 Score: {f1_test:.2f}")
        st.write(f"MCC: {mcc_test:.2f}")
        st.write('Test length: ', len(y_test))
        st.write("tn_test: ", tn_test)
        st.write("fp_test: ", fp_test)
        st.write("fn_test: ", fn_test)
        st.write("tp_test: ", tp_test)
            

    # Displaying the final equation
    st.title('Equation Display')
    st.write('Final Equation:', final_equation)
        
        
    st.title('Graphs Display')
        
        
    # Calculate AUC
    roc_auc = auc(fpr, tpr)

        # Plot the ROC curve
        # Plotting code
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Dashed line for random performance
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")
    # Display the plot in Streamlit
    st.pyplot(fig)
        


    # Plotting the Confusion Matrix
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(conf_m, annot=True, fmt="d", cmap="Blues", cbar=False, square=True, ax=ax_cm)
    class_labels = ['Negative', 'Positive']  # Adjust labels as appropriate for your data
    ax_cm.set_xlabel("Predicted labels")
    ax_cm.set_ylabel("True labels")
    ax_cm.set_title("Confusion Matrix")
    ax_cm.set_xticklabels(class_labels)
    ax_cm.set_yticklabels(class_labels)
    st.subheader('Confusion Matrix')
    st.pyplot(fig_cm)
         
         
    # st.write("tp_test: ", tp_test)
    st.write("X_train", X_train)
    st.write("X_test", X_test)
    st.write("y_train", y_train)
    st.write("y_test", y_test)
         