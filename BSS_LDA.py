from division_methods import kennard_stone_sampling, euclidean_sampling
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, roc_curve, auc)

from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
import random

   
    
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
        
        
    selected_feature_names = [feature_names[i] for i in selected_features]
    
    
    # Fit the model and make predictions on the test set
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    
    y_pred_train = lda.predict(X_train)
    y_pred_test = lda.predict(X_test)

    mcc_train = matthews_corrcoef(y_train, y_pred_train)
    mcc_test = matthews_corrcoef(y_test, y_pred_test)
    
    
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

    # Learning under the curve
    fpr, tpr, thresholds = roc_curve(y_test, lda.predict_proba(X_test)[:, 1])
    
    # final equation
    coefficients = lda.coef_[0]
    
    # Calculate standard errors
    y_pred = lda.predict(X_train)
    residuals = y_train - y_pred
    residual_var = np.var(residuals, ddof=len(coefficients))
    inv_hessian = np.linalg.inv(np.dot(X_train.T, X_train))
    standard_errors = np.sqrt(np.diagonal(residual_var * inv_hessian))

    # Create the equation parts
    equation_parts = [
        f"{coeff:.3f} (+/- {se:.3f}) * {name}" for coeff, se, name in zip(coefficients, standard_errors, selected_feature_names)
    ]
    final_equation = " + ".join(equation_parts)
    

    
    
    
    individual = {
            "mcc_train":mcc_train,
            "mcc_test":mcc_test,  
            "accuracy_test":accuracy_test,
            "accuracy_train":accuracy_train,
            "accuracy_percentage_test":accuracy_percentage_test,
            "accuracy_percentage_train":accuracy_percentage_train,
            "precision_percentage_test":precision_percentage_test,
            "precision_percentage_train":precision_percentage_train,
            "recall_sensitivity_percentage_test":recall_sensitivity_percentage_test,
            "recall_sensitivity_percentage_train":recall_sensitivity_percentage_train,
            "tn_test":tn_test, 
            "fp_test":fp_test, 
            "fn_test":fn_test, 
            "tp_test":tp_test,
            "tn_train":tn_train,
            "fp_train":fp_train,
            "fn_train":fn_train,
            "tp_train":tp_train,
            "specificity_percentage_test":specificity_percentage_test,
            "specificity_percentage_train":specificity_percentage_train,
            "f1_test":f1_test,
            "f1_train":f1_train,
            "fpr" : fpr,
            "tpr" : tpr, 
            "final_equation":final_equation,
            'X_train' : X_train,
            'X_test' : X_test,
            'y_train' : y_train,
            'y_test' : y_test,
            
        }
    
    
    

    return individual



# def BSS_LDA(arr, length):
def BSS_LDA(X, y, feature_names, division_method, test_size, length):
    
    dataset_length = X.shape[1]
    results = []
    n = dataset_length
            
    # Create an index list to represent the current combination
    indices = list(range(length))
    
    
    
    while True:
        current_individual = np.zeros(dataset_length, dtype=int)
        for i in indices:
            current_individual[i] = 1

        # wilks_train, mcc_train, fitness_score, mcc_test, wilks_test, tn, fp, fn, tp, fpr, tpr, final_equation = evaluate_individual(current_individual, X, y, feature_names, division_method, test_size, fitness_method)
        individual = evaluate_individual(current_individual, X, y, feature_names, division_method, test_size)

        
        
        
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
            
    results.sort(key=lambda ind: ind['mcc_train'], reverse=True)
            

 
            
    return results

    
    
    
def BBS_LDA_main(X, y, feature_names, equation_length, division_method, test_size, random_seed):
    

    Individuals = BSS_LDA(X, y, feature_names, division_method, test_size, equation_length)
    
    
    # df_individuals = pd.DataFrame(Individuals)
    df_individuals = pd.DataFrame([
        {key: value for key, value in individual.items() if not isinstance(value, (np.ndarray, list))}
        for individual in Individuals
    ])
    
    st.write("individuals")
    st.write(df_individuals.head())
    
    
    
    mcc_train = Individuals[0]["mcc_train"]
    mcc_test = Individuals[0]["mcc_test"]
    accuracy_test = Individuals[0]["accuracy_test"]
    accuracy_train = Individuals[0]["accuracy_train"]
    accuracy_percentage_test = Individuals[0]["accuracy_percentage_test"]
    accuracy_percentage_train = Individuals[0]["accuracy_percentage_train"]
    precision_percentage_test = Individuals[0]["precision_percentage_test"]
    precision_percentage_train = Individuals[0]["precision_percentage_train"]
    recall_sensitivity_percentage_test = Individuals[0]["recall_sensitivity_percentage_test"]
    recall_sensitivity_percentage_train = Individuals[0]["recall_sensitivity_percentage_train"]
    tn_test = Individuals[0]["tn_test"]
    fp_test = Individuals[0]["fp_test"]
    fn_test = Individuals[0]["fn_test"]
    tp_test = Individuals[0]["tp_test"]
    tn_train = Individuals[0]["tn_train"]
    fp_train = Individuals[0]["fp_train"]
    fn_train = Individuals[0]["fn_train"]
    tp_train = Individuals[0]["tp_train"]
    specificity_percentage_test = Individuals[0]["specificity_percentage_test"]
    specificity_percentage_train = Individuals[0]["specificity_percentage_train"]
    f1_test = Individuals[0]["f1_test"]
    f1_train = Individuals[0]["f1_train"]
    fpr = Individuals[0]["fpr"]
    tpr = Individuals[0]["tpr"]
    final_equation = Individuals[0]["final_equation"]
    X_train = Individuals[0]["X_train"]
    X_test = Individuals[0]["X_test"]
    y_train = Individuals[0]["y_train"]
    y_test = Individuals[0]["y_test"]
    
    
    
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
    
    
    
    roc_auc = auc(fpr, tpr)
    
    
    # Displaying the final equation
    st.title('Equation Display')
    st.write('Final Equation:', final_equation)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    # plt.show()
    st.subheader('ROC Curve')
    st.pyplot(plt)
    
    conf_m = [[tn_test, fp_test],[fn_test, tp_test]]
    fig, ax = plt.subplots()
    sns.heatmap(conf_m, annot=True, fmt="d", cmap="Blues", cbar=False, square=True, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
            
    st.subheader('Confusion Matrix')
    st.pyplot(fig)
    
    
    # training and test data print
    st.write("X_train", X_train)
    st.write("X_test", X_test)
    st.write("y_train", y_train)
    st.write("y_test", y_test)
    return Individuals
    
    
    
   
   