


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
# from Classification_GA import initialize_ga
# from fitness_function import compute_wilks_lambda
# import random
# from sklearn.metrics import matthews_corrcoef
# from scipy.stats import sem  # For standard error of the mean
# from Classification_GA import GA_LDA_main
# from Regression_GA import GA_MLR_main
# from BSS_LDA import BBS_LDA_main
# from BBS_MLR import BBS_MLR_main



# def run_gui():
    
    
#     st.title('LDA QSAR Model with GA Feature Selection')
    
    
#     # /////////////////////////// upload file /////////////////////////////
#     # File upload widget
#     uploaded_file = st.file_uploader("Upload your dataset (CSV or XLSX)", type=["csv", "xlsx"])

#     if uploaded_file is not None:
#         # Check the file type and read accordingly
#         if uploaded_file.name.endswith('.csv'):
#             dataset = pd.read_csv(uploaded_file) 
#         elif uploaded_file.name.endswith('.xlsx'):
#             dataset = pd.read_excel(uploaded_file)

#         # Display the first few rows of the dataset
#         st.write("Dataset preview:")
#         st.write(dataset.head())
        
#         feature_names = dataset.columns[2:]
#         X = dataset.iloc[:, 2:].values
#         uncodedlabels = dataset.iloc[:, 1].values
        
 
#         datasetlength = len(dataset.columns[2:]) > 3
        
#         if datasetlength :
            
#             # Encode the target variable
#             # encoding is just needed for Classification and not for regression.
#             le = LabelEncoder()
#             y = le.fit_transform(uncodedlabels)
#             # y = le.fit_transform(y)
            
#             # /////////////////////////// settings and options /////////////////////////////

#             # CONTROLLERS
#             # fitness_type = st.sidebar.selectbox('Select Fitness Type', ( 'Wilks Lambda Only', 'MCC + Wilks'))
#             qsar_approach = st.selectbox("Select the division method", ["Regression", "Classification"])
#             division_method = st.selectbox("Select the division method", ["random division", "Kennard stone", "Euclidean"])
            
#             # conditions of configurations depeding on the model you are plannin on train
#             if(qsar_approach == "Classification"):
                
#                 development_method = st.selectbox("Select the division method", ["GA-LDA", "BSS-LDA", " Random Forest"])
#                 fitness_method = st.sidebar.selectbox("Select the fitness method", ["Wilks Lambda", "MCC + Wilks Lambda"])
                            
#             else:
#                 development_method = st.selectbox("Select the division method", ["GA-MLR", "BSS-MLR"])
#                 fitness_method = st.sidebar.selectbox("Select the fitness method", ["LOO", "MAE"])
                
#             num_generations = st.sidebar.number_input('Number of Generations/iterations', 1, 100, 10) # 100
#             population_size = st.sidebar.number_input('Population Size', 1, 100, 10) # 50
#             num_features = st.sidebar.number_input('Equation Length', min_value=1, max_value=len(dataset.columns) - 2, value=3)
#             test_size = st.sidebar.slider('test size', 0.1, 1.0, 0.2, step=0.1)
            
#             mut_prob = st.sidebar.slider('Mutation Probability', 0.1, 1.0, .5, step=0.1)
#             cx_prob = st.sidebar.slider('Cross over Probability', 0.1, 1.0, 1., step=0.1)
            
#             random_seed = st.sidebar.number_input('Random Seed', value=1)
            
            
#             # /////////////////////////// run algorithm with the setting selected /////////////////////////////
            
            
#             # classification
#             if(development_method == "GA-LDA"):
                
                
                
#                 if st.button('Run Genetic Algorithm - LDA'):
#                     GA_LDA_main(X, y, feature_names, num_generations, population_size, mut_prob, num_features, division_method, test_size, fitness_method, cx_prob, random_seed)
            
            
#             elif(development_method == "GA-MLR"):
#                 if st.button('Run print'):
#                     print('uncodedlabels :', uncodedlabels, 'y', y)
#                 if st.button('Run Genetic Algorithm - MLR'):
#                     GA_MLR_main(X, y, feature_names, num_generations, population_size, mut_prob, num_features, division_method, test_size, fitness_method, cx_prob, random_seed)
                    
#             elif(development_method == "BSS-LDA"):
#                 if st.button('Run BBS - LDA'):
                    
#                     BBS_LDA_main(X, y, feature_names, num_features, division_method, test_size, fitness_method, random_seed)
#             elif(development_method == "BSS-MLR"):
#                 if st.button('Run BBS - MLR'):
                    
#                     BBS_MLR_main(X, uncodedlabels, feature_names, num_features, division_method, test_size, fitness_method, random_seed)
#                     # GA_MLR_main(X, y, feature_names, num_generations, population_size, mut_prob, num_features, division_method, test_size, fitness_method, cx_prob, random_seed)
#             else:
#                 if st.button('Run BBS - LDA'):
                    
#                     BBS_LDA_main(X, y, feature_names, num_features, division_method, test_size, fitness_method, random_seed)
#                     # GA_MLR_main(X, y, feature_names, num_generations, population_size, mut_prob, num_features, division_method, test_size, fitness_method, cx_prob, random_seed)
            
                  
#             # else:
#             #     if st.button('anything else - MLR'):
                  
#         else:
#             st.error("The dataset should have at least 3 columns of features + ID and Target Y columns.")

#     else:
#         st.write("Please upload a CSV or XLSX file.")

# if __name__ == "__main__":
#     run_gui()


import streamlit as st
import pandas as pd
# import numpy as np
from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.ensemble import RandomForestClassifier
# # from sklearn.metrics import accuracy_score, confusion_matrix
# from sklearn.metrics import (accuracy_score, precision_score, recall_score,
#                              f1_score, confusion_matrix, roc_curve, auc)
# import seaborn as sns
# import matplotlib.pyplot as plt
# from classification_division import kennard_stone_sampling, euclidean_sampling
# from Classification_GA import initialize_ga
# # from fitness_function import compute_wilks_lambda
# import random
# from sklearn.metrics import matthews_corrcoef
# from scipy.stats import sem  # For standard error of the mean


from GA_LDA import GA_LDA_main
from GA_MLR import GA_MLR_main
from BSS_LDA import BBS_LDA_main
from BBS_MLR import BBS_MLR_main
from preTreatment import  filter_features



def run_gui():
    
    
    st.title('LDA QSAR Model with GA Feature Selection')
    
    
    # /////////////////////////// upload file /////////////////////////////
    # File upload widget
    uploaded_file = st.file_uploader("Upload your dataset (CSV or XLSX)", type=["csv", "xlsx"])

    if uploaded_file is not None:
        # Check the file type and read accordingly
        if uploaded_file.name.endswith('.csv'):
            dataset = pd.read_csv(uploaded_file) 
        elif uploaded_file.name.endswith('.xlsx'):
            dataset = pd.read_excel(uploaded_file)

        # Display the first few rows of the dataset
        st.write("Dataset preview:")
        st.write(dataset.head())
        
        feature_names = dataset.columns[2:]
        # X = dataset.iloc[:, 2:].values
        uncodedlabels = dataset.iloc[:, 1].values
        
        datasetlength = len(dataset.columns[2:]) > 3
        
        if datasetlength :
            
            # Encode the target variable
            # encoding is just needed for Classification and not for regression.
            le = LabelEncoder()
            y = le.fit_transform(uncodedlabels)
            # y = le.fit_transform(y)
            
            
            # CONTROLLERS
            # fitness_type = st.sidebar.selectbox('Select Fitness Type', ( 'Wilks Lambda Only', 'MCC + Wilks'))
            qsar_approach = st.selectbox("Select the division method", ["Regression", "Classification"])
            division_method = st.selectbox("Select the division method", ["random division", "Kennard stone", "Euclidean"])
            preTreatment = st.checkbox("perform data pre treatment")
            if  preTreatment:
                variance_cutoff = st.number_input("Variance Cut-Off",
                                  min_value=0.0001, max_value=0.1, value=0.001, step=0.0001,
                                   format="%.4f",
                                  help="Set the minimum variance required to keep a feature")
                inter_r2_cutoff = st.number_input("Inter-R² Cut-Off",
                                  min_value=0.5, max_value=1.0, value=0.99, step=0.01,
                                  help="Set the maximum allowable R² (squared correlation) between features")

                # X = filter_features(dataset.iloc[:, 2:] ,variance_cutoff, inter_r2_cutoff)
                X = filter_features(dataset.iloc[:, 2:], variance_cutoff, inter_r2_cutoff).values
            else: 
                X = dataset.iloc[:, 2:].values
                
                
    
            
            
            # classification
            if(qsar_approach == "Classification"):
                
                # configs
                development_method = st.selectbox("Select the division method", ["GA-LDA", "BSS-LDA"])
                # fitness_method = st.sidebar.selectbox("Select the fitness method", ["Wilks Lambda", "MCC + Wilks Lambda"])
                if (development_method == 'GA-LDA'):
                    num_features = st.sidebar.number_input('Equation Length', min_value=1, max_value=len(dataset.columns) - 3, value=3)
                else:
                    num_features = st.sidebar.number_input('Equation Length', min_value=1, max_value=len(dataset.columns) - 2, value=3)
                    
                
                test_size = st.sidebar.slider('test size', 0.1, 1.0, 0.2, step=0.01)
                if(division_method == "random division"):
                    random_seed = st.sidebar.number_input('Random Seed', value=1)
                else:
                    random_seed = 1
                
                
                
                # function calls
                if(development_method == "GA-LDA"):
                    population_size = st.sidebar.number_input('Population Size', 1, 100, 10) # 50
                    num_generations = st.sidebar.number_input('Number of Generations/iterations', 1, 100, 10) # 100
                    mut_prob = st.sidebar.slider('Mutation Probability', 0.1, 1.0, .5, step=0.1)
                    cx_prob = st.sidebar.slider('Cross over Probability', 0.1, 1.0, 1., step=0.1)
                    
                    if st.button('Run Genetic Algorithm - LDA'):
                        GA_LDA_main(X, y, feature_names, num_generations, population_size, mut_prob, num_features, division_method, test_size, cx_prob, random_seed)
                
                elif(development_method == "BSS-LDA"):
                    if st.button('Run BBS - LDA'):
                        BBS_LDA_main(X, y, feature_names, num_features, division_method, test_size, random_seed)
                    # if st.button('test'):
                        # BBS_LDA_main(X, y, feature_names, num_features, division_method, test_size, fitness_method, random_seed)
            
            # regression approach
            elif(qsar_approach == "Regression"):
                
                # config
                development_method = st.selectbox("Select the division method", ["GA-MLR", "BSS-MLR"])
                fitness_method = st.sidebar.selectbox("Select the fitness method", ["LOO", "MAE"])
                if (development_method == 'GA-MLR'):
                    num_features = st.sidebar.number_input('Equation Length', min_value=1, max_value=len(dataset.columns) - 3, value=3)
                else:
                    num_features = st.sidebar.number_input('Equation Length', min_value=1, max_value=len(dataset.columns) - 2, value=3)
                
                # num_features = st.sidebar.number_input('Equation Length', min_value=1, max_value=len(dataset.columns) - 2, value=3)
                test_size = st.sidebar.slider('test size', 0.1, 1.0, 0.2, step=0.01)
                if(division_method == "random division"):
                    random_seed = st.sidebar.number_input('Random Seed', value=1)
                else:
                    random_seed = 1
                    
                
                # actual function calls
                if(development_method == "GA-MLR"):
                    
                    population_size = st.sidebar.number_input('Population Size', 1, 100, 10) # 50
                    num_generations = st.sidebar.number_input('Number of Generations/iterations', 1, 100, 10) # 100
                    mut_prob = st.sidebar.slider('Mutation Probability', 0.1, 1.0, .5, step=0.1)
                    cx_prob = st.sidebar.slider('Cross over Probability', 0.1, 1.0, 1., step=0.1)
    
                    if st.button('Run Genetic Algorithm - MLR'):
                        GA_MLR_main(X, uncodedlabels, feature_names, num_generations, population_size, mut_prob, num_features, division_method, test_size, fitness_method, cx_prob, random_seed)

                elif(development_method == "BSS-MLR"):
                    if st.button('Run BBS - MLR'):
                        BBS_MLR_main(X, uncodedlabels, feature_names, num_features, division_method, test_size, fitness_method, random_seed)
                        # GA_MLR_main(X, y, feature_names, num_generations, population_size, mut_prob, num_features, division_method, test_size, fitness_method, cx_prob, random_seed)
                else:
                    st.write("Select a valid method")
                    # print("before" , dataset.iloc[:, 2:])
                    # filtered_features = filter_features(dataset.iloc[:, 2:] ,variance_cutoff, inter_r2_cutoff)
                    # st.write(filtered_features.head())
                    # print("filtered_features", filtered_features)
                
                
            else:
                print("something is wrong please reload the page")
                  
        else:
            st.error("The dataset should have at least 3 columns of features + ID and Target Y columns.")

    else:
        st.write("Please upload a CSV or XLSX file.")

if __name__ == "__main__":
    run_gui()
