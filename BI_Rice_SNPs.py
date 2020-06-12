# -*- coding: utf-8 -*-
"""
Important Variable Selection with SNPs
Created on Fri Jan 31 16:31:01 2020

@author: Nhan TV
"""

# Import the libraries 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVR
from sklearn.linear_model import MultiTaskLassoCV, MultiTaskElasticNetCV, LassoCV, ElasticNetCV, MultiTaskElasticNet, MultiTaskLasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Using chunk size to read rice data
def read_x_cont():
    chunksize = 100
    X_ct = pd.DataFrame()
    for chunk in pd.read_csv("X_cont_ls_el.csv",low_memory=False, chunksize=chunksize, memory_map=True):
        X_ct = pd.concat([X_ct, chunk])
    return(X_ct)

# Function of data preprocessing
def process_variable(X, y): 
  # Drop 'IID' columns
  X = X.drop('IID', axis = 1)
  # Split data to training and testing set
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
  # Convert from integer to float
  X_train= X_train.astype(float, 32)
  X_test = X_test.astype(float, 32)
  # Apply the same scaling to both datasets
  scaler = StandardScaler()
  X_train_scl = scaler.fit_transform(X_train)
  X_test_scl = scaler.transform(X_test) #  we transform rather than fit_transform

  return(X_train_scl, X_test_scl, y_train, y_test)

"""Random Forest Regressor"""
#Function to run random forest with grid search and k-fold cross-validation.
def get_rf_model(X_train, y_train, X_test, y_test):
  # Hyperparameters search grid 
  rf_param_grid = {'bootstrap': [False, True],
          'n_estimators': [60, 70, 80, 90, 100],
          'max_features': [0.6, 0.65, 0.7, 0.75, 0.8],
          'min_samples_leaf': [1],
          'min_samples_split': [2]
          }
  # Instantiate random forest regressor
  rf_estimator = RandomForestRegressor(random_state=None)
  # Create the GridSearchCV object
  rf_model = GridSearchCV(estimator=rf_estimator, param_grid=rf_param_grid, cv=10, scoring='neg_mean_squared_error', n_jobs=-1, iid = True)
  # Train the regressor
  rf_model.fit(X_train, y_train)
  # Get the best model
  rf_model_best = rf_model.best_estimator_
  # Make predictions using the optimised parameters
  rf_pred = rf_model_best.predict(X_test)
  # Find mean squared error
  mse = mean_squared_error(y_test, rf_pred)
  # Find r-squared 
  r2 = r2_score(y_test, rf_pred)
  best_prs = rf_model.best_params_

  print("Best Parameters:\n", rf_model.best_params_)
  print("Best Score:\n", 'mse:', mse, 'r2:', r2)
  return(mse, r2, best_prs)

"""Support Vector Regressor"""
#Function to run support vector machine with grid search and k-fold cross-validation.
def get_svm_model(X_train, y_train, X_test, y_test):
  # Parameter grid
  svm_param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001, 10], "kernel": ["rbf"]}  
  # Create SVM grid search regressor
  svm_grid = GridSearchCV(estimator = SVR(), param_grid= svm_param_grid, cv=10, scoring='neg_mean_squared_error', n_jobs=-1, iid = True)
  # Train the regressor
  svm_grid.fit(X_train, y_train)
  # Get the best model
  svm_model_best = svm_grid.best_estimator_
  # Make predictions using the optimised parameters
  svm_pred = svm_model_best.predict(X_test)
  # Find mean squared error
  mse = mean_squared_error(y_test, svm_pred)
  # Find r-squared 
  r2 = r2_score(y_test, svm_pred)
  best_prs = svm_grid.best_params_
  
  print("Best Parameters:\n", svm_grid.best_params_)
  print("Best Score:\n", 'mse:', mse, 'r2:', r2)
  return(mse, r2, best_prs)
  
"""Lasso and Multi Task Lasso"""
#Lasso
def get_lasso_cv(X_train, y_train, X_test, y_test, cols):
  # Create Lasso CV
  ls_grid = LassoCV(cv = 10, random_state = 0, n_jobs = -1)
  # Train the regressor
  ls_grid.fit(X_train, y_train)
  # Make predictions using the optimised parameters
  ls_pred = ls_grid.predict(X_test)
  # Find mean squared error
  mse = mean_squared_error(y_test, ls_pred)
  # Find r-squared 
  r2 = r2_score(y_test, ls_pred)
    
  best_prs = ls_grid.alpha_
  print("Best Parameters:\n", best_prs)
  print("Best Score:\n", 'mse:', mse, 'r2:', r2)

  # Get coefficients of the model 
  coef = pd.DataFrame(ls_grid.coef_.T, index = cols)
  var = list(coef[coef[0] != 0].index)
  print(coef.head())
  print("Lasso picked " + str(sum(coef[0] != 0)) + " variables and eliminated the other " +  str(sum(coef[0] == 0)) + " variables")
  return(mse, r2, var, best_prs)

# Multi-task Lasso  
def get_multitask_lasso_cv(X_train, y_train, X_test, y_test, cols):
  # Create Multi-task Lasso CV
  ls_grid = MultiTaskLassoCV(cv = 10, random_state = 0, n_jobs = -1)
  # Train the regressor
  ls_grid.fit(X_train, y_train)
  # Make predictions using the optimised parameters
  ls_pred = ls_grid.predict(X_test)
  # Find mean squared error
  mse = mean_squared_error(y_test, ls_pred)
  # Find r-squared 
  r2 = r2_score(y_test, ls_pred)
    
  best_prs = ls_grid.alpha_
  print("Best Parameters:\n", best_prs)
  print("Best Score:\n", 'mse:', mse, 'r2:', r2)

  # Get coefficients of the model 
  coef = pd.DataFrame(ls_grid.coef_.T, index = cols)
  var = list(coef[coef[0] != 0].index)
  print(coef.head())
  print("Multit-task Lasso picked " + str(sum(coef[0] != 0)) + " variables and eliminated the other " +  str(sum(coef[0] == 0)) + " variables")
  return(mse, r2, var, best_prs)
  
"""Elastic Net and Multi Task Elastic Net"""

# Elastic Net
def get_elasticnet_cv(X_train, y_train, X_test, y_test, cols): 
  # Create Elastic Net CV
  el_grid = ElasticNetCV(cv = 10, random_state = 0, n_jobs = -1)
  # Train the regressor
  el_grid.fit(X_train, y_train)
  # Make predictions using the optimised parameters
  el_pred = el_grid.predict(X_test)
  # Find mean squared error
  mse = mean_squared_error(y_test, el_pred)
  # Find r-squared 
  r2 = r2_score(y_test, el_pred)
    
  best_prs = [el_grid.alpha_]
  best_prs.append(el_grid.l1_ratio_)
  print("Best Parameters:\n", best_prs)
  print("Best Score:\n", 'mse:', mse, 'r-squared:', r2)
    
  # Get coefficients of the model 
  coef = pd.DataFrame(el_grid.coef_.T, index = cols)
  var = list(coef[coef[0] != 0].index)
  print(coef.head())
  print("ElasticNet picked " + str(sum(coef[0] != 0)) + " variables and eliminated the other " +  str(sum(coef[0] == 0)) + " variables")
  return(mse, r2, var, best_prs)
    
# Multi-task Elastic Net
def get_multitask_elasticnet_cv(X_train, y_train, X_test, y_test, cols): 
  # Create Multi Task Elastic Net CV
  el_grid = MultiTaskElasticNetCV(cv = 10, random_state = 0, n_jobs = -1)
  # Train the regressor
  el_grid.fit(X_train, y_train)
  # Make predictions using the optimised parameters
  el_pred = el_grid.predict(X_test)
  # Find mean squared error
  mse = mean_squared_error(y_test, el_pred)
  # Find r-squared 
  r2 = r2_score(y_test, el_pred)
    
  best_prs = [el_grid.alpha_]
  best_prs.append(el_grid.l1_ratio_)
  print("Best Parameters:\n", best_prs)
  print("Best Score:\n", 'mse:', mse, 'r-squared:', r2)
    
  # Get coefficients of the model 
  coef = pd.DataFrame(el_grid.coef_.T, index = cols)
  var = list(coef[coef[0] != 0].index)
  print(coef.head())
  print("Multi-task ElasticNet picked " + str(sum(coef[0] != 0)) + " variables and eliminated the other " +  str(sum(coef[0] == 0)) + " variables")
  return(mse, r2, var, best_prs)

# Evaluation each trait by multi-task Lasso  
def eval_mtls_split_trait(alpha, X_train, Y_train, X_test, Y_test):
  # Create Multi-Task Lasso 
  ls_tfl_grw = MultiTaskLasso(alpha, random_state = 0)
  # Train the regressor   
  ls_tfl_grw.fit(X_train, Y_train)
  # Make predictions using the optimised parameters
  ls_pred = ls_tfl_grw.predict(X_test)
  # Find mean squared error
  mse_tfl = mean_squared_error(Y_test[:, 0], ls_pred[:, 0])
  mse_grw= mean_squared_error(Y_test[:, 1], ls_pred[:, 1])
  # Find r-squared 
  r2_tfl = r2_score(Y_test[:, 0], ls_pred[:, 0])
  r2_grw = r2_score(Y_test[:, 1], ls_pred[:, 1])
  return(mse_tfl, mse_grw, r2_tfl, r2_grw)
  
# Evaluation each trait by multi-task Elastic Net    
def eval_mtel_split_trait(alpha, l1_ratio, X_train, Y_train, X_test, Y_test):
  # Create Multi-Task Lasso 
  el_tfl_grw = MultiTaskElasticNet(alpha, l1_ratio, random_state = 0)
  # Train the regressor   
  el_tfl_grw.fit(X_train, Y_train)
  # Make predictions using the optimised parameters
  el_pred = el_tfl_grw.predict(X_test)
  # Find mean squared error
  mse_tfl = mean_squared_error(Y_test[:, 0], el_pred[:, 0])
  mse_grw= mean_squared_error(Y_test[:, 1], el_pred[:, 1])
  # Find r-squared 
  r2_tfl = r2_score(Y_test[:, 0], el_pred[:, 0])
  r2_grw = r2_score(Y_test[:, 1], el_pred[:, 1])
  return(mse_tfl, mse_grw, r2_tfl, r2_grw)

if __name__ == '__main__':
    print("")
    print("")
    print("|============================================================================|")
    print("|                                                                            |")
    print("|         -----     IMPORTANT VARIABLE SELECTION WITH SNPS    -----          |")
    print("|                                                                            |")
    print("|============================================================================|")
    print("")
    print("")
    print("********************************* INPUT DATA *********************************")
    print("")
    print("Import data may take several minutes, please wait...")
    print("")
    
    # Import data
    X_cont = read_x_cont()
    cols = X_cont.columns[1::]
    
    # Load data after pre-processinng
    y_tfl = pd.read_csv("y_tfl.csv", header=None)
    y_grw = pd.read_csv("y_grw.csv", header=None)
    y_tfl_grw = pd.read_csv("y_tfl_grw.csv", header=None)
    
    X_grw_2 = pd.read_csv("X_grw_2.csv", header='infer')
    X_grw_3 = pd.read_csv("X_grw_3.csv", header='infer')
    X_grw_4 = pd.read_csv("X_grw_4.csv", header='infer')
    X_grw_5 = pd.read_csv("X_grw_5.csv", header='infer')
    
    X_tfl_2 = pd.read_csv("X_tfl_2.csv", header='infer')
    X_tfl_3 = pd.read_csv("X_tfl_3.csv", header='infer')
    X_tfl_4 = pd.read_csv("X_tfl_4.csv", header='infer')
    X_tfl_5 = pd.read_csv("X_tfl_5.csv", header='infer')
    X_tfl_6 = pd.read_csv("X_tfl_6.csv", header='infer')
    
    X_tfl_grw_2 = pd.read_csv("X_tfl_grw_2.csv", header='infer')
    X_tfl_grw_25 = pd.read_csv("X_tfl_grw_25.csv", header='infer')
    X_tfl_grw_1 = pd.read_csv("X_tfl_grw_1.csv", header='infer')
    X_tfl_grw_75 = pd.read_csv("X_tfl_grw_75.csv", header='infer')
    X_tfl_grw_3 = pd.read_csv("X_tfl_grw_3.csv", header='infer')
    print("")
    
    # Transform response variables to matrix type.
    y_tfl = y_tfl.values.ravel()
    y_grw = y_grw.values.ravel()
    y_tfl_grw = y_tfl_grw.values
    
    # Normalize rice data
    X_grw_2_train, X_grw_2_test, y_grw_2_train, y_grw_2_test = process_variable(X_grw_2, y_grw)
    X_grw_3_train, X_grw_3_test, y_grw_3_train, y_grw_3_test = process_variable(X_grw_3, y_grw)
    X_grw_4_train, X_grw_4_test, y_grw_4_train, y_grw_4_test = process_variable(X_grw_4, y_grw)
    X_grw_5_train, X_grw_5_test, y_grw_5_train, y_grw_5_test = process_variable(X_grw_5, y_grw)
    
    X_tfl_2_train, X_tfl_2_test, y_tfl_2_train, y_tfl_2_test = process_variable(X_tfl_2, y_tfl)
    X_tfl_3_train, X_tfl_3_test, y_tfl_3_train, y_tfl_3_test = process_variable(X_tfl_3, y_tfl)
    X_tfl_4_train, X_tfl_4_test, y_tfl_4_train, y_tfl_4_test = process_variable(X_tfl_4, y_tfl)
    X_tfl_5_train, X_tfl_5_test, y_tfl_5_train, y_tfl_5_test = process_variable(X_tfl_5, y_tfl)
    X_tfl_6_train, X_tfl_6_test, y_tfl_6_train, y_tfl_6_test = process_variable(X_tfl_6, y_tfl)
    
    X_tfl_grw_2_train, X_tfl_grw_2_test, y_tfl_grw_2_train, y_tfl_grw_2_test = process_variable(X_tfl_grw_2, y_tfl_grw)
    X_tfl_grw_25_train, X_tfl_grw_25_test, y_tfl_grw_25_train, y_tfl_grw_25_test = process_variable(X_tfl_grw_25, y_tfl_grw)
    X_tfl_grw_1_train, X_tfl_grw_1_test, y_tfl_grw_1_train, y_tfl_grw_1_test = process_variable(X_tfl_grw_1, y_tfl_grw)
    X_tfl_grw_75_train, X_tfl_grw_75_test, y_tfl_grw_75_train, y_tfl_grw_75_test = process_variable(X_tfl_grw_75, y_tfl_grw)
    X_tfl_grw_3_train, X_tfl_grw_3_test, y_tfl_grw_3_train, y_tfl_grw_3_test = process_variable(X_tfl_grw_3, y_tfl_grw)
   
    X_grw_train, X_grw_test, y_grw_train, y_grw_test = process_variable(X_cont, y_grw)
    X_tfl_train, X_tfl_test, y_tfl_train, y_tfl_test = process_variable(X_cont, y_tfl)
    X_tfl_grw_train, X_tfl_grw_test, y_tfl_grw_train, y_tfl_grw_test = process_variable(X_cont, y_tfl_grw)
    
    print("")
    print("******************************* TRAINING MODELS *****************************")
    print("")
    
    rf_grw_mse = []
    rf_grw_r2 = []
    rf_tfl_mse = []
    rf_tfl_r2 = []
    rf_grw_prs = []
    rf_tfl_prs = []
    rf_tfl_grw_mse_0 = []
    rf_tfl_grw_r2_0 = []
    rf_tfl_grw_prs_0 = []
    rf_tfl_grw_mse_1 = []
    rf_tfl_grw_r2_1 = []
    rf_tfl_grw_prs_1 = []
    
    svr_grw_mse = []
    svr_grw_r2 = []
    svr_tfl_mse = []
    svr_tfl_r2 = []
    svr_grw_prs = []
    svr_tfl_prs = []
    svr_tfl_grw_mse_0 = []
    svr_tfl_grw_r2_0 = []
    svr_tfl_grw_prs_0 = []
    svr_tfl_grw_mse_1 = []
    svr_tfl_grw_r2_1 = []
    svr_tfl_grw_prs_1 = []
    
    # Filtering variables by p_value. 
    p_value = ['<=5e-6', '<=5e-5', '<=5e-4', '<=5e-3', '<=5e-2']
    p_value_2 = ['<=5e-3','<=7.5e-3', '<=1e-2', '<=2.5e-2', '<=5e-2']
    
    print("Find mse and r-squared for random forest model of grain weight...") 
    rf_grw_mse_2, rf_grw_r2_2, rf_grw_prs_2  = get_rf_model(X_grw_2_train, y_grw_2_train, X_grw_2_test, y_grw_2_test)
    rf_grw_mse.append(rf_grw_mse_2)
    rf_grw_r2.append(rf_grw_r2_2)
    rf_grw_prs.append(rf_grw_prs_2)
    
    rf_grw_mse_3, rf_grw_r2_3, rf_grw_prs_3  = get_rf_model(X_grw_3_train, y_grw_3_train, X_grw_3_test, y_grw_3_test)
    rf_grw_mse.append(rf_grw_mse_3)
    rf_grw_r2.append(rf_grw_r2_3)
    rf_grw_prs.append(rf_grw_prs_3)
    
    rf_grw_mse_4, rf_grw_r2_4, rf_grw_prs_4  = get_rf_model(X_grw_4_train, y_grw_4_train, X_grw_4_test, y_grw_4_test)
    rf_grw_mse.append(rf_grw_mse_4)
    rf_grw_r2.append(rf_grw_r2_4)
    rf_grw_prs.append(rf_grw_prs_4)
    
    rf_grw_mse_5, rf_grw_r2_5, rf_grw_prs_5  = get_rf_model(X_grw_5_train, y_grw_5_train, X_grw_5_test, y_grw_5_test)
    rf_grw_mse.append(rf_grw_mse_5)
    rf_grw_r2.append(rf_grw_r2_5)
    rf_grw_prs.append(rf_grw_prs_5)
    
    rf_grw = pd.DataFrame({'rf_grw_mse':rf_grw_mse[::-1], 'rf_grw_r2':rf_grw_r2[::-1], 'rf_grw_prs':rf_grw_prs[::-1]})
    rf_grw.set_index(pd.Index(p_value[1:5]), 'p_value', inplace = True)
    rf_grw.to_csv('rf_grw.csv')
    print('RF of grain weight is saved')
    
    print("Find mse and r-squared for random forest model of time to flowering...")
    rf_tfl_mse_2, rf_tfl_r2_2, rf_tfl_prs_2 = get_rf_model(X_tfl_2_train, y_tfl_2_train, X_tfl_2_test, y_tfl_2_test)
    rf_tfl_mse.append(rf_tfl_mse_2)
    rf_tfl_r2.append(rf_tfl_r2_2)
    rf_tfl_prs.append(rf_tfl_prs_2)
    
    rf_tfl_mse_3, rf_tfl_r2_3, rf_tfl_prs_3 = get_rf_model(X_tfl_3_train, y_tfl_3_train, X_tfl_3_test, y_tfl_3_test)
    rf_tfl_mse.append(rf_tfl_mse_3)
    rf_tfl_r2.append(rf_tfl_r2_3)
    rf_tfl_prs.append(rf_tfl_prs_3)
    
    rf_tfl_mse_4, rf_tfl_r2_4, rf_tfl_prs_4 = get_rf_model(X_tfl_4_train, y_tfl_4_train, X_tfl_4_test, y_tfl_4_test)
    rf_tfl_mse.append(rf_tfl_mse_4)
    rf_tfl_r2.append(rf_tfl_r2_4)
    rf_tfl_prs.append(rf_tfl_prs_4)
    
    rf_tfl_mse_5, rf_tfl_r2_5, rf_tfl_prs_5 = get_rf_model(X_tfl_5_train, y_tfl_5_train, X_tfl_5_test, y_tfl_5_test)
    rf_tfl_mse.append(rf_tfl_mse_5)
    rf_tfl_r2.append(rf_tfl_r2_5)
    rf_tfl_prs.append(rf_tfl_prs_5)
    
    rf_tfl_mse_6, rf_tfl_r2_6, rf_tfl_prs_6 = get_rf_model(X_tfl_6_train, y_tfl_6_train, X_tfl_6_test, y_tfl_6_test)
    rf_tfl_mse.append(rf_tfl_mse_6)
    rf_tfl_r2.append(rf_tfl_r2_6)
    rf_tfl_prs.append(rf_tfl_prs_6)
    
    rf_tfl = pd.DataFrame({'rf_tfl_mse':rf_tfl_mse[::-1], 'rf_tfl_r2':rf_tfl_r2[::-1], 'rf_tfl_prs':rf_tfl_prs[::-1]})
    rf_tfl.set_index(pd.Index(p_value), 'p_value', inplace = True)
    rf_tfl.to_csv('rf_tfl.csv')
    print('RF of time to flowering is saved')
    
    print("Find mse and r-squared for random forest model of time to flowering and grain weight...")
    # Output is time to flowering
    rf_tfl_grw_mse_2_0, rf_tfl_grw_r2_2_0, rf_tfl_grw_prs_2_0 = get_rf_model(X_tfl_grw_2_train, y_tfl_grw_2_train[:, 0], X_tfl_grw_2_test, y_tfl_grw_2_test[:, 0])
    rf_tfl_grw_mse_0.append(rf_tfl_grw_mse_2_0)
    rf_tfl_grw_r2_0.append(rf_tfl_grw_r2_2_0)
    rf_tfl_grw_prs_0.append(rf_tfl_grw_prs_2_0)
    
    rf_tfl_grw_mse_25_0, rf_tfl_grw_r2_25_0, rf_tfl_grw_prs_25_0 = get_rf_model(X_tfl_grw_25_train, y_tfl_grw_25_train[:, 0], X_tfl_grw_25_test, y_tfl_grw_25_test[:, 0])
    rf_tfl_grw_mse_0.append(rf_tfl_grw_mse_25_0)
    rf_tfl_grw_r2_0.append(rf_tfl_grw_r2_25_0)
    rf_tfl_grw_prs_0.append(rf_tfl_grw_prs_25_0)
    
    rf_tfl_grw_mse_1_0, rf_tfl_grw_r2_1_0, rf_tfl_grw_prs_1_0 = get_rf_model(X_tfl_grw_1_train, y_tfl_grw_1_train[:, 0], X_tfl_grw_1_test, y_tfl_grw_1_test[:, 0])
    rf_tfl_grw_mse_0.append(rf_tfl_grw_mse_1_0)
    rf_tfl_grw_r2_0.append(rf_tfl_grw_r2_1_0)
    rf_tfl_grw_prs_0.append(rf_tfl_grw_prs_1_0)
    
    rf_tfl_grw_mse_75_0, rf_tfl_grw_r2_75_0, rf_tfl_grw_prs_75_0 = get_rf_model(X_tfl_grw_75_train, y_tfl_grw_75_train[:, 0], X_tfl_grw_75_test, y_tfl_grw_75_test[:, 0])
    rf_tfl_grw_mse_0.append(rf_tfl_grw_mse_75_0)
    rf_tfl_grw_r2_0.append(rf_tfl_grw_r2_75_0)
    rf_tfl_grw_prs_0.append(rf_tfl_grw_prs_75_0)
    
    rf_tfl_grw_mse_3_0, rf_tfl_grw_r2_3_0, rf_tfl_grw_prs_3_0 = get_rf_model(X_tfl_grw_3_train, y_tfl_grw_3_train[:, 0], X_tfl_grw_3_test, y_tfl_grw_3_test[:, 0])
    rf_tfl_grw_mse_0.append(rf_tfl_grw_mse_3_0)
    rf_tfl_grw_r2_0.append(rf_tfl_grw_r2_3_0)
    rf_tfl_grw_prs_0.append(rf_tfl_grw_prs_3_0)
    
    rf_tfl_grw_0 = pd.DataFrame({'rf_tfl_grw_mse_0':rf_tfl_grw_mse_0[::-1], 'rf_tfl_grw_r2_0':rf_tfl_grw_r2_0[::-1], 'rf_tfl_grw_prs_0':rf_tfl_grw_prs_0[::-1]})
    rf_tfl_grw_0.set_index(pd.Index(p_value_2), 'p_value', inplace = True)
    rf_tfl_grw_0.to_csv('rf_tfl_grw_0.csv')
    
    # Output is grain weight
    rf_tfl_grw_mse_2_1, rf_tfl_grw_r2_2_1, rf_tfl_grw_prs_2_1 = get_rf_model(X_tfl_grw_2_train, y_tfl_grw_2_train[:, 1], X_tfl_grw_2_test, y_tfl_grw_2_test[:, 1])
    rf_tfl_grw_mse_1.append(rf_tfl_grw_mse_2_1)
    rf_tfl_grw_r2_1.append(rf_tfl_grw_r2_2_1)
    rf_tfl_grw_prs_1.append(rf_tfl_grw_prs_2_1)
    
    rf_tfl_grw_mse_25_1, rf_tfl_grw_r2_25_1, rf_tfl_grw_prs_25_1 = get_rf_model(X_tfl_grw_25_train, y_tfl_grw_25_train[:, 1], X_tfl_grw_25_test, y_tfl_grw_25_test[:, 1])
    rf_tfl_grw_mse_1.append(rf_tfl_grw_mse_25_1)
    rf_tfl_grw_r2_1.append(rf_tfl_grw_r2_25_1)
    rf_tfl_grw_prs_1.append(rf_tfl_grw_prs_25_1)
    
    rf_tfl_grw_mse_1_1, rf_tfl_grw_r2_1_1, rf_tfl_grw_prs_1_1 = get_rf_model(X_tfl_grw_1_train, y_tfl_grw_1_train[:, 1], X_tfl_grw_1_test, y_tfl_grw_1_test[:, 1])
    rf_tfl_grw_mse_1.append(rf_tfl_grw_mse_1_1)
    rf_tfl_grw_r2_1.append(rf_tfl_grw_r2_1_1)
    rf_tfl_grw_prs_1.append(rf_tfl_grw_prs_1_1)
    
    rf_tfl_grw_mse_75_1, rf_tfl_grw_r2_75_1, rf_tfl_grw_prs_75_1 = get_rf_model(X_tfl_grw_75_train, y_tfl_grw_75_train[:, 1], X_tfl_grw_75_test, y_tfl_grw_75_test[:, 1])
    rf_tfl_grw_mse_1.append(rf_tfl_grw_mse_75_1)
    rf_tfl_grw_r2_1.append(rf_tfl_grw_r2_75_1)
    rf_tfl_grw_prs_1.append(rf_tfl_grw_prs_75_1)
    
    rf_tfl_grw_mse_3_1, rf_tfl_grw_r2_3_1, rf_tfl_grw_prs_3_1 = get_rf_model(X_tfl_grw_3_train, y_tfl_grw_3_train[:, 1], X_tfl_grw_3_test, y_tfl_grw_3_test[:, 1])
    rf_tfl_grw_mse_1.append(rf_tfl_grw_mse_3_1)
    rf_tfl_grw_r2_1.append(rf_tfl_grw_r2_3_1)
    rf_tfl_grw_prs_1.append(rf_tfl_grw_prs_3_1)
    
    rf_tfl_grw_1 = pd.DataFrame({'rf_tfl_grw_mse_1':rf_tfl_grw_mse_1[::-1], 'rf_tfl_grw_r2_1':rf_tfl_grw_r2_1[::-1], 'rf_tfl_grw_prs_1':rf_tfl_grw_prs_1[::-1]})
    rf_tfl_grw_1.set_index(pd.Index(p_value_2), 'p_value', inplace = True)
    rf_tfl_grw_1.to_csv('rf_tfl_grw_1.csv')
    print('RF of time to flowering and grain weight is saved')
    
    print("Find mse and r-squared for svm model of grain weight...") 
    
    svr_grw_mse_2, svr_grw_r2_2, svr_grw_prs_2 = get_svm_model(X_grw_2_train, y_grw_2_train, X_grw_2_test, y_grw_2_test)
    svr_grw_mse.append(svr_grw_mse_2)
    svr_grw_r2.append(svr_grw_r2_2)
    svr_grw_prs.append(svr_grw_prs_2)
    
    svr_grw_mse_3, svr_grw_r2_3, svr_grw_prs_3 = get_svm_model(X_grw_3_train, y_grw_3_train, X_grw_3_test, y_grw_3_test)
    svr_grw_mse.append(svr_grw_mse_3)
    svr_grw_r2.append(svr_grw_r2_3)
    svr_grw_prs.append(svr_grw_prs_3)
    
    svr_grw_mse_4, svr_grw_r2_4, svr_grw_prs_4 = get_svm_model(X_grw_4_train, y_grw_4_train, X_grw_4_test, y_grw_4_test)
    svr_grw_mse.append(svr_grw_mse_4)
    svr_grw_r2.append(svr_grw_r2_4)
    svr_grw_prs.append(svr_grw_prs_4)
    
    svr_grw_mse_5, svr_grw_r2_5, svr_grw_prs_5 = get_svm_model(X_grw_5_train, y_grw_5_train, X_grw_5_test, y_grw_5_test)
    svr_grw_mse.append(svr_grw_mse_5)
    svr_grw_r2.append(svr_grw_r2_5)
    svr_grw_prs.append(svr_grw_prs_5)
    
    svr_grw = pd.DataFrame({'svr_grw_mse':svr_grw_mse[::-1], 'svr_grw_r2':svr_grw_r2[::-1], 'svr_grw_prs':svr_grw_prs[::-1]})
    svr_grw.set_index(pd.Index(p_value[1:5]), 'p_value', inplace = True)
    svr_grw.to_csv('svr_grw.csv')
    print('SVR of grain weight is saved')
    
    print("Find mse and r-squared for svm model of time to flowering...")
    
    svr_tfl_mse_2, svr_tfl_r2_2, svr_tfl_prs_2 = get_svm_model(X_tfl_2_train, y_tfl_2_train, X_tfl_2_test, y_tfl_2_test)
    svr_tfl_mse.append(svr_tfl_mse_2)
    svr_tfl_r2.append(svr_tfl_r2_2)
    svr_tfl_prs.append(svr_tfl_prs_2)
    
    svr_tfl_mse_3, svr_tfl_r2_3, svr_tfl_prs_3 = get_svm_model(X_tfl_3_train, y_tfl_3_train, X_tfl_3_test, y_tfl_3_test)
    svr_tfl_mse.append(svr_tfl_mse_3)
    svr_tfl_r2.append(svr_tfl_r2_3)
    svr_tfl_prs.append(svr_tfl_prs_3)
    
    svr_tfl_mse_4, svr_tfl_r2_4, svr_tfl_prs_4 = get_svm_model(X_tfl_4_train, y_tfl_4_train, X_tfl_4_test, y_tfl_4_test)
    svr_tfl_mse.append(svr_tfl_mse_4)
    svr_tfl_r2.append(svr_tfl_r2_4)
    svr_tfl_prs.append(svr_tfl_prs_4)
    
    svr_tfl_mse_5, svr_tfl_r2_5, svr_tfl_prs_5 = get_svm_model(X_tfl_5_train, y_tfl_5_train, X_tfl_5_test, y_tfl_5_test)
    svr_tfl_mse.append(svr_tfl_mse_5)
    svr_tfl_r2.append(svr_tfl_r2_5)
    svr_tfl_prs.append(svr_tfl_prs_5)
    
    svr_tfl_mse_6, svr_tfl_r2_6, svr_tfl_prs_6 = get_svm_model(X_tfl_6_train, y_tfl_6_train, X_tfl_6_test, y_tfl_6_test)
    svr_tfl_mse.append(svr_tfl_mse_6)
    svr_tfl_r2.append(svr_tfl_r2_6)
    svr_tfl_prs.append(svr_tfl_prs_6)
    
    svr_tfl = pd.DataFrame({'svr_tfl_mse':svr_tfl_mse[::-1], 'svr_tfl_r2':svr_tfl_r2[::-1], 'svr_tfl_prs':svr_tfl_prs[::-1]})
    svr_tfl.set_index(pd.Index(p_value), 'p_value', inplace = True)
    svr_tfl.to_csv('svr_tfl.csv')
    print('SVR of time to flowering is saved')
    
    print("Find mse and r-squared for svm model of time to flowering and grain weight... ")
    # Output is time to flowering
    svr_tfl_grw_mse_2_0, svr_tfl_grw_r2_2_0, svr_tfl_grw_prs_2_0 = get_svm_model(X_tfl_grw_2_train, y_tfl_grw_2_train[:, 0], X_tfl_grw_2_test, y_tfl_grw_2_test[:, 0])
    svr_tfl_grw_mse_0.append(svr_tfl_grw_mse_2_0)
    svr_tfl_grw_r2_0.append(svr_tfl_grw_r2_2_0)
    svr_tfl_grw_prs_0.append(svr_tfl_grw_prs_2_0)
    
    svr_tfl_grw_mse_25_0, svr_tfl_grw_r2_25_0, svr_tfl_grw_prs_25_0 = get_svm_model(X_tfl_grw_25_train, y_tfl_grw_25_train[:, 0], X_tfl_grw_25_test, y_tfl_grw_25_test[:, 0])
    svr_tfl_grw_mse_0.append(svr_tfl_grw_mse_25_0)
    svr_tfl_grw_r2_0.append(svr_tfl_grw_r2_25_0)
    svr_tfl_grw_prs_0.append(svr_tfl_grw_prs_25_0)
    
    svr_tfl_grw_mse_1_0, svr_tfl_grw_r2_1_0, svr_tfl_grw_prs_1_0 = get_svm_model(X_tfl_grw_1_train, y_tfl_grw_1_train[:, 0], X_tfl_grw_1_test, y_tfl_grw_1_test[:, 0])
    svr_tfl_grw_mse_0.append(svr_tfl_grw_mse_1_0)
    svr_tfl_grw_r2_0.append(svr_tfl_grw_r2_1_0)
    svr_tfl_grw_prs_0.append(svr_tfl_grw_prs_1_0)
    
    svr_tfl_grw_mse_75_0, svr_tfl_grw_r2_75_0, svr_tfl_grw_prs_75_0 = get_svm_model(X_tfl_grw_75_train, y_tfl_grw_75_train[:, 0], X_tfl_grw_75_test, y_tfl_grw_75_test[:, 0])
    svr_tfl_grw_mse_0.append(svr_tfl_grw_mse_75_0)
    svr_tfl_grw_r2_0.append(svr_tfl_grw_r2_75_0)
    svr_tfl_grw_prs_0.append(svr_tfl_grw_prs_75_0)
    
    svr_tfl_grw_mse_3_0, svr_tfl_grw_r2_3_0, svr_tfl_grw_prs_3_0 = get_svm_model(X_tfl_grw_3_train, y_tfl_grw_3_train[:, 0], X_tfl_grw_3_test, y_tfl_grw_3_test[:, 0])
    svr_tfl_grw_mse_0.append(svr_tfl_grw_mse_3_0)
    svr_tfl_grw_r2_0.append(svr_tfl_grw_r2_3_0)
    svr_tfl_grw_prs_0.append(svr_tfl_grw_prs_3_0)
   
    svr_tfl_grw_0 = pd.DataFrame({'svr_tfl_grw_mse_0':svr_tfl_grw_mse_0[::-1], 'svr_tfl_grw_r2_0':svr_tfl_grw_r2_0[::-1], 'svr_tfl_grw_prs_0':svr_tfl_grw_prs_0[::-1]})
    svr_tfl_grw_0.set_index(pd.Index(p_value_2), 'p_value', inplace = True)
    svr_tfl_grw_0.to_csv('svr_tfl_grw_0.csv')
    
    # Output is grain weight
    svr_tfl_grw_mse_2_1, svr_tfl_grw_r2_2_1, svr_tfl_grw_prs_2_1 = get_svm_model(X_tfl_grw_2_train, y_tfl_grw_2_train[:, 1], X_tfl_grw_2_test, y_tfl_grw_2_test[:, 1])
    svr_tfl_grw_mse_1.append(svr_tfl_grw_mse_2_1)
    svr_tfl_grw_r2_1.append(svr_tfl_grw_r2_2_1)
    svr_tfl_grw_prs_1.append(svr_tfl_grw_prs_2_1)
    
    svr_tfl_grw_mse_25_1, svr_tfl_grw_r2_25_1, svr_tfl_grw_prs_25_1 = get_svm_model(X_tfl_grw_25_train, y_tfl_grw_25_train[:, 1], X_tfl_grw_25_test, y_tfl_grw_25_test[:, 1])
    svr_tfl_grw_mse_1.append(svr_tfl_grw_mse_25_1)
    svr_tfl_grw_r2_1.append(svr_tfl_grw_r2_25_1)
    svr_tfl_grw_prs_1.append(svr_tfl_grw_prs_25_1)
    
    svr_tfl_grw_mse_1_1, svr_tfl_grw_r2_1_1, svr_tfl_grw_prs_1_1 = get_svm_model(X_tfl_grw_1_train, y_tfl_grw_1_train[:, 1], X_tfl_grw_1_test, y_tfl_grw_1_test[:, 1])
    svr_tfl_grw_mse_1.append(svr_tfl_grw_mse_1_1)
    svr_tfl_grw_r2_1.append(svr_tfl_grw_r2_1_1)
    svr_tfl_grw_prs_1.append(svr_tfl_grw_prs_1_1)
    
    svr_tfl_grw_mse_75_1, svr_tfl_grw_r2_75_1, svr_tfl_grw_prs_75_1 = get_svm_model(X_tfl_grw_75_train, y_tfl_grw_75_train[:, 1], X_tfl_grw_75_test, y_tfl_grw_75_test[:, 1])
    svr_tfl_grw_mse_1.append(svr_tfl_grw_mse_75_1)
    svr_tfl_grw_r2_1.append(svr_tfl_grw_r2_75_1)
    svr_tfl_grw_prs_1.append(svr_tfl_grw_prs_75_1)
    
    svr_tfl_grw_mse_3_1, svr_tfl_grw_r2_3_1, svr_tfl_grw_prs_3_1 = get_svm_model(X_tfl_grw_3_train, y_tfl_grw_3_train[:, 1], X_tfl_grw_3_test, y_tfl_grw_3_test[:, 1])
    svr_tfl_grw_mse_1.append(svr_tfl_grw_mse_3_1)
    svr_tfl_grw_r2_1.append(svr_tfl_grw_r2_3_1)
    svr_tfl_grw_prs_1.append(svr_tfl_grw_prs_3_1)
    
    svr_tfl_grw_1 = pd.DataFrame({'svr_tfl_grw_mse_1':svr_tfl_grw_mse_1[::-1], 'svr_tfl_grw_r2_1':svr_tfl_grw_r2_1[::-1], 'svr_tfl_grw_prs_1':svr_tfl_grw_prs_1[::-1]})
    svr_tfl_grw_1.set_index(pd.Index(p_value_2), 'p_value', inplace = True)
    svr_tfl_grw_1.to_csv('svr_tfl_grw_1.csv')
    print("")
    print("Create data frames...")
    print("")
    grw_mse = pd.DataFrame({'rf_grw_mse':rf_grw_mse[::-1], 'svr_grw_mse':svr_grw_mse[::-1]})
    grw_mse.set_index(pd.Index(p_value[1:5]), 'p_value', inplace = True)
    
    grw_r2 = pd.DataFrame({'rf_grw_r2':rf_grw_r2[::-1], 'svr_grw_r2':svr_grw_r2[::-1]})
    grw_r2.set_index(pd.Index(p_value[1:5]), 'p_value', inplace = True)
    
    tfl_mse = pd.DataFrame({'rf_tfl_mse':rf_tfl_mse[::-1], 'svr_tfl_mse':svr_tfl_mse[::-1]})
    tfl_mse.set_index(pd.Index(p_value), 'p_value', inplace = True)
    
    tfl_r2 = pd.DataFrame({'rf_tfl_r2':rf_tfl_r2[::-1], 'svr_tfl_r2':svr_tfl_r2[::-1]})
    tfl_r2.set_index(pd.Index(p_value), 'p_value', inplace = True)
    
    tfl_grw_mse = pd.DataFrame({'rf_tfl_mse':rf_tfl_grw_mse_0[::-1], 'rf_grw_mse':rf_tfl_grw_mse_1[::-1], 'svr_tfl_mse':svr_tfl_grw_mse_0[::-1], 'svr_grw_mse':svr_tfl_grw_mse_1[::-1]})
    tfl_grw_mse.set_index(pd.Index(p_value_2), 'p_value', inplace = True)
    
    tfl_grw_r2 = pd.DataFrame({'rf_tfl_r2':rf_tfl_grw_r2_0[::-1], 'rf_grw_r2':rf_tfl_grw_r2_1[::-1], 'svr_tfl_r2':svr_tfl_grw_r2_0[::-1], 'svr_grw_r2':svr_tfl_grw_r2_1[::-1]})
    tfl_grw_r2.set_index(pd.Index(p_value_2), 'p_value', inplace = True)
    
    print("")
    print("Find mse and r-squared for lasso and multitasklasso model...")
    print("")
    print("For grain weight...")
    print("")
    mse_grw_ls, r2_grw_ls, var_grw_ls, ls_grw_prs = get_lasso_cv(X_grw_train, y_grw_train, X_grw_test, y_grw_test, cols)
    print("")
    print("For time to flowering...")
    print("")
    mse_tfl_ls, r2_tfl_ls, var_tfl_ls, ls_tfl_prs = get_lasso_cv(X_tfl_train, y_tfl_train, X_tfl_test, y_tfl_test, cols)
    print("")
    print("For time to flowering and grain weight...")
    print("")
    mse_tfl_grw_ls, r2_tfl_grw_ls, var_tfl_grw_ls, ls_tfl_grw_prs = get_multitask_lasso_cv(X_tfl_grw_train, y_tfl_grw_train, X_tfl_grw_test, y_tfl_grw_test, cols)
    print("")
    print("Find mse and r-squared for elasticnet and multitaskelasticnet model...")
    print("")
    print("For grain weight...")
    print("")
    
    mse_grw_el, r2_grw_el, var_grw_el, el_grw_prs = get_elasticnet_cv(X_grw_train, y_grw_train, X_grw_test, y_grw_test, cols)
    print("")
    print("For time to flowering...")
    print("")
    mse_tfl_el, r2_tfl_el, var_tfl_el, el_tfl_prs  = get_elasticnet_cv(X_tfl_train, y_tfl_train, X_tfl_test, y_tfl_test, cols)
    print("")
    print("For time to flowering and grain weight...")
    print("")
    mse_tfl_grw_el, r2_tfl_grw_el, var_tfl_grw_el, el_tfl_grw_prs = get_multitask_elasticnet_cv(X_tfl_grw_train, y_tfl_grw_train, X_tfl_grw_test, y_tfl_grw_test, cols)
    
    # Mse, r2 of each trait with the multi-task problem
    mtls_mse_tfl, mtls_mse_grw, mtls_r2_tfl, mtls_r2_grw = eval_mtls_split_trait(2.41812258083032, X_tfl_grw_train, y_tfl_grw_train, X_tfl_grw_test, y_tfl_grw_test)
    mtel_mse_tfl, mtel_mse_grw, mtel_r2_tfl, mtel_r2_grw = eval_mtel_split_trait(4.20631940576943, 0.5, X_tfl_grw_train, y_tfl_grw_train, X_tfl_grw_test, y_tfl_grw_test)

    ls_table = pd.DataFrame({'mse_grw_ls':[mse_grw_ls], 'r2_grw_ls':[r2_grw_ls], 
                             'mse_tfl_ls':[mse_tfl_ls], 'r2_tfl_ls':[r2_tfl_ls],
                             'mse_tfl_grw_ls':[mse_tfl_grw_ls], 'r2_tfl_grw_ls':[r2_tfl_grw_ls],
                             'ls_grw_prs':[ls_grw_prs], 'ls_tfl_prs':[ls_tfl_prs], 'ls_tfl_grw_prs':[ls_tfl_grw_prs]})
    el_table = pd.DataFrame({'mse_grw_el':[mse_grw_el], 'r2_grw_el':[r2_grw_el], 
                             'mse_tfl_el':[mse_tfl_el], 'r2_tfl_el':[r2_tfl_el],
                             'mse_tfl_grw_el':[mse_tfl_grw_el], 'r2_tfl_grw_el':[r2_tfl_grw_el],
                             'el_grw_prs':[el_grw_prs], 'el_tfl_prs':[el_tfl_prs], 'el_tfl_grw_prs':[el_tfl_grw_prs]})
    
    ls_split_trait = pd.DataFrame({'mtls_mse_tfl':[mtls_mse_tfl],'mtls_mse_grw':[mtls_mse_grw], 'mtls_r2_tfl':[mtls_r2_tfl], 'mtls_r2_grw':[mtls_r2_grw]})
    el_split_trait = pd.DataFrame({'mtel_mse_tfl':[mtel_mse_tfl],'mtel_mse_grw':[mtel_mse_grw], 'mtel_r2_tfl':[mtel_r2_tfl], 'mtel_r2_grw':[mtel_r2_grw]})
      
    var_tfl_ls = pd.DataFrame({'var_tfl_ls':var_tfl_ls})
    var_grw_ls = pd.DataFrame({'var_grw_ls':var_grw_ls})
    var_tfl_grw_ls = pd.DataFrame({'var_tfl_grw_ls':var_tfl_grw_ls})
    
    var_tfl_el = pd.DataFrame({'var_tfl_el':var_tfl_el})
    var_grw_el = pd.DataFrame({'var_grw_el':var_grw_el})
    var_tfl_grw_el = pd.DataFrame({'var_tfl_grw_el':var_tfl_grw_el})
    
    print("")
    print("*********************************** RESULTS *********************************")
    print("")
    print("Grain weight mean squared error:\n", grw_mse)
    print("Grain weight r-squared:\n", grw_r2)
    print("Time to flowering mean squared error:\n", tfl_mse)
    print("Time to flowering r-squared:\n", tfl_r2)
    print("Time to flowering and grain weight mean squared error:\n", tfl_grw_mse)
    print("Time to flowering and grain weight r-squared:\n", tfl_grw_r2)
    print("Lasso for grain weight:\n", "mse:", mse_grw_ls, "r2:", r2_grw_ls)
    print("Lasso for time to flowering:\n", "mse:", mse_tfl_ls, "r2:", r2_tfl_ls)
    print("MultiTaskLasso for time to flowering and grain weight:\n", "mse:", mse_tfl_grw_ls, "r2:", r2_tfl_grw_ls)
    print("ElasticNet for grain weight:\n", "mse:", mse_grw_el, "r2:", r2_grw_el)
    print("ElasticNet for time to flowering:\n", "mse:", mse_tfl_el, "r2:", r2_tfl_el)
    print("MultiTaskElasticNet for time to flowering and grain weight:\n", "mse:", mse_tfl_grw_el, "r2:", r2_tfl_grw_el)
    print("")
    print("")
    print("********************************** SAVING **********************************")
    print("")
    
    grw_mse.to_csv('grw_mse.csv')
    grw_r2.to_csv('grw_r2.csv')
    tfl_mse.to_csv('tfl_mse.csv')
    tfl_r2.to_csv('tfl_r2.csv')
    tfl_grw_mse.to_csv('tfl_grw_mse.csv')
    tfl_grw_r2.to_csv('tfl_grw_r2.csv')
    
    ls_table.to_csv('ls_table.csv')
    el_table.to_csv('el_table.csv')
    ls_split_trait.to_csv('ls_split_trait.csv')
    el_split_trait.to_csv('el_split_trait.csv')
    
    var_tfl_ls.to_csv('var_tfl_ls.csv')
    var_grw_ls.to_csv('var_grw_ls.csv')
    var_tfl_grw_ls.to_csv('var_tfl_grw_ls.csv')
    var_tfl_el.to_csv('var_tfl_el.csv')
    var_grw_el.to_csv('var_grw_el.csv')
    var_tfl_grw_el.to_csv('var_tfl_grw_el.csv')

    print("")
    print("********************************* FINISHED *********************************")
    print("")