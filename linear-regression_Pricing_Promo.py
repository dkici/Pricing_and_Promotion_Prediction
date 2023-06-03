# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 14:42:09 2022

@author: DKici
"""


# %% libraries
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import scale
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline

# %% read csv
#data = pd.read_csv("data.csv")
#data.drop(["Unnamed: 32","id"],axis=1,inplace = True)
#data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
#print(data.info())

data = pd.read_csv('C:\\Users\\DKici\\Documents\\PricingPromo\\data\\pricing_promo_2019_2021_all.csv')
data = data.drop(columns = ["Unnamed: 0", "level_0", "Date"], axis = 1)
# print(data.head())

# %% Predict Traffic
x = data.iloc[:,1:-4].values
y = data.Traffic.values.reshape(-1,1)

# train test split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=42)

# step-1: create a cross-validation scheme
folds = KFold(n_splits = 5, shuffle = True, random_state = 100)

# step-2: specify range of hyperparameters to tune
hyper_params = [{'n_features_to_select': list(range(1, 107))}]


# step-3: perform grid search
# 3.1 specify model
lm = LinearRegression()
lm.fit(x_train, y_train)
rfe = RFE(lm)             

# 3.2 call GridSearchCV()`
model_cv = GridSearchCV(estimator = rfe, 
                        param_grid = hyper_params, 
                        scoring= 'r2', 
                        cv = folds, 
                        verbose = 1,
                        return_train_score=True)      

# fit the model
model_cv.fit(x_train, y_train)                  

cv_results = pd.DataFrame(model_cv.cv_results_)
# print(cv_results)

# plotting cv results
plt.figure(figsize=(16,6))

plt.plot(cv_results["param_n_features_to_select"], cv_results["mean_test_score"])
plt.plot(cv_results["param_n_features_to_select"], cv_results["mean_train_score"])
plt.xlabel('number of features')
plt.ylabel('r-squared')
plt.title("Optimal Number of Features")
plt.legend(['test score', 'train score'], loc='upper left')

# final model
n_features_optimal = 45

lm = LinearRegression()
lm.fit(x_train, y_train)

rfe = RFE(lm, n_features_to_select=n_features_optimal)             
rfe = rfe.fit(x_train, y_train)

# predict prices of X_test
y_pred = lm.predict(x_test)
r2 = sklearn.metrics.r2_score(y_test, y_pred)
print("r_square_score for Traffic Cross-Validation:", r2)


# %% Predict Traffic
x = data.iloc[:,1:-4].values
y = data.Traffic.values.reshape(-1,1)

#  train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=42)


# fitting data
multiple_linear_regression = LinearRegression()
multiple_linear_regression.fit(x_train,y_train)

# print("b0: ", multiple_linear_regression.intercept_)
# print("bi: ",multiple_linear_regression.coef_)


# predict
y_pred = multiple_linear_regression.predict(x_test)

from sklearn.metrics import r2_score
print("r_square_score for Traffic:", r2_score(y_test, y_pred))

# R square for Traffic is pretty low: 0.399

# %%  Written Sales - CV

x = data.iloc[:,1:-4].values
y = data.WrittenSales.values.reshape(-1,1)

# train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=42)


# print("x_train: ",x_train.shape)
# print("x_test: ",x_test.shape)
# print("y_train: ",y_train.shape)
# print("y_test: ",y_test.shape)

# step-1: create a cross-validation scheme
folds = KFold(n_splits = 5, shuffle = True, random_state = 100)

# step-2: specify range of hyperparameters to tune
hyper_params = [{'n_features_to_select': list(range(1, 107))}]


# step-3: perform grid search
# 3.1 specify model
lm = LinearRegression()
lm.fit(x_train, y_train)
rfe = RFE(lm)             

# 3.2 call GridSearchCV()`
model_cv = GridSearchCV(estimator = rfe, 
                        param_grid = hyper_params, 
                        scoring= 'r2', 
                        cv = folds, 
                        verbose = 1,
                        return_train_score=True)      

# fit the model
model_cv.fit(x_train, y_train)                  

cv_results = pd.DataFrame(model_cv.cv_results_)
# print(cv_results)

# plotting cv results
plt.figure(figsize=(16,6))

plt.plot(cv_results["param_n_features_to_select"], cv_results["mean_test_score"])
plt.plot(cv_results["param_n_features_to_select"], cv_results["mean_train_score"])
plt.xlabel('number of features')
plt.ylabel('r-squared')
plt.title("Optimal Number of Features")
plt.legend(['test score', 'train score'], loc='upper left')

# final model
n_features_optimal = 25

lm = LinearRegression()
lm.fit(x_train, y_train)

rfe = RFE(lm, n_features_to_select=n_features_optimal)             
rfe = rfe.fit(x_train, y_train)

# predict prices of X_test
y_pred = lm.predict(x_test)
r2 = sklearn.metrics.r2_score(y_test, y_pred)
print("r_square_score for Written Sales Cross-Validation:", r2)

# %% Predict Written Sales

x = data.iloc[:,1:-4].values
y = data.WrittenSales.values.reshape(-1,1)
#  train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=42)


#  fitting data
multiple_linear_regression = LinearRegression()
multiple_linear_regression.fit(x_train,y_train)

# print("b0: ", multiple_linear_regression.intercept_)
# print("bi: ",multiple_linear_regression.coef_)


# predict
y_pred = multiple_linear_regression.predict(x_test)

from sklearn.metrics import r2_score
print("r_square_score for written sales:", r2_score(y_test, y_pred))


# R square for Written sales is pretty low too: 0.298

# %%  Margin - CV

x = data.iloc[:,1:-4].values
y = data.Margin.values.reshape(-1,1)

# train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=42)

# step-1: create a cross-validation scheme
folds = KFold(n_splits = 5, shuffle = True, random_state = 100)

# step-2: specify range of hyperparameters to tune
hyper_params = [{'n_features_to_select': list(range(1, 107))}]

# step-3: perform grid search
# 3.1 specify model
lm = LinearRegression()
lm.fit(x_train, y_train)
rfe = RFE(lm)             

# 3.2 call GridSearchCV()`
model_cv = GridSearchCV(estimator = rfe, 
                        param_grid = hyper_params, 
                        scoring= 'r2', 
                        cv = folds, 
                        verbose = 1,
                        return_train_score=True)      

# fit the model
model_cv.fit(x_train, y_train)                  

cv_results = pd.DataFrame(model_cv.cv_results_)
# print(cv_results)

# plotting cv results
plt.figure(figsize=(16,6))

plt.plot(cv_results["param_n_features_to_select"], cv_results["mean_test_score"])
plt.plot(cv_results["param_n_features_to_select"], cv_results["mean_train_score"])
plt.xlabel('number of features')
plt.ylabel('r-squared')
plt.title("Optimal Number of Features")
plt.legend(['test score', 'train score'], loc='upper left')

# final model
n_features_optimal = 25

lm = LinearRegression()
lm.fit(x_train, y_train)

rfe = RFE(lm, n_features_to_select=n_features_optimal)             
rfe = rfe.fit(x_train, y_train)

# predict prices of X_test
y_pred = lm.predict(x_test)
r2 = sklearn.metrics.r2_score(y_test, y_pred)
print("r_square_score for Margin Cross-Validation:", r2)

# %% Predict Margin
x = data.iloc[:,1:-4].values
y = data.Margin.values.reshape(-1,1)

# %% train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=42)


# fitting data
multiple_linear_regression = LinearRegression()
multiple_linear_regression.fit(x_train,y_train)

# print("b0: ", multiple_linear_regression.intercept_)
# print("bi: ",multiple_linear_regression.coef_)


# predict
y_pred = multiple_linear_regression.predict(x_test)

from sklearn.metrics import r2_score
print("r_square_score for Margin:", r2_score(y_test, y_pred))


# R square for Margin is: 0.292


# %% Predict Financed Amount
x = data.iloc[:,1:-4].values
y = data.FinancedAmount.values.reshape(-1,1)

# %% train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=42)


# print("x_train: ",x_train.shape)
# print("x_test: ",x_test.shape)
# print("y_train: ",y_train.shape)
# print("y_test: ",y_test.shape)

# step-1: create a cross-validation scheme
folds = KFold(n_splits = 5, shuffle = True, random_state = 100)

# step-2: specify range of hyperparameters to tune
hyper_params = [{'n_features_to_select': list(range(1, 107))}]


# step-3: perform grid search
# 3.1 specify model
lm = LinearRegression()
lm.fit(x_train, y_train)
rfe = RFE(lm)             

# 3.2 call GridSearchCV()`
model_cv = GridSearchCV(estimator = rfe, 
                        param_grid = hyper_params, 
                        scoring= 'r2', 
                        cv = folds, 
                        verbose = 1,
                        return_train_score=True)      

# fit the model
model_cv.fit(x_train, y_train)                  

cv_results = pd.DataFrame(model_cv.cv_results_)
# print(cv_results)

# plotting cv results
plt.figure(figsize=(16,6))

plt.plot(cv_results["param_n_features_to_select"], cv_results["mean_test_score"])
plt.plot(cv_results["param_n_features_to_select"], cv_results["mean_train_score"])
plt.xlabel('number of features')
plt.ylabel('r-squared')
plt.title("Optimal Number of Features")
plt.legend(['test score', 'train score'], loc='upper left')

# final model
n_features_optimal = 25

lm = LinearRegression()
lm.fit(x_train, y_train)

rfe = RFE(lm, n_features_to_select=n_features_optimal)             
rfe = rfe.fit(x_train, y_train)

# predict prices of X_test
y_pred = lm.predict(x_test)
r2 = sklearn.metrics.r2_score(y_test, y_pred)
print("r_square_score for Financed Amount Cross-Validation:", r2)



# fitting data
multiple_linear_regression = LinearRegression()
multiple_linear_regression.fit(x_train,y_train)

# print("b0: ", multiple_linear_regression.intercept_)
# print("bi: ",multiple_linear_regression.coef_)


# predict
y_pred = multiple_linear_regression.predict(x_test)

from sklearn.metrics import r2_score
print("r_square_score for Financed Amount:", r2_score(y_test, y_pred))


# R square for FinancedAmount is: 0.244



