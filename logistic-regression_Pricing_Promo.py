# -*- coding: utf-8 -*-
"""
@author: user
"""

# %% libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('C:\\Users\\DKici\\Documents\\PricingPromo\\data\\pricing_promo_2019_2021_all.csv')
data = data.drop(columns = "Unnamed: 0")
# print(data.head())


data["Traffic"] = pd.to_numeric(data["Traffic"], errors='coerce') 
# print(data.info())


#add a new column category next to the Traffic. 
bins=list(range(-1,int(data["Traffic"].max()+10000000), 100000))
names = list(range(0, len(bins)-1))


data["Range"] = pd.cut(data["Traffic"], bins, labels=names)

data["Range"] = pd.Categorical(data["Range"]) 
print(data["Range"].unique())



# print("NULL",data.isna().sum().sum())
# data = data.dropna(axis = 1)



y = data.Range.values

x_data = data.drop(["Date","Traffic","Margin", "WrittenSales","FinancedAmount","Range"],axis=1)
# print(x_data)

# # %% normalization
x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data)).values



# # %% train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=42)


x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T

# print("x_train: ",x_train.shape)
# print("x_test: ",x_test.shape)
# print("y_train: ",y_train.shape)
# print("y_test: ",y_test.shape)



# sklearn with Logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train.T,y_train.T)
y_pred = lr.predict(x_test.T)

from sklearn.metrics import r2_score
print("r_square_score for Traffic:", r2_score(y_test, y_pred))

# print("test accuracy {}".format(lr.score(x_test.T,y_test.T)))



#%%