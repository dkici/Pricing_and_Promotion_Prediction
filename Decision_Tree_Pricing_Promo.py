# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 13:12:51 2022

@author: DKici
"""

# %% libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


# %% read csv
#data = pd.read_csv("data.csv")
#data.drop(["Unnamed: 32","id"],axis=1,inplace = True)
#data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
#print(data.info())

data = pd.read_csv('C:\\Users\\DKici\\Documents\\PricingPromo\\data\\pricing_promo_2019_2021_all.csv')
data = data.drop(columns = ["Unnamed: 0", "level_0"], axis = 1)
print(data.head())

print(data.columns)

data["Traffic"] = pd.to_numeric(data["Traffic"], errors='coerce') 
# print(data.info())


#add a new column category next to the Traffic. 
bins=list(range(-1,int(data["Traffic"].max()+100000), 100000))
names = list(range(0, len(bins)-1))


data["Range"] = pd.cut(data["Traffic"], bins, labels=names)

data["Range"] = pd.Categorical(data["Range"]) 
print(data["Range"].unique())

# print("NULL",data.isna().sum().sum())
# data = data.dropna(axis = 1)

y = data.Range.values

x_data = data.drop(["Date","Traffic","Margin", "WrittenSales","FinancedAmount","Range"],axis=1)
# print(x_data)

# normalization
x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))

# train test split
from sklearn.model_selection import train_test_split
x_train, x_test,y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state = 42)

#
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)

print("score: ", dt.score(x_test,y_test))















