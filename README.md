# Data-Analysis-Airbnb-NYC
Exploratory analysis and regression modeling

# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 21:13:38 2019

@author: Jessiezhao
"""
# install data and import the file for analysis

import pandas as pd

data=pd.read_csv("AB_NYC_2019.csv")

# Data Exploration

data.shape

data.head()

data.info
# list all the columns' name
data.columns
# summary()
data.describe()
# binary results of each column
data.isnull().any()
# % of null of each column
data.isnull().sum()/data.shape[0]
# show data types, generally: int64, float64 and object
data.dtypes

data.room_type.unique()
# value_counts count how many unique values
data.neighbourhood.value_counts()/data.neighbourhood.notnull().sum()
# start draw graphs
# first, peek at the counts and set expectation for how the graph gonna look like
data.neighbourhood_group.unique()
data.neighbourhood_group.value_counts()
#plot the bar chart
data.neighbourhood_group.value_counts().plot(kind='bar')

#data manipulation
data['id']
data.info
data[0:10]

data_manhattan=data[data['neighbourhood_group']=='Manhattan']
data_manhattan.price.hist(bins=100)
data_manhattan.price.hist(bins=1000)
data_manhattan.price.describe()
data.price.describe()
data_brooklyn=data[data['neighbourhood_group']=='Brooklyn']
data_brooklyn.price.describe()

data_manhattan.shape[0]
data.neighbourhood_group.value_counts()/data.neighbourhood_group.notnull().sum()

196-124

data.room_type.value_counts().plot(kind='bar')
data_manhattan.room_type.value_counts().plot(kind='bar')
data_brooklyn.room_type.value_counts().plot(kind='bar')

#linear regression of price and other factor for prediction 
import seaborn as sb
import numpy as np

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, RidgeCV, LassoCV, ElasticNetCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import random

#drop irrelated data
data_1=data.drop(['id','name','host_id','last_review','host_name','reviews_per_month'], axis=1)

#check null data
data_1.isnull().sum()
data_1.isnull().sum()/ data_1.shape[0]

import matplotlib.pyplot as plt



#check the distribution of price 
plt.figure(figsize=(20,6))
sb.distplot(data_1['price'], rug=True)
#show the histagram of price and box plot, find a lot outliers
data_1.price.hist(bins=1000)
data_1.boxplot(column='price', by='room_type', figsize=(20,6), rot=90)
data_manhattan.boxplot(column='price', by='room_type', figsize=(20,6), rot=90)

data_1.boxplot(column='price', by='neighbourhood_group', figsize=(20,6), rot=90)
#use the logitude nad latitude information to show price range on map
data_1.plot.scatter(x='longitude', y='latitude', c='price', figsize=(20,20), cmap='cool', alpha=0.5);

#show price udner 200(the majority of price, seen from boxplot)
data_1[data_1['price']<200].plot.scatter(x='longitude', y='latitude', c='price', figsize=(20,20), cmap='cool', alpha=0.5);

#check the relation of number_of_reviews and price
data_1.plot.scatter(x='number_of_reviews', y='price',figsize=(20,8));
data.price.describe()

#remove outliners for further analysis
data_outliers = data_1[(data_1.price < data_1.price.quantile(.995)) & (data_1.price > data_1.price.quantile(.005))]
data_outliers.boxplot('price')
data_outliers.boxplot(column='price', by='neighbourhood_group', figsize=(20,6), rot=90)
data_outliers.hist('price')

#change columns with object values to dummy variables
data_dummies = pd.get_dummies(data_outliers)
data_dummies.head()
data_dummies.isnull().any()
data_dummies.dropna()
data_dummies.columns
#set X and y for prediction model 
X = data_dummies.copy().drop('price', axis = 1)
y = data_dummies['price'].copy()

#split data in training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y)


#scale data to have similar dimensions
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

#delete host_name from the dataset as it creates too many dummies(this step is done in original comamnd)

#baseline: median and error
baseline=y_train.median()
baseline_error = np.sqrt(mean_squared_error(y_pred=np.ones_like(y_test) * baseline, y_true=y_test))

#Machine learning

lr = LinearRegression()
alphas = [1000, 100, 50, 20, 10, 1, 0.1, 0.01]
l1_ratios = [0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9]
ridge = RidgeCV(alphas=alphas)
lasso = LassoCV(alphas=alphas, max_iter=10000)
elastic = ElasticNetCV(alphas=alphas, l1_ratio=l1_ratios)

for model, name in zip([lr, ridge, lasso, elastic], ['LinearRegression', 'Ridge', 'Lasso', 'ElasticNet']):
    model.fit(X_train_scaled, y_train)
    y_pred_train = model.predict(X_train_scaled)
    mrse_train = np.sqrt(mean_squared_error(y_pred=y_pred_train, y_true=y_train))
    y_pred = model.predict(X_test_scaled)
    mrse_test = np.sqrt(mean_squared_error(y_pred=y_pred, y_true=y_test))
    best_alpha = ''
    if name != 'LinearRegression':
        best_alpha = ' best alpha: ' + str(model.alpha_)
    best_l1 = ''
    if name == 'ElasticNet':
        best_l1 = ' best l1: '+ str(model.l1_ratio_)
    print(name + ' mrse_train: ' + str(mrse_train) + ', mrse_test: ' + str(mrse_test) + best_alpha + best_l1)

#Non-zero Lasso coefficients ordered by importance--this part has errors
    order = np.argsort(np.abs(lasso.coef_))[::-1]
for i in order:
    coef_ = lasso.coef_[i]
    if coef_ > 0:
        print(X.columns[i] + ', ' + str(lasso.coef_[i]))
        
#use polynomial regression--this part works
pf = PolynomialFeatures(degree=2)
pf.fit(X_train_scaled)
X_train_scaled = pf.transform(X_train_scaled)
X_test_scaled = pf.transform(X_test_scaled)

for model, name in zip([lr, ridge, lasso], ['LinearRegression', 'Ridge', 'Lasso']):
    model.fit(X_train_scaled, y_train)
    y_pred_train = model.predict(X_train_scaled)
    mrse_train = np.sqrt(mean_squared_error(y_pred=y_pred_train, y_true=y_train))
    y_pred = model.predict(X_test_scaled)
    mrse_test = np.sqrt(mean_squared_error(y_pred=y_pred, y_true=y_test))
    print(name + ' mrse_train: ' + str(mrse_train) + ', mrse_test: ' + str(mrse_test))  

poly_features = pf.get_feature_names(X_train.columns)
order = np.argsort(np.abs(lasso.coef_))[::-1]
for i in order:
    coef_ = lasso.coef_[i]
    if coef_ > 0:
        print(poly_features[i] + ', ' + str(lasso.coef_[i]))
        

difference_100=X_train[np.abs(diff) > 100].describe()

#merge y_pred and y_test to draw a graph to show comparison
error_diff = pd.DataFrame({'actual': np.array(y_test).flatten(), 'predicted': y_pred.flatten()})
error_diff1 = error_diff.head(20)
error_diff1.columns

# this part is for drwing the comparison graph but the graph doesn't show.
import plotly as plotly
import plotly.graph_objects as go
title=['Pred vs Actual']
fig = go.Figure(
        data=[go.Bar(name='Predicted',x=error_diff1.index, y=error_diff1['predicted']),
              go.Bar(name='Actual', x=error_diff1.index, y=error_diff1['actual'])
])

fig.update_layout(barmode='group')
fig.show()





      
