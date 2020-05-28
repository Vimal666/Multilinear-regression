# -*- coding: utf-8 -*-
"""
Created on Wed May 27 20:02:32 2020

@author: Vimal PM
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from ml_metrics import rmse
Computerdata=pd.read_csv("D:\DATA SCIENCE\ASSIGNMENT\ASSIGNMENT5\Computerdata.csv")
Le = preprocessing.LabelEncoder() ##Label encoder() using for levels of categorical features into numerical values
Computerdata['Cd'] = Le.fit_transform(Computerdata['cd'])
Computerdata = Computerdata.drop('cd',axis = 1)
Computerdata['Multi'] = Le.fit_transform(Computerdata['multi'])
Computerdata = Computerdata.drop('multi',axis = 1)
Computerdata['Premium'] = Le.fit_transform(Computerdata['premium'])
Computerdata = Computerdata.drop('premium',axis = 1)
Computerdata.describe()
sns.pairplot(Computerdata)
Computerdata.columns
Computerdata.corr()#Correlation of coeficent
import statsmodels.formula.api as smf
#Building my model
#To predict the price of computers,here I'm  adding speed+hd+ram+screen+ads+trend+Cd+Multi+Premium against the Price
Model=smf.ols("price~speed+hd+ram+screen+ads+trend+Cd+Multi+Premium",data=Computerdata).fit()
Model.params
Model.summary()
#From my first model I got each and every variables as significant which means P-value less than 0.05 
#Here i'm predicting the price of computers from my model
Pred=Model.predict(Computerdata)
Pred
0       2020.518889
1       2002.478116
2       2213.968113
3       2793.127639
4       2877.415391
    
6254    1586.853395
6255    2072.985141
6256    2945.221470
6257    2285.550870
6258    2531.728954
#Above shows the predicted values for first five rows and last 5 rows