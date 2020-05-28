# -*- coding: utf-8 -*-
"""
Created on Wed May 27 20:02:32 2020

@author: intel
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from ml_metrics import rmse
Computerdata=pd.read_csv("D:\DATA SCIENCE\ASSIGNMENT\ASSIGNMENT5\Computerdata.csv")
Le = preprocessing.LabelEncoder()
Computerdata['Cd'] = Le.fit_transform(Computerdata['cd'])
Computerdata = Computerdata.drop('cd',axis = 1)
Computerdata['Multi'] = Le.fit_transform(Computerdata['multi'])
Computerdata = Computerdata.drop('multi',axis = 1)
Computerdata['Premium'] = Le.fit_transform(Computerdata['premium'])
Computerdata = Computerdata.drop('premium',axis = 1)
Computerdata.describe()
sns.pairplot(Computerdata)
Computerdata.columns
Computerdata.corr()
import statsmodels.formula.api as smf
Model=smf.ols("price~speed+hd+ram+screen+ads+trend+Cd+Multi+Premium",data=Computerdata).fit()
Model.params
Model.summary()
Pred=Model.predict(Computerdata)
Pred
