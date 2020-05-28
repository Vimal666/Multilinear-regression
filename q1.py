# -*- coding: utf-8 -*-
"""
Created on Wed May 20 09:54:25 2020

@author: intel
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from ml_metrics import rmse
 
startups=pd.read_csv("D:\DATA SCIENCE\ASSIGNMENT\ASSIGNMENT5\startups.csv")
#Label encoder() using for levels of categorical features into numerical values
Le = preprocessing.LabelEncoder()
startups['st'] = Le.fit_transform(startups['St'])
startups = startups.drop('St',axis = 1)
startups.columns
startups
startups.describe()
startups.columns
startups.corr()
n_startups = preprocessing.normalize(startups)
plt.hist(startups.Rd)
plt.hist(startups.Ad)
plt.hist(startups.Pr)
plt.hist(startups.Ms)
plt.hist(startups.st)
sns.pairplot(startups)
import statsmodels.formula.api as smf
model1=smf.ols("Pr~Rd+Ad+Ms+st",data=startups).fit()
model1.params
model1.summary()
pred=model1.predict(startups)
pred
import statsmodels.formula.api as smf
mod1_Ad = smf.ols('Pr~Ad',data=startups).fit()
mod1_Ad.summary()
mod1_Ms = smf.ols('Pr~Ms',data=startups).fit()
mod1_Ms.summary()
mod1_st = smf.ols('Pr~st',data=startups).fit()
mod1_st.summary()
import statsmodels.api as sm
sm.graphics.influence_plot(model1)
startups_new = startups.drop(startups.index[[19,46,48,49]],axis=0)
model2=smf.ols("Pr~Rd+Ad+Ms",data=startups_new).fit()
model2.summary()
pred2=model2.predict(startups_new)
pred2
startups_new
startups_new.head()
rsq_Rd = smf.ols("Rd~Ad+Ms",data=startups_new).fit().rsquared  
vif_Rd = 1/(1-rsq_Rd) 
rsq_Ad = smf.ols("Ad~Ms+Rd",data=startups_new).fit().rsquared  
vif_Ad = 1/(1-rsq_Ad) 
rsq_Ms = smf.ols("Ms~Ad+Rd",data=startups_new).fit().rsquared  
vif_Ms = 1/(1-rsq_Ms) 

d1 = {'Variables':['Rd','Ad','Ms',],'VIF':[vif_Rd,vif_Ad,vif_Ms,]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame

modelA=smf.ols("Pr~Ad",data=startups).fit()
modelA.summary()
modelM=smf.ols("Pr~Ms",data=startups).fit()
modelM.summary()
modelAM=smf.ols("Pr~Ad+Ms",data=startups).fit()
modelAM.summary()
sm.graphics.plot_partregress_grid(model2)
model3=smf.ols("Pr~Rd+Ms",data=startups_new).fit()
model3.params
model3.summary()
import statsmodels.api as sm
sm.graphics.plot_partregress_grid(model3)
pred3=model3.predict(startups_new)
pred
rootmse = rmse(pred3,startups_new.Pr)
rootmse
Actual=startups_new.Pr
df = pd.DataFrame(list(zip(pred3, Actual)),columns =['Predicted Prices', 'Actual Prices'])
values = list([model1.rsquared,model2.rsquared,model3.rsquared])
coded_variables = list(['model1.rsquared','model2.rsquared','model3.rsquared'])
variables = list(['Model 1','Model 2','Model 3'])
Rsquared_model = pd.DataFrame(list(zip(variables,coded_variables,values)),columns = ['Models','Variabels Named in the code','R^Squared Values'])
Rsquared_model
