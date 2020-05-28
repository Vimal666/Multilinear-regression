# -*- coding: utf-8 -*-
"""
Created on Wed May 20 09:54:25 2020

@author: Vimal PM
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
startups.columns #Index(['Rd', 'Ad', 'Ms', 'Pr', 'st'], dtype='object')
startups
startups.describe()
 Rd             Ad             Ms             Pr         st
count      50.000000      50.000000      50.000000      50.000000  50.000000
mean    73721.615600  121344.639600  211025.097800  112012.639200   1.000000
std     45902.256482   28017.802755  122290.310726   40306.180338   0.832993
min         0.000000   51283.140000       0.000000   14681.400000   0.000000
25%     39936.370000  103730.875000  129300.132500   90138.902500   0.000000
50%     73051.080000  122699.795000  212716.240000  107978.190000   1.000000
75%    101602.800000  144842.180000  299469.085000  139765.977500   2.000000
max    165349.200000  182645.560000  471784.100000  192261.830000   2.000000

startups.corr() #Correlation Coeficent
n_startups = preprocessing.normalize(startups) #Normalizing the data's
#visualization
plt.hist(startups.Rd)
plt.hist(startups.Ad)
plt.hist(startups.Pr)
plt.hist(startups.Ms)
plt.hist(startups.st)
sns.pairplot(startups) #pair plot
import statsmodels.formula.api as smf
model1=smf.ols("Pr~Rd+Ad+Ms+st",data=startups).fit()
model1.params
model1.summary()
pred=model1.predict(startups)
pred
#P-value for Ad,Ms,st is higher then 0.05,Then we have to go and check significance level
import statsmodels.formula.api as smf
mod1_Ad = smf.ols('Pr~Ad',data=startups).fit()
mod1_Ad.summary()
mod1_Ms = smf.ols('Pr~Ms',data=startups).fit()
mod1_Ms.summary()
mod1_st = smf.ols('Pr~st',data=startups).fit()
mod1_st.summary()
#Above summary I can say my 'st' variable has higher p value compare to other variables,so ignoring the 'st' variable from my data set 
import statsmodels.api as sm
#Next I'm going for influence plot to see which are the values are away from my line equation
sm.graphics.influence_plot(model1)
#(6,19,46,48,49)these are the values far away from my line equation,Therefor i'm removing those row values
startups_new = startups.drop(startups.index[[6,19,46,48,49]],axis=0)
model2=smf.ols("Pr~Rd+Ad+Ms",data=startups_new).fit()
model2.summary()
pred2=model2.predict(startups_new)
pred2
startups_new
startups_new.head()
#From my model2 i can say the variables called Ad and Ms has higher p-value>0.05,There for i'm going for Varience influence factor

rsq_Ad = smf.ols("Ad~Ms",data=startups_new).fit().rsquared  
vif_Ad = 1/(1-rsq_Ad) 
rsq_Ms = smf.ols("Ms~Ad",data=startups_new).fit().rsquared  
vif_Ms = 1/(1-rsq_Ms) 

d1 = {'Variables':['Ad','Ms',],'VIF':[vif_Ad,vif_Ms,]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame
# Variables       VIF
# 0        Ad  1.007246
# 1        Ms  1.007246
#My vif values are less than 10,so again i'm going for significance level for Ad and Ms
modelA=smf.ols("Pr~Ad",data=startups).fit()
modelA.summary()
modelM=smf.ols("Pr~Ms",data=startups).fit()
modelM.summary()
modelAM=smf.ols("Pr~Ad+Ms",data=startups).fit()
modelAM.summary()
#From above analysis I got P-value for 'Ms' is less than 0.05 and for 'Ad'  P-value(0.17) is greater than 0.05
#Because of significance error i'm avoiding the'Ad' variable from my data set
sm.graphics.plot_partregress_grid(model2)
model3=smf.ols("Pr~Rd+Ms",data=startups_new).fit()
model3.params
model3.summary()
import statsmodels.api as sm
sm.graphics.plot_partregress_grid(model3)
#Predicting the price of model3
pred3=model3.predict(startups_new)
pred
#Finally i'm going for Root mean square error(RMSE) to check the average error in my data set
rootmse = rmse(pred3,startups_new.Pr)
rootmse
Actual=startups_new.Pr
#Creating a dataframe set for actual and predicted price
df = pd.DataFrame(list(zip(pred3, Actual)),columns =['Predicted Prices', 'Actual Prices'])
#Next i'm going for to create a r^2 value table for my three models
values = list([model1.rsquared,model2.rsquared,model3.rsquared])#R^2 values
coded_variables = list(['model1.rsquared','model2.rsquared','model3.rsquared'])#
variables = list(['Model 1','Model 2','Model 3'])
Rsquared_model = pd.DataFrame(list(zip(variables,coded_variables,values)),columns = ['Models','Variabels Named in the code','R^Squared Values'])
Rsquared_model
Models Variabels Named in the code  R^Squared Values
0  Model 1             model1.rsquared          0.950746
1  Model 2             model2.rsquared          0.961353
2  Model 3             model3.rsquared          0.960088
#Here I can say my R^2 values are improved