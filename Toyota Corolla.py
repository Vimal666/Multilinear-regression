# -*- coding: utf-8 -*-
"""
Created on Wed May 27 22:47:29 2020

@author: Vimal PM
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from ml_metrics import rmse
#importing data set
tc = pd.read_csv("D://DATA SCIENCE//ASSIGNMENT//ASSIGNMENT5//tc.csv", encoding= 'unicode_escape')
#Above encoding of my csv using the encoding attribute on the to_csv call. 'unicode_escape' is for producing an ascii encoding of the Unicode string
tc.columns
Tc=tc.drop(['Id','Model','Mfg_Month','Mfg_Year','Fuel_Type','Met_Color','Color','Automatic','Cylinders','Mfr_Guarantee','BOVAG_Guarantee','Guarantee_Period','ABS','Airbag_1','Airbag_2','Airco','Automatic_airco','Boardcomputer','CD_Player','Central_Lock','Powered_Windows','Power_Steering','Radio','Mistlamps','Sport_Model','Backseat_Divider','Metallic_Rim','Radio_cassette','Tow_Bar'],axis = 1) #here am dropping the unwanted variables from my data
Tc.describe()
Tc.corr()#checking the correlation
sns.pairplot(Tc)# to visualize the pairplot
Tc.columns
#Index(['Price', 'Age', 'KM', 'HP', 'cc', 'Doors', 'Gears', 'Quarterly','Weight']
       
import statsmodels.formula.api as smf
#buliding first model
#Here i'm adding Age+KM+HP+cc+Doors+Gears+Quarterly+Weight against Price 
Model1=smf.ols("Price~Age+KM+HP+cc+Doors+Gears+Quarterly+Weight",data=Tc).fit()
Model1.params
Model1.summary()
#predicting the price for my first model1
pred1=Model1.predict(Tc)
pred1
#from first model 'cc' and 'doors' are the only variables which having high p-value greater than 0.05
#Pvalue for doors=0.968,Pvalue for cc=0.179
#Next going for significance
modelCC=smf.ols("Price~cc",data=Tc).fit()#price against cc 
modelCC.summary()
modelDOORS=smf.ols("Price~Doors",data=Tc).fit()#price against doors
modelDOORS.summary()
ModelCCD=smf.ols("Price~cc+Doors",data=Tc).fit()
ModelCCD.summary()#seems like there is no issue  for pvalue of cc+Doors against Price
import statsmodels.api as sm
#Next i'm going for influence plot to check what are the data points are far away from my line equation
sm.graphics.influence_plot(Model1)
#From the influence plot visualization i can say  80,221,960 are the data points are far away
#Therefor i'm removing these data points from my data set and save this new data in new variable called Tc_NEW
Tc_NEW=Tc.drop(Tc.index[[80,960,221]],axis=0)
#Building the second model
Model2=smf.ols("Price~Age+KM+HP+cc+Doors+Gears+Quarterly+Weight",data=Tc_NEW).fit()#Age+KM+HP+cc+Doors+Gears+Quarterly+Weight against price
Model2.params
Model2.summary()
#predicting price from my second model
pred2=Model2.predict(Tc_NEW)
pred2
#from this model2 i can say my gear variable is having p-value(0.010) which is greater than 0.05.There for i'm going for significance 
model2Gears=smf.ols("Price~Gears",data=Tc_NEW).fit()#Price against gear
model2Gears.summary()
#from this analysis i can say the Pvalue of gear which still having greater than 0.05
#so i would like to remove some data points again to check what are the data points are away from the line equation
#visualizing the influence plot fro my model2
sm.graphics.influence_plot(Model2)
#from the influence plot i can say 109,110,111,956,991,601,654 are the data points are away from the line equation
#therefor i'm removing those data points from previous dataset called Tc_NEW and store the modified data in new dataset called TC_new
TC_new=Tc_NEW.drop(Tc_NEW.index[[109,110,111,601,654,956,991]],axis=0)
#building the 3rd model without removing any variables from my datasets 
Model3=smf.ols("Price~Age+KM+HP+cc+Doors+Gears+Quarterly+Weight",data=TC_new).fit()
Model3.summary()
#predicting the price for my model3
pred3=Model3.predict(TC_new)
pred3
#from my model3  i can see my all pvalues for each and every variables is less than 0.05 and it's significant.R^2 value i got is 0.879,so i can say model3 is a good one.
#Going for logrithmic transformation to improve my R^2 value
#BULILDING fourth model
Model4=smf.ols("np.log(Price)~Age+KM+HP+cc+Doors+Gears+Quarterly+Weight",data=TC_new).fit() 
Model4.summary()
#predicting the price for model4
pred4=Model4.predict(TC_new)
pred4
#from model4 i can see the 'doors' having higher pvalue(0.125) which is greater than 0.05
#so i would like to remove that door variable from the dataset using drop() and store this modified data in a new variable called TC_NEW
TC_NEW=TC_new.drop(["Doors"],axis=1)
#building the fifth model without 'doors' variable
Model5=smf.ols("Price~Age+KM+HP+cc+Gears+Quarterly+Weight",data=TC_NEW).fit() #Age+KM+HP+cc+Gears+Quarterly+Weight against price
Model5.summary()
#predicting the price for model5
pred5=Model5.predict(TC_NEW)
pred5
#from model5 i can see pvalues for every variables which is less than 0.05 and R^2 value i got is 0.878
#again going for influence plot to improve my R^2 value
sm.graphics.influence_plot(Model5)#from this plot i can see 109,601,956,991 are the data points are away from the line equation
TC_New=TC_NEW.drop(TC_NEW.index[[109,601,956,991]],axis=0)#here i'm removing those data points and store the modified dataset in a new variable called TC_New
#going for 6th model
Model6=smf.ols("np.log(Price)~Age+KM+HP+cc+Gears+Quarterly+Weight",data=TC_New).fit()
Model6.summary()
#predicting the price from model6
pred6=Model6.predict(TC_New)
pred6
#from this model6 i can see pvalues for each and every variable which is having less than 0.05 and R^2 values i got 0.852
#so i can say R^2 value hasn't improved
#Here i'm bulding a table to see which model having higher R^2 value
values = list([Model1.rsquared,Model2.rsquared,Model3.rsquared,Model4.rsquared,Model5.rsquared,Model6.rsquared])
coded_variables = list(['Model1.rsquared','Model2.rsquared','Model3.rsquared','Model4.rsquared','Model5.rsquared','Model6.rsquared'])
variables = list(['Model 1','Model 2','Model 3','Model 4','Model 5','Model 6'])
Rsquared_model = pd.DataFrame(list(zip(variables,coded_variables,values)),columns = ['Models','Variabels Named in the code','R^Squared Values'])
Rsquared_model
#From my analysis I can say my Model3 is better one compare to other models,which having higher R^2 value (0.8789) and from that Model3 I didn't removed any variables because all variables having less than 0.05 P-values
#finally am checking the root mean square error for my pred3 data with actual data called TC_new
import statsmodels.api as sm
rootmse = rmse(pred3,TC_new.Price)
rootmse
#Out[215]: 1227.2689621781449
#Next i'm dataframing predicted data with actual data and store it in a new variable called df
Actual=TC_new.Price
df = pd.DataFrame(list(zip(pred3, Actual)),columns =['Predicted Prices', 'Actual Prices'])
