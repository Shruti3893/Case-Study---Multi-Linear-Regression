# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 00:31:56 2020

@author: Lenovo
"""
#Prepare a prediction model for profit of 50_startups data.
#Do transformations for getting better predictions of profit and
#make a table containing R^2 value for each prepared model.

#R&D Spend -- Research and devolop spend in the past few years
#Administration -- spend on administration in the past few years
#Marketing Spend -- spend on Marketing in the past few years
#State -- states from which data is collected
#Profit  -- profit of each state in the past few years


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

# loading the data
Startups = pd.read_csv("50_Startups.csv")
type(Startups)
Startups.head(10) # to get top n rows use cars.head(10)

#Convert State Column from Character to Binary
from sklearn import preprocessing
df = preprocessing.LabelEncoder()
Startups['State'] = df.fit_transform(Startups['State'])

# Correlation matrix 
Startups.corr()
 
# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(Startups.iloc[:,:])

# columns names
Startups.columns
Startups = Startups.rename(columns = {"R&D Spend":"RandD_Spend"})
Startups = Startups.rename(columns = {"Marketing Spend":"Marketing_Spend"})

# pd.tools.plotting.scatter_matrix(Startups); -> also used for plotting all in one graph
# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model         
# Preparing model                  
ml1 = smf.ols('Profit~RandD_Spend+Administration+Marketing_Spend+State',data=Startups).fit() # regression model

# Getting coefficients of variables               
ml1.params

# Summary
ml1.summary()
# p-values for State, Administration and Marketing_Spend  are more than 0.05
pred = ml1.predict(Startups) 
pred
 

# Preparing model based only on State
ml_v=smf.ols('Profit~State',data = Startups).fit()  
ml_v.summary()  #-0.010
# p-value >0.05 .. It is Insignificant || Also R^2 Value is negative

# Preparing model based only on Administration
ml_w=smf.ols('Profit~Administration',data = Startups).fit()  
ml_w.summary() #0.020
 # p-value >0.05 .. It is Insignificant 

# Preparing model based only on Marketing_Spend
ml_wv=smf.ols('Profit~Marketing_Spend',data = Startups).fit()  
ml_wv.summary() # 0.550
#p-value < 0.05 .. It is Significant

# Preparing model based on State, Administration and Marketing_Spend
ml_wv=smf.ols('Profit~State+Administration+Marketing_Spend',data = Startups).fit()  
ml_wv.summary() # 0.586
#State Variable has got p-value >0.05. Hence, check with remaining both Variables.

# Preparing model based on Administration and Marketing_Spend
ml_wv=smf.ols('Profit~Administration+Marketing_Spend',data = Startups).fit()  
ml_wv.summary() #0.593
#P-value for both coefficients is <0.05 and R^2 value is also high compared to other Variables.

# Checking whether data has any influential values 
# influence index plots
import statsmodels.api as sm
sm.graphics.influence_plot(ml1)
# index 49 AND 48 is showing high influence so we can exclude that entire row
# Studentized Residuals = Residual/standard deviation of residuals
Startups_new = Startups.drop(Startups.index[[48,49]],axis=0) # ,inplace=False)
Startups_new

# Preparing model                  
ml_new = smf.ols('Profit~RandD_Spend+Administration+Marketing_Spend+State',data = Startups_new).fit()    

# Getting coefficients of variables        
ml_new.params

# Summary
ml_new.summary() #0.959

# Confidence values 99%
print(ml_new.conf_int(0.05)) # 99% confidence level

# Predicted values of Profit 
Profit_pred = ml_new.predict(Startups_new)
Profit_pred

# calculating VIF's values of independent variables
rsq_RandD_Spend = smf.ols('RandD_Spend~Administration+Marketing_Spend+State',data=Startups_new).fit().rsquared  
vif_RandD_Spend = 1/(1-rsq_RandD_Spend) 

rsq_Administration = smf.ols('Administration~RandD_Spend+Marketing_Spend+State',data=Startups_new).fit().rsquared  
vif_Administration = 1/(1-rsq_Administration)

rsq_Marketing_Spend = smf.ols('Marketing_Spend~RandD_Spend+Administration+State',data=Startups_new).fit().rsquared  
vif_Marketing_Spend = 1/(1-rsq_Marketing_Spend)

rsq_State = smf.ols('State~RandD_Spend+Administration+Marketing_Spend',data=Startups_new).fit().rsquared  
vif_State = 1/(1-rsq_State)

# Storing vif values in a data frame
d1 = {'Variables':['RandD_Spend','Administration','Marketing_Spend','State'],'VIF':[vif_RandD_Spend,vif_Administration,vif_Marketing_Spend,vif_State]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame

# Added varible plot 
sm.graphics.plot_partregress_grid(ml_new)

# final model
final_ml= smf.ols('Profit~RandD_Spend+Administration+Marketing_Spend+State',data = Startups_new).fit()
final_ml.params
final_ml.summary() #0.959

Profit_pred = final_ml.predict(Startups_new)
Profit_pred
import statsmodels.api as sm
# Added variable plot for the final model
sm.graphics.plot_partregress_grid(final_ml)