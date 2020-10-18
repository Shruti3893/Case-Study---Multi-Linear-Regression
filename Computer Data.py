# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 15:31:46 2020

@author: Lenovo
"""


#Predict Price of the computer
#A dataframe containing :
#price : price in US dollars of 486 PCs
#speed : clock speed in MHz
#hd : size of hard drive in MB
#ram : size of Ram in in MB
#screen : size of screen in inches
#cd : is a CD-ROM present ?
#multi : is a multimedia kit (speakers, sound card) included ?
#premium : is the manufacturer was a "premium" firm (IBM, COMPAQ) ?
#ads : number of 486 price listings for each month
#trend : time trend indicating month starting from January of 1993 to November of 1995.


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

# loading the data
Computer = pd.read_csv("Computer_Data.csv")
type(Computer)
# to get top 6 rows
Computer.head(10) # to get top n rows use Computer.head(10)

#Convert State Column from Character to Binary
from sklearn import preprocessing
df = preprocessing.LabelEncoder()
Computer['cd'] = df.fit_transform(Computer['cd'] )
Computer['multi'] = df.fit_transform(Computer['multi'] )
Computer['premium'] = df.fit_transform(Computer['premium'] )

#Remove Unnamed column from the Dataset
Computer = Computer.loc[:, ~Computer.columns.str.contains('^Unnamed')]

# Correlation matrix 
Computer.corr()
# we see there exists High collinearity between input variables especially between
# [hd,ram] so there exists collinearity problem
 
# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(Computer.iloc[:,:])

# columns names
Computer.columns
                             
# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
         
# Preparing model                  
ml1 = smf.ols('price~speed+hd+ram+screen+cd+multi+premium+ads+trend',data=Computer).fit() # regression model

# Getting coefficients of variables               
ml1.params

# Summary
ml1.summary() #0.775
pred = ml1.predict(Computer) 
pred

import statsmodels.api as sm
sm.graphics.influence_plot(ml1)

# calculating VIF's values of independent variables
rsq_speed = smf.ols('speed~hd+ram+screen+cd+multi+premium+ads+trend',data=Computer).fit().rsquared  
vif_speed = 1/(1-rsq_speed) 

rsq_hd = smf.ols('hd~speed+ram+screen+cd+multi+premium+ads+trend',data=Computer).fit().rsquared  
vif_hd = 1/(1-rsq_hd) 

rsq_ram = smf.ols('ram~speed+hd+screen+cd+multi+premium+ads+trend',data=Computer).fit().rsquared  
vif_ram = 1/(1-rsq_ram) 

rsq_screen = smf.ols('screen~speed+hd+ram+cd+multi+premium+ads+trend',data=Computer).fit().rsquared  
vif_screen = 1/(1-rsq_screen)

rsq_cd = smf.ols('cd~speed+hd+ram+screen+multi+premium+ads+trend',data=Computer).fit().rsquared  
vif_cd = 1/(1-rsq_cd)

rsq_multi = smf.ols('multi~speed+hd+ram+screen+cd+premium+ads+trend',data=Computer).fit().rsquared  
vif_multi = 1/(1-rsq_multi)

rsq_premium = smf.ols('premium~speed+hd+ram+screen+cd+multi+ads+trend',data=Computer).fit().rsquared  
vif_premium = 1/(1-rsq_premium)

rsq_ads = smf.ols('ads~speed+hd+ram+screen+cd+multi+premium+trend',data=Computer).fit().rsquared  
vif_ads = 1/(1-rsq_ads)

rsq_trend = smf.ols('trend~speed+hd+ram+screen+cd+multi+premium+ads',data=Computer).fit().rsquared  
vif_trend = 1/(1-rsq_trend)

# Storing vif values in a data frame
d1 = {'Variables':['speed','hd','ram','screen','cd','multi','premium','ads','trend'],'VIF':[vif_speed,vif_hd,vif_ram,vif_screen,vif_cd,vif_multi,vif_premium,vif_ads,vif_trend]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame
# As hd is having higher VIF value, we are not going to include this prediction model

# Added varible plot 
sm.graphics.plot_partregress_grid(ml1)

# model 2
ml2= smf.ols('price~speed+ram+screen+cd+multi+premium+ads+trend',data = Computer).fit()
ml2.params
ml2.summary() # 0.746
# As we can see that r-squared value has decreased from 0.775 to 0.746.
#Hence, final model is ml1

# Preparing model                  
Final_ml1 = smf.ols('price~speed+hd+ram+screen+cd+multi+premium+ads+trend',data=Computer).fit() # regression model

# Getting coefficients of variables               
Final_ml1.params

# Summary
Final_ml1.summary() #0.775
pred = ml1.predict(Computer) 
pred
