# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 22:54:34 2020

@author: Lenovo
"""

#Consider only the below columns and prepare a prediction model for predicting Price.

#Corolla<-Corolla[c("Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight")]

#Model -- model of the car
#Price  -- Offer Price in EUROs	
#Age_08_04 -- Age in months as in August 2004	
#Mfg_Month -- Manufacturing month (1-12)	
#Mfg_Year	-- Manufacturing Year
#KM -- Accumulated Kilometers on odometer
#Fuel_Type	 -- Fuel Type (Petrol, Diesel, CNG)
#HP -- Horse Power
#Met_Color	 -- Metallic Color?  (Yes=1, No=0)
#Color -- Color (Blue, Red, Grey, Silver, Black, etc.)
#Automatic	-- Automatic ( (Yes=1, No=0)
#cc -- Cylinder Volume in cubic centimeters
#Doors -- Number of doors
#Cylinders	-- Number of cylinders
#Gears -- Number of gear positions
#Quarterly_Tax -- Quarterly road tax in EUROs
#Weight -- Weight in Kilograms
#Mfr_Guarantee -- Within Manufacturer's Guarantee period  (Yes=1, No=0)
#BOVAG_Guarantee -- BOVAG (Dutch dealer network) Guarantee  (Yes=1, No=0)
#Guarantee_Period -- 	Guarantee period in months
#ABS -- Anti-Lock Brake System (Yes=1, No=0)
#Airbag_1 -- Driver_Airbag  (Yes=1, No=0)
#Airbag_2 -- Passenger Airbag  (Yes=1, No=0)
#Airco -- Airconditioning  (Yes=1, No=0)
#Automatic_airco -- Automatic Airconditioning  (Yes=1, No=0)
#Boardcomputer -- Boardcomputer  (Yes=1, No=0)
#CD_Player -- CD Player  (Yes=1, No=0)
#Central_Lock -- Central Lock  (Yes=1, No=0)
#Powered_Windows -- Powered Windows  (Yes=1, No=0)
#Power_Steering -- Power Steering  (Yes=1, No=0)
#Radio -- Radio  (Yes=1, No=0)
#Mistlamps	-- Mistlamps  (Yes=1, No=0)
#Sport_Model -- Sport Model  (Yes=1, No=0)
#Backseat_Divider -- Backseat Divider  (Yes=1, No=0)
#Metallic_Rim --Metallic Rim  (Yes=1, No=0)
#Radio_cassette -- Radio Cassette  (Yes=1, No=0)
#Tow_Bar -- Tow Bar  (Yes=1, No=0)


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

# loading the data
Toyota = pd.read_csv("ToyotaCorolla.csv",encoding= "ISO-8859-1")
type(Toyota)
Toyota.head(10) # to get top n rows use cars.head(10)

# Correlation matrix 
Toyota.corr()
 
# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(Toyota.iloc[:,:])

#Remove unnecessary columns from the data set
Toyota=Toyota.drop('Id',axis=1)
Toyota=Toyota.drop('Model',axis=1)
Toyota=Toyota.drop('Mfg_Month',axis=1)
Toyota=Toyota.drop('Mfg_Year',axis=1)
Toyota=Toyota.drop('Fuel_Type',axis=1)
Toyota=Toyota.drop('Met_Color',axis=1)
Toyota=Toyota.drop('Color',axis=1)
Toyota=Toyota.drop('Automatic',axis=1)
Toyota=Toyota.drop('Cylinders',axis=1)
Toyota=Toyota.drop('Mfr_Guarantee',axis=1)
Toyota=Toyota.drop('BOVAG_Guarantee',axis=1)
Toyota=Toyota.drop('Guarantee_Period',axis=1)
Toyota=Toyota.drop('ABS',axis=1)
Toyota=Toyota.drop('Airbag_1',axis=1)
Toyota=Toyota.drop('Airbag_2',axis=1)
Toyota=Toyota.drop('Airco',axis=1)
Toyota=Toyota.drop('Automatic_airco',axis=1)
Toyota=Toyota.drop('Boardcomputer',axis=1)
Toyota=Toyota.drop('CD_Player',axis=1)
Toyota=Toyota.drop('Central_Lock',axis=1)
Toyota=Toyota.drop('Powered_Windows',axis=1)
Toyota=Toyota.drop('Power_Steering',axis=1)
Toyota=Toyota.drop('Radio',axis=1)
Toyota=Toyota.drop('Mistlamps',axis=1)
Toyota=Toyota.drop('Sport_Model',axis=1)
Toyota=Toyota.drop('Backseat_Divider',axis=1)
Toyota=Toyota.drop('Metallic_Rim',axis=1)
Toyota=Toyota.drop('Radio_cassette',axis=1)
Toyota=Toyota.drop('Tow_Bar',axis=1)

# columns names
Toyota.columns

# Preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model         

# Preparing model                  
ml1 = smf.ols('Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=Toyota).fit() # regression model

# Getting coefficients of variables               
ml1.params

# Summary
ml1.summary() #0.863
#p-values for cc and Doors are more than 0.05
pred = ml1.predict(Toyota) 
pred

# Preparing model based only on cc
ml_v=smf.ols('Price~cc',data = Toyota).fit()  
ml_v.summary()  #0.015
# p-value <0.05 .. It is Significant 

# Preparing model based only on Doors
ml_w=smf.ols('Price~Doors',data = Toyota).fit()  
ml_w.summary() #0.034
 # p-value < 0.05 .. It is Significant 

# Preparing model based on cc and Doors
ml_wv=smf.ols('Price~cc+Doors',data = Toyota).fit()  
ml_wv.summary() #0.046
#p-value < 0.05 and R^2 value is also increased

# Checking whether data has any influential values 
# influence index plots
import statsmodels.api as sm
sm.graphics.influence_plot(ml1)
# index 80 is showing high influence so we can exclude that entire row
# Studentized Residuals = Residual/standard deviation of residuals
Toyota_new = Toyota.drop(Toyota.index[80],axis=0) # ,inplace=False)
Toyota_new

# Preparing model                  
ml_new = smf.ols('Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=Toyota_new).fit()    

# Getting coefficients of variables        
ml_new.params

# Summary
ml_new.summary() #0.869
#R^2 Value increased from 0.863 to 0.869. Hence, ml_new is a better model.

# Confidence values 99%
print(ml_new.conf_int(0.05)) # 99% confidence level

# Predicted values of Profit 
Profit_pred = ml_new.predict(Toyota_new)
Profit_pred

# calculating VIF's values of independent variables
rsq_Age_08_04 = smf.ols('Age_08_04~KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=Toyota_new).fit().rsquared  
vif_Age_08_04 = 1/(1-rsq_Age_08_04) 

rsq_KM = smf.ols('KM~Age_08_04+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=Toyota_new).fit().rsquared  
vif_KM = 1/(1-rsq_KM)

rsq_HP = smf.ols('HP~Age_08_04+KM+cc+Doors+Gears+Quarterly_Tax+Weight',data=Toyota_new).fit().rsquared  
vif_HP = 1/(1-rsq_HP)

rsq_cc = smf.ols('cc~Age_08_04+KM+HP+Doors+Gears+Quarterly_Tax+Weight',data=Toyota_new).fit().rsquared  
vif_cc = 1/(1-rsq_cc)

rsq_Doors = smf.ols('Doors~Age_08_04+KM+HP+cc+Gears+Quarterly_Tax+Weight',data=Toyota_new).fit().rsquared  
vif_Doors = 1/(1-rsq_Doors)

rsq_Gears = smf.ols('Gears~Age_08_04+KM+HP+cc+Doors+Quarterly_Tax+Weight',data=Toyota_new).fit().rsquared  
vif_Gears = 1/(1-rsq_Gears)

rsq_Quarterly_Tax = smf.ols('Quarterly_Tax~Age_08_04+KM+HP+cc+Doors+Gears+Weight',data=Toyota_new).fit().rsquared  
vif_Quarterly_Tax = 1/(1-rsq_Quarterly_Tax)

rsq_Weight = smf.ols('Weight~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax',data=Toyota_new).fit().rsquared  
vif_Weight = 1/(1-rsq_Weight)

# Storing vif values in a data frame
d1 = {'Variables':['Age_08_04','KM','HP','cc','Doors','Gears','Quarterly_Tax','Weight'],'VIF':[vif_Age_08_04,vif_KM,vif_HP,vif_cc,vif_Doors,vif_Gears,vif_Quarterly_Tax,vif_Weight]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame

# Added varible plot 
sm.graphics.plot_partregress_grid(ml_new)

# final model
final_ml= smf.ols('Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=Toyota_new).fit()
final_ml.params
final_ml.summary() #0.869

Price_pred = final_ml.predict(Toyota_new)
Price_pred
import statsmodels.api as sm
# Added variable plot for the final model
sm.graphics.plot_partregress_grid(final_ml)