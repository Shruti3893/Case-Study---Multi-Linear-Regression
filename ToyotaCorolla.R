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
                        # cc -- Cylinder Volume in cubic centimeters
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


install.packages("readr")
library(readr)
setwd("C://Users//Lenovo//Desktop//ExcelR//Assignments//Multi Linear Regression")
getwd()
Toyata <- read.csv("ToyotaCorolla.csv")
View(Toyata)

## We delete sum columns as per model building above
## we have mention which column is needed for building a Model
## Column need to be remove as follows Column no. 1,2,5,6,8,10,11,12,15,19:38 

Toyata_data <- Toyata[-c(1:2,5:6,8,10:12,15,19:38)]
View(Toyata_data)

library(readr)
summary(Toyata_data)
str(Toyata_data)

attach(Toyata_data)

##Histogram  Toyata data set 'variables'

hist(Toyata_data$Price)
hist(Toyata_data$Age_08_04)
hist(Toyata_data$KM)
hist(Toyata_data$HP)
hist(Toyata_data$cc)
hist(Toyata_data$Doors)
hist(Toyata_data$Gears)
hist(Toyata_data$Quarterly_Tax)
hist(Toyata_data$Weight)

####Variance of  Toyata data set "Variables"

var(Toyata_data$Price)
var(Toyata_data$Age_08_04)
var(Toyata_data$KM)
var(Toyata_data$HP)
var(Toyata_data$cc)
var(Toyata_data$Doors)
var(Toyata_data$Gears)
var(Toyata_data$Quarterly_Tax)
var(Toyata_data$Weight)

################ Standard deviance of Toyata"variable"

sd(Toyata_data$Price)
sd(Toyata_data$Age_08_04)
sd(Toyata_data$KM)
sd(Toyata_data$HP)
sd(Toyata_data$cc)
sd(Toyata_data$Doors)
sd(Toyata_data$Gears)
sd(Toyata_data$Quarterly_Tax)
sd(Toyata_data$Weight)

###Busines 3rd and 4th Moment. (skewness and kurtosis)
library(moments)

skewness(Toyata_data$Price)
skewness(Toyata_data$Age_08_04)
skewness(Toyata_data$KM)
skewness(Toyata_data$HP)
skewness(Toyata_data$cc)
skewness(Toyata_data$Doors)
skewness(Toyata_data$Gears)
skewness(Toyata_data$Quarterly_Tax)
skewness(Toyata_data$Weight)


kurtosis(Toyata_data$Price)
kurtosis(Toyata_data$Age_08_04)
kurtosis(Toyata_data$KM)
kurtosis(Toyata_data$HP)
kurtosis(Toyata_data$cc)
kurtosis(Toyata_data$Doors)
kurtosis(Toyata_data$Gears)
kurtosis(Toyata_data$Quarterly_Tax)
kurtosis(Toyata_data$Weight)



### Bar Plot########

barplot(Toyata_data$Price)
barplot(Toyata_data$Age_08_04)
barplot(Toyata_data$KM)
barplot(Toyata_data$HP)
barplot(Toyata_data$cc)
barplot(Toyata_data$Doors)
barplot(Toyata_data$Gears)
barplot(Toyata_data$Quarterly_Tax)
barplot(Toyata_data$Weight)


########Box plot############

boxplot(Toyata_data$Price)
boxplot(Toyata_data$Age_08_04)
boxplot(Toyata_data$KM)
boxplot(Toyata_data$HP)
boxplot(Toyata_data$cc)
boxplot(Toyata_data$Doors)
boxplot(Toyata_data$Gears)
boxplot(Toyata_data$Quarterly_Tax)
boxplot(Toyata_data$Weight)



########Mean, Median, Mode###########

mean(Toyata_data$Price)
mean(Toyata_data$Age_08_04)
mean(Toyata_data$KM)
mean(Toyata_data$HP)
mean(Toyata_data$cc)
mean(Toyata_data$Doors)
mean(Toyata_data$Gears)
mean(Toyata_data$Quarterly_Tax)
mean(Toyata_data$Weight)


median(Toyata_data$Price)
median(Toyata_data$Age_08_04)
median(Toyata_data$KM)
median(Toyata_data$HP)
median(Toyata_data$cc)
median(Toyata_data$Doors)
median(Toyata_data$Gears)
median(Toyata_data$Quarterly_Tax)
median(Toyata_data$Weight)

attach(Toyata_data)

########## Lets begin the model#######

Model_Toyata <- lm(Price~., data = Toyata_data)
summary(Model_Toyata)
sum(is.na(Model_Toyata))
sqrt(sum(Model_Toyata$residuals)^2)/nrow(Toyata_data)

## R_squared value for this model R^2= 0.8638
##Sqrt value is 5.542e-14

##All variables has good P value apart from that "cc, Doors"

## These 2 varialbe are insignificant

#Lets explore more, build a model in these four variable alone

Model_cc <- lm(Price~cc, data = Toyata_data)
summary(Model_cc)
#R^2 is 0.015
##When we build alone with this variable we can say that "cc" is insignificant


Model_door <- lm(Price~Doors, data = Toyata_data)
summary(Model_door)
# R^2 is 0.0343,
#Here again we find the build a model alone with Door,this is significant


## Build a Model with these two varialbe together

Model_Both <- lm(Price~cc+Doors, data=Toyata_data)
summary(Model_Both)
##R^2 is, 0.04688, both are significant when we take it together

library(psych)
library(car)
pairs.panels(Toyata_data)   #Through this plot we can check so many things
# like r value, graph linear or not linear, co-linearity relationship


influence.measures(Model_Toyata)
#From consolve window we can say that row no. values in row no. 50,53,54
#we delete all than again lost lots off data

#Ploting the influence measures
influenceIndexPlot(Model_Toyata)
influencePlot(Model_Toyata) ## More clear picture from that plot
# we find that value in 81,  rows has insignificant to that Model


##### Building a model again to delete this values 
Model_2 <- lm(Price~., data = Toyata_data[-c(50,53:54),])
summary(Model_2)

## we find that R^2 value increased R= 0.867  but however
## "cc", "Doors" are insignificant to the Model

## Going to delete the column, but how we chose which column
## we can decide this from the "vif" function

vif(Model_Toyata) # vif>10 tells the collinearity but however 
## from "cc" and "Doors" we delete which has higher vif value

## using backword regression technique

## Model building removeing the variable "cc"

Model_B <- lm(Price~Age_08_04+KM+HP+Doors+Gears+Quarterly_Tax+Weight)
summary(Model_B)

##Removing "Doors"

Model_C <- lm(Price~Age_08_04+KM+HP+Gears+Quarterly_Tax+Weight)
summary(Model_C)

### after above Analysis we can say that Model-C is good. 
## All variables are significant to the Model.


## Building a different type of Models from these variables

Model_final_A <- lm(Price~Age_08_04+KM+HP+Gears+Quarterly_Tax+Weight, data = Toyata_data)

summary(Model_final_A)
# R^2 is 0.8636, P values for all the input variable is also in range less than 0.05

RMSE_1 <- sqrt(sum(Model_final_A$residuals)^2)/nrow(Toyata_data)
RMSE_1 ##4.166e-14
plot(Model_final_A)
qqPlot(Model_final_A)

### Exponential Model###


Model_final_B <- lm(log(Price)~Age_08_04+KM+HP+Gears+Quarterly_Tax+Weight, data = Toyata_data)

summary(Model_final_B)
# R^2 is 0.8505, P values for all the input variable is also in range less than 0.05

RMSE_2 <- sqrt(sum(Model_final_B$residuals)^2)/nrow(Toyata_data)
RMSE_2 ##4.66599e-14, not so much difference in Both RMSE value but little bit.
plot(Model_final_A)
qqPlot(Model_final_A)


## we can say that from above two model 
## Model_final_B is better than "Model_final_A" 
## Because . RMSE values is less than that