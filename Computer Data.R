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

install.packages("readr")
library(readr)
setwd("C://Users//Lenovo//Desktop//ExcelR//Assignments//Completed//Unchecked//Multi Linear Regression")
getwd()
Computer <- read.csv("Computer_Data.csv")
View(Computer)
install.packages("dummies")
#### creating a dummy variable#### 

library(dummies)
Cd_dummy <- dummy(Computer$cd)
View(Cd_dummy)

multi_dummy <- dummy(Computer$multi)
View(multi_dummy)

premium_dummy <- dummy(Computer$premium)
View(premium_dummy)

#bind this  dummy data sets, by cibnd function

Computer_data <- cbind(Computer,Cd_dummy, multi_dummy, premium_dummy)
View(Computer_data)

Computer_data <- Computer_data[-c(7,8,9)]
View(Computer_data)

Computer_data <- Computer_data[-1]
View(Computer_data)

library(readr)
summary(Computer_data)
str(Computer_data)

attach(Computer_data)

##Histogram  Computer data set 'variables'
hist(Computer_data$price)
hist(Computer_data$speed)
hist(Computer_data$hd)
hist(Computer_data$ram)
hist(Computer_data$screen)
hist(Computer_data$ads)
hist(Computer_data$trend)
hist(Computer_data$cdno)
hist(Computer_data$cdyes)
hist(Computer_data$multino)
hist(Computer_data$multiyes)
hist(Computer_data$premiumno)
hist(Computer_data$premiumyes)

####Variance of Computer data set "Variables"
var(Computer_data$price)
var(Computer_data$speed)
var(Computer_data$hd)
var(Computer_data$ram)
var(Computer_data$screen)
var(Computer_data$ads)
var(Computer_data$trend)
var(Computer_data$cdno)
var(Computer_data$cdyes)
var(Computer_data$multino)
var(Computer_data$multiyes)
var(Computer_data$premiumno)
var(Computer_data$premiumyes)

################ Standard deviance of Computer  "variable"

sd(Computer_data$price)
sd(Computer_data$speed)
sd(Computer_data$hd)
sd(Computer_data$ram)
sd(Computer_data$screen)
sd(Computer_data$ads)
sd(Computer_data$trend)
sd(Computer_data$cdno)
sd(Computer_data$cdyes)
sd(Computer_data$multino)
sd(Computer_data$multiyes)
sd(Computer_data$premiumno)
sd(Computer_data$premiumyes)


###Busines 3rd and 4th Moment. (skewness and kurtosis)
library(moments)

skewness(Computer_data$price)
skewness(Computer_data$speed)
skewness(Computer_data$hd)
skewness(Computer_data$ram)
skewness(Computer_data$screen)
skewness(Computer_data$ads)
skewness(Computer_data$trend)
skewness(Computer_data$cdno)
skewness(Computer_data$cdyes)
skewness(Computer_data$multino)
skewness(Computer_data$multiyes)
skewness(Computer_data$premiumno)
skewness(Computer_data$premiumyes)



kurtosis(Computer_data$price)
kurtosis(Computer_data$speed)
kurtosis(Computer_data$hd)
kurtosis(Computer_data$ram)
kurtosis(Computer_data$screen)
kurtosis(Computer_data$ads)
kurtosis(Computer_data$trend)
kurtosis(Computer_data$cdno)
kurtosis(Computer_data$cdyes)
kurtosis(Computer_data$multino)
kurtosis(Computer_data$multiyes)
kurtosis(Computer_data$premiumno)
kurtosis(Computer_data$premiumyes)


### Bar Plot########
barplot(Computer_data$price)
barplot(Computer_data$speed)
barplot(Computer_data$hd)
barplot(Computer_data$ram)
barplot(Computer_data$screen)
barplot(Computer_data$ads)
barplot(Computer_data$trend)
barplot(Computer_data$cdno)
barplot(Computer_data$cdyes)
barplot(Computer_data$multino)
barplot(Computer_data$multiyes)
barplot(Computer_data$premiumno)
barplot(Computer_data$premiumyes)


########Box plot############
boxplot(Computer_data$price)
boxplot(Computer_data$speed)
boxplot(Computer_data$hd)
boxplot(Computer_data$ram)
boxplot(Computer_data$screen)
boxplot(Computer_data$ads)
boxplot(Computer_data$trend)
boxplot(Computer_data$cdno)
boxplot(Computer_data$cdyes)
boxplot(Computer_data$multino)
boxplot(Computer_data$multiyes)
boxplot(Computer_data$premiumno)
boxplot(Computer_data$premiumyes)


########Mean, Median, Mode###########
mean(Computer_data$price)
mean(Computer_data$speed)
mean(Computer_data$hd)
mean(Computer_data$ram)
mean(Computer_data$screen)
mean(Computer_data$ads)
mean(Computer_data$trend)
mean(Computer_data$cdno)
mean(Computer_data$cdyes)
mean(Computer_data$multino)
mean(Computer_data$multiyes)
mean(Computer_data$premiumno)
mean(Computer_data$premiumyes)

median(Computer_data$price)
median(Computer_data$speed)
median(Computer_data$hd)
median(Computer_data$ram)
median(Computer_data$screen)
median(Computer_data$ads)
median(Computer_data$trend)
median(Computer_data$cdno)
median(Computer_data$cdyes)
median(Computer_data$multino)
median(Computer_data$multiyes)
median(Computer_data$premiumno)
median(Computer_data$premiumyes)

attach(Computer_data)

########## Lets begin the model#######

Model_Computer <- lm(price~log., data = Computer_data)
summary(Model_Computer)
sum(is.na(Computer_data))
sqrt(sum(Model_Computer$residuals)^2)/nrow(Computer_data)

# R_squared value for this model R^2= 0.7756
## Sum of SQRT is 1.24558e-14
## Exponential Model#####

Model_Computer_exp <- lm(log(price)~., data = Computer_data)
summary(Model_Computer_exp)
sum(is.na(Computer_data))
sqrt(sum(Model_Computer_exp$residuals)^2)/nrow(Computer_data)

# R_squared value for this model R^2= 0.7832, better than above model

## Exonential model has good R^2 value as compared to the above 
## RMSE value of exponential method is good than other model
## we can say exponential method is quite good method


## R^2 is 0.7832
## Here the P values for the input variable lies in the range.

library(psych)
library(car)


plot(Model_Computer)
qqPlot(Model_Computer)