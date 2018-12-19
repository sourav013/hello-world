#lets clear the environment
rm(list = ls())


#set working directory 
setwd("S:/analytics/Projects/Employee Absenteeism")

getwd()

#load XLSX library
library("xlsx")

#Read the data

data = read.xlsx("Absenteeism_at_work_Project.xls", sheetIndex = 1)

#lets observe the dimension of data set
dim(data)

#lets get familiar with the column names
colnames(data)

#lets observe the structure, summary and class of data set respectively
str(data)
summary(data)
class(data)

#NOTE: as we can see all the 21 variables are in continuous form 

# NOTE: we have to categorise which of the attributes are continuous and which are categorical

### MISSING VALUE ANALYSIS ###
#lets check for the missing values
sum(is.na(data))  #number of missing values = 135\

#create a data frame with missing values 
missing_values = data.frame(apply(data,2,function(x)sum(is.na(x))))

names(missing_values)[1] = "Missing_Values"

missing_values$Variables = row.names(missing_values)

row.names(missing_values) = NULL


#Reaarange the columns
missing_values = missing_values[, c(2,1)]

#place the variables according to their number of missing values.
missing_values = missing_values[order(-missing_values$Missing_Values),]

#Calculate the missing value percentage
missing_values$percentage = (missing_values$Missing_Values/nrow(data) )* 100


#Store the missing value information in a csv file
write.csv(missing_values,"EA Project_Missing_value.csv", row.names = F)


### IMPORTANT NOTE: we will not drop any data because missing percentage is  less and data is also less as well. 


#created a copy of data for back up as we will need it further
df = data
#data = df
#imputing missing values

## create a missing value in any variable
data$Body.mass.index[12] #23
data$Body.mass.index[12] = NA

#Actual values = 23
#value from mean method = 26.68
#value from MEdian method = 25
#value from KNN ethod = 23

##lets apply mean method to impute this generated value and observe the result
#data$Body.mass.index[is.na(data$Body.mass.index)]  =  mean(data$Body.mass.index, na.rm = T) #26.68

##lets apply median method to impute this generated value and observe the result
#data$Body.mass.index[is.na(data$Body.mass.index)]  =  median(data$Body.mass.index, na.rm = T) #25


###Now convert the variables into their respective types
#structure of data
str(data)

data$ID = as.factor(as.character(data$ID))
data$Day.of.the.week = as.factor(as.character(data$Day.of.the.week))
data$Education = as.factor(as.character(data$Education))
data$Social.drinker = as.factor(as.character(data$Social.drinker))
data$Social.smoker = as.factor(as.character(data$Social.smoker))
data$Reason.for.absence = as.factor(as.character(data$Reason.for.absence))
data$Seasons = as.factor(as.character(data$Seasons))
data$Month.of.absence = as.factor(as.character(data$Month.of.absence))
data$Disciplinary.failure = as.factor(as.character(data$Disciplinary.failure))


#now apply KNN method to impute

library(DMwR)
data = knnImputation(data,k=3)

#after analysis of different data using median and KNN we find that KNN is more accurate and closerto original value than median for imputation

#Check presence of missing values once to confirm

apply(data,2, function(x){sum(is.na(x))})

#### NO more missing values are present here ###

#create subset of the dataset which have only numeric varaiables
#fisrt of all we will define a numeric index
numeric_index = sapply(data, is.numeric)
numeric_data = data[,numeric_index]

#now, we will store the numeric variables in a proper data frame which is a subset of dataset
numeric_data = as.data.frame(numeric_data)

#storing all the colnames excluding target variable name
cnames = colnames(numeric_data)[-12]

#outlier analysis
##box-plot distribution and outlier check
library(ggplot2)

for (i in 1:length(cnames)) {
  assign(paste0("gn",i), ggplot(aes_string( y = (cnames[i]), x= "Absenteeism.time.in.hours") , data = subset(data)) +
           stat_boxplot(geom = "errorbar" , width = 0.5) +
           geom_boxplot(outlier.color = "red", fill = "grey", outlier.shape = 20, outlier.size = 1, notch = FALSE)+
           theme(legend.position = "bottom")+
           labs(y = cnames[i], x= "Absenteeism.time.in.hours")+
           ggtitle(paste("Boxplot" , cnames[i])))
           #print(i)
}

options(warn = 0)

#lets plot the boxplots
gridExtra::grid.arrange(gn1, gn2,gn3, ncol=3)
gridExtra::grid.arrange(gn4,gn5,gn6, ncol=3)
gridExtra::grid.arrange(gn7,gn8,gn9, ncol =3)
gridExtra::grid.arrange(gn10,gn11, ncol =3 )

#here, either we can remove the outliers from all variables or can impute them by providin them value NA

#val = data$Distance.from.Residence.to.Work[data$Distance.from.Residence.to.Work %in% boxplot.stats(data$Distance.from.Residence.to.Work)$out]

#as the number of outliers is already less and volume of data is also low, we would go making each outlier NA and then will impute them

#getting outliers using boxplot.stat method
for (i in cnames) {
  print(i)
  val = data[,i][data[,i] %in% boxplot.stats(data[,i])$out]
  print(length(val))
  print(val)
}

#Make each outlier as NA
for (i in cnames) {
  val = data[,i][data[,i] %in% boxplot.stats(data[,i])$out]
  data[,i][data[,i] %in% val] = NA
}

#checking the missing values
sum(is.na(data)) #231

#find thenumber of missing values in each variable
for (i in cnames) {
  print(i)
  print(sum(is.na(data[,i])))
}

#Impute the values using KNN imputation method
data = knnImputation(data, k=3)

#Check again for missing value if present in case
sum(is.na(data)) #0


#lets confirm for the outliers by both the methods

for (i in 1:length(cnames)) {
  print(i)
  assign(paste0("g",i), ggplot(aes_string(y = (cnames[i]), x= "Absenteeism.time.in.hours"), data=subset(data))+
           stat_boxplot(geom = "errorbar", width= 0.5) +
           geom_boxplot(outlier.color = "red" , fill = "grey", outlier.shape = 18, outlier.size = 1, notch = FALSE) +
           theme(legend.position = "bottom")+
           labs(y = cnames[i], x="Absenteeism.time.in.hours") +
           ggtitle(paste("Box Plot of Employee data", cnames[i])))
  
}

gridExtra::grid.arrange(g1,g2, ncol = 2) 
gridExtra::grid.arrange(g3,g4,g5, ncol = 3)
gridExtra::grid.arrange(g6,g7,g8, ncol=3) 
gridExtra::grid.arrange(g9,g10,g11, ncol=3)

#no outlier detected

#Confirm using boxplot.stat method to see whether outlier exists
for (i in cnames) {
  print(i)
  val = data[,i][data[,i] %in% boxplot.stats(data[,i])$out]
  print(length(val))
  print(val)
}

#lets go Correlation plot
library(corrgram)
corrgram(na.omit(data))
dim(data)
corrgram(data[, cnames],order = F, upper.panel = panel.pie, text.panel = panel.txt, main = "correlation plot" )


#Here we can see in correlation plot that BMI and weight are highly positively correlated
#so we can drop either of the two variables bcz they are having  almost similar characteristics.


# Select the relevant numerical features

num_vars = c("Transportation.expense", "Distance.from.Residence.to.work", "Service.time", "Age", "Work.load.Average.day","Hit.target", "Son", "Pet", "Weight", "Height")
cnames
names(data)


#now, anova test regarding dimension reduction

install.packages("lsr")

library(lsr)

anova_test = aov(Absenteeism.time.in.hours ~ ID + Day.of.the.week + Education + Social.smoker + Social.drinker + Reason.for.absence + Seasons + Month.of.absence + Disciplinary.failure, data = data)
summary(anova_test)


#Dimension reduction
data_sorted = subset(data,select = -c(ID,Body.mass.index, Education, Social.smoker, Social.drinker, Seasons, Disciplinary.failure))


#feature selection

#copy of data set 
features_sorted = data_sorted

write.csv(data_sorted,"Sorted_Features.csv", row.names = F)


colnames(data_sorted)

#draw histogram on random variables to check if the distributions are normal

hist(data_sorted$Hit.target)
hist(data_sorted$Work.load.Average.day.)
hist(data_sorted$Weight)
hist(data_sorted$Transportation.expense)

#we will choose normalisation instead of standardisation bcz variables are not normally distributed

print(sapply(data_sorted ,is.numeric))
num_v = c("Transportation.expense", "Distance.from.Residence.to.Work","Service.time","Age","Work.load.Average.day.","Hit.target","Son","Pet","Weight","Height")
num_v

#feature scaling 
for (i in num_v) {
  print(i)
  data_sorted[,i] = ((data_sorted[,i] - min(data_sorted[,i])) /
                         (max(data_sorted[,i]) - min(data_sorted[,i])))
  
}

colnames(data_sorted)

#now we have normalized datastored in data_sorted
#Write this normalized data in csv form 
write.csv(data_sorted,"normalized_data_sorted.csv", row.names = F)

#######clean the environment except the data_sorted
library(DataCombine)
rmExcept(c("data_sorted","features_sorted"))


##as we have regression problem statement, so we will use decision trees for regression 
library("rpart")
library(MASS)
train_index = sample(1:nrow(data_sorted), 0.8* nrow(data_sorted))

train = data_sorted[train_index,]
test = data_sorted[-train_index,]

##model development (decision tree regression model)

#reg_model = rpart(Absenteeism.time.in.hours ~. , data = train, method = "anova")


#predicting reg_model for test cases
#predictions = predict(reg_model , test[,-14])

#RMSE OR MSEis the technique which can be used to evaluate the performance of a regression model

#here, i will use root mean square error technique to evaluate the performance of the model, moreover the data is a time series data

#library("DMwR")
#RMSE = regr.eval(test[,14], predictions, stats = 'rmse')
#RMSE #14.92

##accuracy = 85.02%
##  Thus in Decision tree regression model the error is 14.92 which tells that our model is 85.08% accurate____

#lets try with random forest regression model
###+____________Random Forest model+__
library("randomForest")
RF_model = randomForest(Absenteeism.time.in.hours~. , train, importance = TRUE,  ntree=100)

#Extract the rules generated as a result of random Forest model
library("inTrees")
rules_list = RF2List(RF_model)

#Extract rules from rules_list
rules = extractRules(rules_list, train[,-14])
rules[1:2,]


#Convert the rules in readable format
read_rules = presentRules(rules,colnames(train))
read_rules[1:2,]


#Determining the rule metric
rule_metric = getRuleMetric(rules, train[,-14], train$Absenteeism.time.in.hours)
rule_metric[1:2,]

#Prediction of the target variable data using the random Forest model
RF_prediction = predict(RF_model,test[,-14])
RMSE_RF = regr.eval(test[,13], RF_prediction, stats = 'rmse')
#RMSE = 7.88
#Accuracy = 92.12%

#Thus the error rate in Random Forest Model is 7.88% and the accuracy of the model is 100-7.88 = 92.12%.

###________________ LINEAR REGRESSION _______________________________

#library("usdm")
#LR_data_select = subset(data_sorted ,select = -c(Reason.for.absence,Day.of.the.week))
#colnames(LR_data_select)
#vif(LR_data_select[,-12])
#vifcor(LR_data_select[,-12], th=0.9)

####Execute the linear regression model over the data
#lr_model = lm(Absenteeism.time.in.hours~. , data = train)

#summary(lr_model)
#colnames(test)

###___Multiple R-squared:  0.2674,	Adjusted R-squared:  0.1953, which means our target variable can explain 26.74% of variance which is not acceptable.
#Predict the data 
#LR_predict_data = predict(lr_model, test[,1:13])

#Calculate MAPE
#MAPE(test[,14], LR_predict_data)
#library(Matrix)
#rmse(test[,14],LR_predict_data)

##linear regression model works best for continuous variables, but here in LR_data_select we hav
##________ Till here we have implemented Decision Tree, Random Forest and Linear Regression. Among all of these Random Forest is having highest accuracy.

#__ To calculate loss month wise we need to include month of absence variable again in our data set 
# LOSS = Work.load.average.per.day * Absenteeism.time.in.hours

data = features_sorted

colnames(data)
data$loss = data$Work.load.Average.day. * data$Absenteeism.time.in.hours

i=1


# NOW calculate Month wise loss encountered due to absenteeism of employees 

#Calculate loss in january
loss_jan = as.data.frame(data$loss[data$Month.of.absence %in% 1])
names(loss_jan)[1] = "Loss"
sum(loss_jan[1])
write.csv(loss_jan,"jan_loss.csv", row.names = F)

#Calculate loss in febreuary
loss_feb = as.data.frame(data$loss[data$Month.of.absence %in% 2])
names(loss_feb)[1] = "Loss"
sum(loss_feb[1])
write.csv(loss_feb,"feb_loss.csv", row.names = F)

#Calculate loss in march

loss_march = as.data.frame(data$loss[data$Month.of.absence %in% 3])
names(loss_march)[1] = "Loss"
sum(loss_march[1])
write.csv(loss_march,"march_loss.csv", row.names = F)

#Calculate loss in april

loss_apr = as.data.frame(data$loss[data$Month.of.absence %in% 4])
names(loss_apr)[1] = "Loss"
sum(loss_apr[1])
write.csv(loss_apr,"apr_loss.csv", row.names = F)


#calculate loss in may

loss_may = as.data.frame(data$loss[data$Month.of.absence %in% 5])
names(loss_may)[1] = "Loss"
sum(loss_may[1])
write.csv(loss_may,"may_loss.csv", row.names = F)


#calculate in june

loss_jun = as.data.frame(data$loss[data$Month.of.absence %in% 6])
names(loss_jun)[1] = "Loss"
sum(loss_jun[1])
write.csv(loss_jun,"jun_loss.csv", row.names = F)

#Calculate loss in july

loss_jul = as.data.frame(data$loss[data$Month.of.absence %in% 7])
names(loss_jul)[1] = "Loss"
sum(loss_jul[1])
write.csv(loss_jul,"jul_loss.csv", row.names = F)

#calculate loss in august

loss_aug = as.data.frame(data$loss[data$Month.of.absence %in% 8])
names(loss_aug)[1] = "Loss"
sum(loss_aug[1])
write.csv(loss_aug,"aug_loss.csv", row.names = F)

#Calculate loss in september

loss_sep = as.data.frame(data$loss[data$Month.of.absence %in% 9])
names(loss_sep)[1] = "Loss"
sum(loss_sep[1])
write.csv(loss_sep,"sep_loss.csv", row.names = F)

#calculate loss in october

loss_oct = as.data.frame(data$loss[data$Month.of.absence %in% 10])
names(loss_oct)[1] = "Loss"
sum(loss_oct[1])
write.csv(loss_oct,"oct_loss.csv", row.names = F)

#calculate loss in november

loss_nov = as.data.frame(data$loss[data$Month.of.absence %in% 11])
names(loss_nov)[1] = "Loss"
sum(loss_nov[1])
write.csv(loss_nov,"nov_loss.csv", row.names = F)

#calculate loss in december
loss_dec = as.data.frame(data$loss[data$Month.of.absence %in% 12])
names(loss_dec)[1] = "Loss"
sum(loss_dec[1])
write.csv(loss_dec,"dec_loss.csv", row.names = F)

