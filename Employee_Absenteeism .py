
# coding: utf-8

# In[1]:


# Importing Libraries
import pandas as pd
from pandas import Timestamp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from fancyimpute import KNN 
import os
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from scipy import stats
import statsmodels.api as sm
import seaborn as sns
import matplotlib.gridspec as gridspec 
get_ipython().run_line_magic('matplotlib', 'inline')
from ggplot import *


# In[2]:


#set working directory
os.chdir("S:/analytics/Projects/Employee Absenteeism python")


# In[3]:


os.getcwd()


# In[4]:


#loading the data 
data = pd.read_excel("Absenteeism_at_work_Project.xls")


# In[5]:


data


# In[10]:


data.dtypes


# In[8]:


data.shape


# # data pre processing

# In[9]:


#Exploratory Data Analysis
data['Reason for absence']=data['Reason for absence'].astype(object)
data['Month of absence']=data['Month of absence'].astype(object)
data['Day of the week']=data['Day of the week'].astype(object)
data['Seasons']=data['Seasons'].astype(object)
data['Service time']=data['Service time'].astype(object)
data['Hit target']=data['Hit target'].astype(object)
data['Disciplinary failure']=data['Disciplinary failure'].astype(object)
data['Education']=data['Education'].astype(object)
data['Son']=data['Son'].astype(object)
data['Social drinker']=data['Social drinker'].astype(object)
data['Social smoker']=data['Social smoker'].astype(object)
data['Pet']=data['Pet'].astype(object)


# In[16]:


#finding the number of missing values
data.isnull().sum().sum()


# In[17]:


# Categorising into continuous and categorical
continuous_vars = ['Distance from Residence to Work', 'Service time', 'Age', 'Work load Average/day ', 'Transportation expense',
       'Hit target', 'Weight', 'Height', 'Body mass index', 'Absenteeism time in hours']

categorical_vars = ['ID','Reason for absence','Month of absence','Day of the week',
                     'Seasons','Disciplinary failure', 'Education', 'Social drinker',
                     'Social smoker', 'Pet', 'Son']


# # missing value analysis

# In[19]:


#Creating dataframe with missing values present in each variable
missing_val = pd.DataFrame(data.isnull().sum()).reset_index()


# In[20]:


missing_val


# In[21]:


#Renaming variables of missing_val dataframe
missing_val = missing_val.rename(columns = {'index': 'Variables', 0: 'Missing_percentage'})


# In[22]:


#Calculating percentage missing value
missing_val['Missing_percentage'] = (missing_val['Missing_percentage']/len(data))*100


# In[25]:


missing_val


# In[24]:


# Sorting missing_val in Descending order
missing_val = missing_val.sort_values('Missing_percentage', ascending = False).reset_index(drop = True)


# # Imputing missing values

# In[26]:


data['Body mass index'].iloc[12]


# In[27]:


#creating missing value
data['Body mass index'].iloc[12] = np.nan


# In[28]:


data['Body mass index'].iloc[12]


# In[29]:


# Checking for "Body mass index" column
# Actual value = 23
# Mean = 26.68
# Median = 25
# KNN = 23.20


# In[30]:


# Imputing with mean
#data['Body mass index'] = data['Body mass index'].fillna(data['Body mass index'].mean())
# data['Body mass index'].iloc[12]

# Imputing with median
#data['Body mass index'] = data['Body mass index'].fillna(data['Body mass index'].median())
# data['Body mass index'].iloc[12]


# In[31]:


#Apply KNN imputation algorithm
data = pd.DataFrame(KNN(k = 5).complete(data), columns = data.columns)


# In[32]:


data['Body mass index'].iloc[12]


# In[33]:


#confirming for missing values
data.isnull().sum()


# # outlier analysis

# In[41]:


# Ploting BoxPlot of continuous variables
plt.boxplot([data['Transportation expense']])
plt.xlabel('Transportation expense')
plt.title("BoxPlot of Variables 'Transportation expense' ")
plt.ylabel('Values')


# In[40]:


# Ploting BoxPlot of continuous variables
plt.boxplot(data['Height'])
plt.xlabel('Height')
plt.title("BoxPlot of Variables 'Height' ")
plt.ylabel('Values')


# In[42]:


plt.boxplot(data['Work load Average/day '])
plt.xlabel("Work load Average/day ")
plt.title("BoxPlot of Variable 'Work load Average/day '")
plt.ylabel('Values')


# In[43]:


plt.boxplot([ data['Distance from Residence to Work'], data['Service time'], data['Age'], data['Hit target'], data['Weight'], data['Body mass index']])
plt.xlabel(['1. Distance from Residence to Work', '2. Service time', '3. Age', '4. Hit target', '5. Weight', '6. Body mass index'])
plt.title("BoxPlot of rest of the Variables")
plt.ylabel('Values')


# In[44]:


# From the above boxplot we can clearly see that in variables 'Distance from Residence to Work', 'Weight' and 'Body mass index'
# there is no outlier


# In[45]:


# list of variables which doesn't have outlier
neglect = ['Distance from Residence to Work', 'Weight', 'Body mass index']


# In[50]:


#detect and replace outliers with NAs
for i in continuous_vars:
    #getting 75 & 25percentile of variable 'i'
    q75, q25 = np.percentile(data[i], [75,25])
    #calculating inter quartile range
    iqr = q75 - q25
    
    #calculating upper fence and lower fence
    minimum = q25 - (iqr*1.5)
    maximum = q75 + (iqr*1.5)
    
    


# In[52]:


maximum


# In[55]:


# Replacing all the outliers value to NA
data.loc[data[i]< minimum,i] = np.nan
data.loc[data[i]> maximum,i] = np.nan


# In[57]:


#checking for NAs
data.isnull().sum().sum()


# In[58]:


# Imputing missing values with KNN
data = pd.DataFrame(KNN(k = 5).complete(data), columns = data.columns)


# In[59]:


#checking for nas after knn imputation
data.isnull().sum().sum()


# # Feature selection

# In[60]:


from scipy.stats import chi2_contingency
import seaborn as sns


# In[61]:


#correlation analysis for continuous_vars
#correlation plot
df_corr = data.loc[:, continuous_vars]


# In[62]:


df_corr.shape


# In[75]:


#set the width and height of correlation plot
f, ax = plt.subplots(figsize=(7,5))

#generate correlation matrix
corr = df_corr.corr()

#plot with seaborn library
sns.heatmap(corr, mask = np.zeros_like(corr, dtype=np.bool), cmap = sns.diverging_palette(220, 10, as_cmap = True),
           square = True, ax=ax)
plt.savefig('correlation.png')


# In[77]:


#Chisquare test of independence
#Save categorical variables
catorical_vars = ["Reason for absence", "Month of absence", "Day of the week", "Seasons", "Service time", "Disciplinary failure", "Education", "Son", "Social drinker","Social smoker","Pet"]


# In[78]:


continuous_vars , categorical_vars


# In[81]:


#loop for chi square values
for i in categorical_vars:
    print(i)
    chi2, p, dof, ex = chi2_contingency(pd.crosstab(data['Absenteeism time in hours'], data[i]))
    print(p)


# In[87]:


#as we can see weight and body mass index are highly correlated with each other
# Droping the variables which has redundant information
to_drop = ['Weight']
data = data.drop(to_drop, axis = 1)
data = data.drop([ 'ID', 'Education', 'Social smoker', 'Pet'], axis=1)


# In[90]:


# Updating the Continuous Variables and Categorical Variables after droping some variables
continuous_vars = [i for i in continuous_vars if i not in to_drop]
categorical_vars = [i for i in categorical_vars if i not in to_drop]


# In[88]:


clean_data = data.copy()


# # Feature scaling

# In[98]:


##normality check
for i in continuous_vars:
    if i == 'Absenteeism time in hours':
        continue
    sns.distplot(data[i],bins = 'auto')
    plt.title("Checking Distribution for Variable "+str(i))
    plt.ylabel("Density")
    plt.show()


# In[108]:


#as we can see data is not normally distributed , we will go for normalization 
for i in continuous_vars:
    print(i)
    data[i] = (data[i] - data[i].min())/(data[i].max()-data[i].min())


# In[109]:


data


# # data sampling

# In[113]:


from sklearn.cross_validation import train_test_split
#Divide data into train and test
train, test = train_test_split(data, test_size=0.20, random_state=42)


# # machine learning models

# In[115]:


from sklearn.tree import DecisionTreeRegressor
#Decision tree for regression
fit_DT = DecisionTreeRegressor(max_depth=2).fit(train.iloc[:,0:9], train.iloc[:,9])


# In[116]:


fit_DT


# In[117]:


#checking for any missing valuses that has leeked in
np.where(data.values >= np.finfo(np.float64).max)


# In[118]:


np.isnan(data.values.any())


# In[119]:


test = test.fillna(train.mean())


# In[120]:


#Decision tree for regression
fit_DT = DecisionTreeRegressor(max_depth=2).fit(train.iloc[:,0:15], train.iloc[:,15])


# In[121]:


data.shape


# In[122]:


#Apply model on test data
predictions_DT = fit_DT.predict(test.iloc[:,0:15])


# In[123]:


predictions_DT


# In[124]:


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


# In[125]:


rmse(test.iloc[:,15], predictions_DT)


# In[126]:


#rmse using DT = 0.16972039562573937


# In[127]:


#Divide data into train and test
X = data.values[:, 0:15]
Y = data.values[:,15]

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.2)


# In[128]:


X_test.shape


# In[133]:


from sklearn.ensemble import RandomForestRegressor


# In[134]:


# Building model on top of training dataset
fit_RF = RandomForestRegressor(n_estimators = 500).fit(X_train,y_train)


# In[135]:


fit_RF


# In[146]:


# Calculating RMSE for training data to check for over fitting
RF_predictions_train = fit_RF.predict(X_train)
rmse_for_train = np.sqrt(mean_squared_error(y_train,RF_predictions_train))


# In[145]:


# Calculating RMSE for test data to check accuracy
RF_predictions_test = fit_RF.predict(X_test)
rmse_for_test =np.sqrt(mean_squared_error(y_test,RF_predictions_test))


# In[143]:


rmse_for_train


# In[139]:


rmse_for_test


# In[140]:


print("Root Mean Squared Error For Training data = "+str(rmse_for_train))
print("Root Mean Squared Error For Test data = "+str(rmse_for_test))


# # Linear regression

# In[147]:


# Importing libraries for Linear Regression
from sklearn.linear_model import LinearRegression

# Building model on top of training dataset
fit_LR = LinearRegression().fit(X_train , y_train)

# Calculating RMSE for training data to check for over fitting
LR_pred_train = fit_LR.predict(X_train)
rmse_for_train = np.sqrt(mean_squared_error(y_train,LR_pred_train))

# Calculating RMSE for test data to check accuracy
LR_pred_test = fit_LR.predict(X_test)
rmse_for_test =np.sqrt(mean_squared_error(y_test,LR_pred_test))

print("Root Mean Squared Error For Training data = "+str(rmse_for_train))
print("Root Mean Squared Error For Test data = "+str(rmse_for_test))


# # data visualization

# In[148]:


#Visualising Important components

ggplot(clean_data, aes(x='Month of absence', y='Absenteeism time in hours')) +    geom_bar(fill= "SlateBlue") +    scale_color_brewer(type='diverging', palette=4) +    xlab("Month") + ylab("absenteeism in hr") + ggtitle("Absenteeism Analysis") + theme_bw()


# In[152]:


ggplot(clean_data, aes(x='Age', y='Absenteeism time in hours')) +    geom_bar(fill= "Gray") +    scale_color_brewer(type='diverging', palette=5) +    xlab("Age") + ylab("absenteeism in hr") + ggtitle("Absenteeism Analysis") + theme_bw()


# In[153]:


ggplot(clean_data, aes(x='Social drinker', y='Absenteeism time in hours')) +    geom_bar(fill= "Magenta") +    scale_color_brewer(type='diverging', palette=5) +    xlab("Social drinker") + ylab("absenteeism in hr") + ggtitle("Absenteeism Analysis") + theme_bw()


# In[154]:


ggplot(clean_data, aes(x='Body mass index', y='Absenteeism time in hours')) +    geom_bar(fill= "Yellow") +    scale_color_brewer(type='diverging', palette=5) +    xlab("Body mass index") + ylab("absenteeism in hr") + ggtitle("Absenteeism Analysis") + theme_bw()


# In[175]:


sns.stripplot(x="Reason for absence", y="Absenteeism time in hours", data=data,size = 10);
plt.savefig('Reason for absence.png')


# In[158]:


sns.stripplot(x="Month of absence", y="Absenteeism time in hours", data=data, size = 5);
plt.savefig('Month of absence.png')


# In[160]:


sns.stripplot(x="Day of the week", y="Absenteeism time in hours", data=data, size = 10);
plt.savefig('Day of the week.png')


# In[161]:


sns.stripplot(x="Seasons", y="Absenteeism time in hours", data=data, size = 10);
plt.savefig('Seasons.png')


# In[165]:


ggplot(clean_data, aes(x='Transportation expense', y='Absenteeism time in hours')) +    geom_bar(fill= "Black") +    scale_color_brewer(type='diverging', palette=5) +    xlab("Transportation expense") + ylab("absenteeism in hr") + ggtitle("Absenteeism Analysis") + theme_bw()


# In[172]:


ggplot(clean_data, aes(x='Distance from Residence to Work', y='Absenteeism time in hours')) +    geom_bar(fill= "Green") +    scale_color_brewer(type='diverging', palette=5) +    xlab("Distance from Residence to Work") + ylab("absenteeism in hr") + ggtitle("Absenteeism Analysis") + theme_bw()


# In[168]:


sns.stripplot(x="Disciplinary failure", y="Absenteeism time in hours", data=data);
plt.savefig('Disciplinary failure.png')


# In[173]:


ggplot(clean_data, aes(x='Son', y='Absenteeism time in hours')) +    geom_bar(fill= "Purple") +    scale_color_brewer(type='diverging', palette=5) +    xlab("Son") + ylab("absenteeism in hr") + ggtitle("Absenteeism Analysis") + theme_bw()


# In[174]:


clean_data.columns

