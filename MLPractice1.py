#!/usr/bin/env python
# coding: utf-8

# In[108]:


##I. Preparation
##I.0 Imports
##I.1 Read the data
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import math
raw_data = pd.read_csv("/Users/avanish/Downloads/bestsellers with categories.csv")


# In[109]:


raw_data.head()


# In[110]:


raw_data.describe()


# In[111]:


raw_data.info()


# In[112]:


raw_data.drop("Name", axis=1, inplace=True)


# In[113]:


raw_data.info()


# In[114]:


##1. Data Understanding
##1.1 Numerical Data
##User Rating


# In[115]:


sns.distplot(raw_data["User Rating"])


# In[116]:


sns.boxplot(raw_data["User Rating"])


# In[117]:


raw_data["User Rating"].quantile(0.1)


# In[118]:


##Most User Ratings are between 4.3 and 4.9.
##User Ratings below 4.1 are classified as outliers.


# In[119]:


##Price


# In[120]:


sns.distplot(raw_data["Price"])


# In[121]:


sns.boxplot(raw_data["Price"])


# In[122]:


raw_data["Price"].quantile(0.98)


# In[123]:


##There are books, which are significantly more expensive than the average.


# In[124]:


##Year


# In[125]:


sns.distplot(raw_data["Year"])


# In[126]:


sns.countplot(raw_data["Year"])


# In[127]:


##Genre


# In[128]:


sns.countplot(raw_data["Genre"])


# In[129]:


##Fiction Books are underepresented.


# In[130]:


##Reviews


# In[131]:


sns.distplot(raw_data["Reviews"])


# In[132]:


sns.boxplot(raw_data["Reviews"])


# In[133]:


raw_data["Reviews"].quantile(0.97)


# In[134]:


##Several books have way more reviews than the average of the books.


# In[135]:


##Categorial Data


# In[136]:


raw_data["Author"].value_counts()


# In[137]:


##2. Data Peparation


# In[138]:


data_prep = raw_data.loc[(raw_data["Price"] < raw_data["Price"].quantile(0.98)) & (raw_data["User Rating"] > raw_data["User Rating"].quantile(0.01)) & (raw_data["Reviews"] < raw_data["Reviews"].quantile(0.97)) ]


# In[139]:


data_prep.describe()


# In[140]:


data_prep.info()


# In[141]:


##2.2 One-hot encoding


# In[142]:


data_prep.drop("Author", axis=1, inplace=True)
data_enc = pd.get_dummies(data_prep, drop_first = True)


# In[143]:


data_enc


# In[144]:


##2.3 Scale the numerical features


# In[145]:


# target variables and predictors
y = data_enc["User Rating"]
X = data_enc.drop(["User Rating"], axis = 1)


# In[146]:


# Scale between 0 and 1
from sklearn.preprocessing import MinMaxScaler

cols = []

for col in X.columns:
    cols.append(col)
    

num_features = cols
scaler = MinMaxScaler(feature_range = (0,1))


X[num_features] = scaler.fit_transform(X[num_features])
X.head()


# In[147]:


##2.4 Check correlation and multi-collinearity


# In[ ]:


corr_mat = data_enc.corr()
sns.heatmap(corr_mat, annot = True)
plt.show()


# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor 
  
# the independent variables set 
vif_test = X
  
# VIF dataframe 
vif_data = pd.DataFrame() 
vif_data["feature"] = vif_test.columns 
  
# calculating VIF for each feature 
vif_data["VIF"] = [variance_inflation_factor(vif_test.values, i) 
                          for i in range(len(vif_test.columns))] 
  
print(vif_data)


# In[150]:


##No multi-collinearity as none of the features have an VIF of over 10.


# In[151]:


##3. Modeling¶
##3.0 Build Training- and Testset


# In[152]:


# Split data in test and training set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=110)


# In[153]:


##3.1 Linear Regression


# In[154]:


from sklearn import linear_model

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, y_train)

y_hat = regr.predict(X_test)


# In[155]:


##Evaluation


# In[156]:


print(y.max())
print(y.min())


# In[157]:


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
print("Mean Absolute Error: " + str(mean_absolute_error(y_test, y_hat)))
print("Mean Square Error: " + str(mean_squared_error(y_test, y_hat)))
print("Root Mean Square Error: " + str(math.sqrt(mean_squared_error(y_test, y_hat))))
print("R2: " + str(r2_score(y_test, y_hat)))


# In[158]:


##3.2 Decision Tree


# In[159]:


from sklearn.tree import DecisionTreeRegressor

regr_tree_2 = DecisionTreeRegressor(max_depth=2)
regr_tree_2.fit(X, y)
y_hat_tree_2 = regr_tree_2.predict(X_test)


# In[160]:


print("Decision Tree with depth 2:")
print("Mean Absolute Error: " + str(mean_absolute_error(y_test, y_hat_tree_2)))
print("Mean Square Error: " + str(mean_squared_error(y_test, y_hat_tree_2)))
print("Root Mean Square Error: " + str(math.sqrt(mean_squared_error(y_test, y_hat_tree_2))))
print("R2: " + str(r2_score(y_test, y_hat_tree_2)))


# In[161]:


##Hyperparameter Tuning Decision Tree


# In[162]:


from sklearn.model_selection import GridSearchCV


# In[163]:


# define the parameter values that should be searched
k_range = list(range(1, 11))
print(k_range)


# In[164]:


# create a parameter grid: map the parameter names to the values that should be searched
param_grid = dict(max_depth=k_range)
print(param_grid)


# In[165]:


# instantiate the grid
grid = GridSearchCV(regr_tree_2, param_grid, cv=10, scoring='neg_root_mean_squared_error', return_train_score=False)


# In[166]:


grid.fit(X, y)


# In[167]:


print("Highest score of " + str(grid.best_score_) + " with the parameters: " + str(grid.best_params_) + ".")


# In[168]:


##Best Tree:


# In[169]:


from sklearn.tree import DecisionTreeRegressor

regr_tree_best = DecisionTreeRegressor(max_depth=2)
regr_tree_best.fit(X, y)
y_hat_tree_best = regr_tree_2.predict(X_test)


# In[170]:


print("Best Tree")
print("Absolute Error : " + str(mean_absolute_error(y_test, y_hat_tree_best)))
print("Mean Square Error : " + str(mean_squared_error(y_test, y_hat_tree_best)))
print("Root Mean Square Error : " + str(math.sqrt(mean_squared_error(y_test, y_hat_tree_best))))
print("R2: " + str(r2_score(y_test, y_hat_tree_best)))


# In[171]:


##3.3 Random Forest


# In[172]:


from sklearn.ensemble import RandomForestRegressor
regr_rtree_2 = RandomForestRegressor(max_depth=2, random_state=0)
regr_rtree_2.fit(X, y)
y_hat_rtree_2 = regr_rtree_2.predict(X_test)


# In[173]:


print("Für Random Forest mit Tiefe 2:")
print("Der Mean Absolute Error beträgt: " + str(mean_absolute_error(y_test, y_hat_rtree_2)))
print("Der Mean Square Error beträgt: " + str(mean_squared_error(y_test, y_hat_rtree_2)))
print("Der Root Mean Square Error beträgt: " + str(math.sqrt(mean_squared_error(y_test, y_hat_rtree_2))))
print("Der R2 beträgt: " + str(r2_score(y_test, y_hat_rtree_2)))


# In[174]:


##Hyperparameter Tuning Random Forest


# In[175]:


# define the parameter values that should be searched
k_range = list(range(1, 31))
print(k_range)
n_estimators = list(range(10, 210, 10))
print(n_estimators)


# In[176]:


# create a parameter grid: map the parameter names to the values that should be searched
param_grid = dict(max_depth=k_range, n_estimators=n_estimators, random_state=[101])
print(param_grid)


# In[177]:


# instantiate the grid
grid = GridSearchCV(regr_rtree_2, param_grid, cv=5, scoring='neg_root_mean_squared_error', return_train_score=False)


# In[ ]:


grid.fit(X, y)


# In[ ]:


print("Highest score of " + str(grid.best_score_) + " with the parameters: " + str(grid.best_params_) + ".")


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
regr_rtree_best = RandomForestRegressor(max_depth=2, n_estimators=80, random_state=101)
regr_rtree_best.fit(X, y)
y_hat_rtree_best = regr_rtree_2.predict(X_test)


# In[ ]:


##Best Random Forest:


# In[ ]:


print("Best Random Forest:")
print("Mean Absolute Error: " + str(mean_absolute_error(y_test, y_hat_rtree_best)))
print("Mean Square Error: " + str(mean_squared_error(y_test, y_hat_rtree_best)))
print("Root Mean Square Error: " + str(math.sqrt(mean_squared_error(y_test, y_hat_rtree_best))))
print("R2 : " + str(r2_score(y_test, y_hat_rtree_best)))


# In[ ]:


##3.4 Support Vector Machine


# In[ ]:


from sklearn.svm import SVR
svm_model = SVR()
svm_model.fit(X_train, y_train)
y_hat_svm = svm_model.predict(X_test)


# In[ ]:


print("Support Vector Machine:")
print("Mean Absolute Error: " + str(mean_absolute_error(y_test, y_hat_svm)))
print("Mean Square Error: " + str(mean_squared_error(y_test, y_hat_svm)))
print("Root Mean Square Error: " + str(math.sqrt(mean_squared_error(y_test, y_hat_svm))))
print("R2: " + str(r2_score(y_test, y_hat_svm)))


# In[ ]:


##Hyperparameter Tuning SVM


# In[ ]:


# define the parameter values that should be searched
c_range = list(range(1, 210, 10))
kernels = ["linear", "poly", "rbf", "sigmoid"]


# In[ ]:


# create a parameter grid: map the parameter names to the values that should be searched
param_grid = dict(C=c_range, kernel=kernels)
print(param_grid)


# In[ ]:


# instantiate and fit the grid
grid = GridSearchCV(svm_model, param_grid, cv=5, scoring='neg_root_mean_squared_error', return_train_score=False)
grid.fit(X, y)


# In[ ]:


print("Highest score of " + str(grid.best_score_) + " with the parameters: " + str(grid.best_params_) + ".")


# In[ ]:


svm_model_best = SVR(C=61, kernel="linear")
svm_model_best.fit(X_train, y_train)
y_hat_svm_best = svm_model.predict(X_test)


# In[ ]:


##Best SVM:


# In[ ]:


print("Support Vector Machine:")
print("Absolute Error: " + str(mean_absolute_error(y_test, y_hat_svm_best)))
print("Mean Square Error: " + str(mean_squared_error(y_test, y_hat_svm_best)))
print("Root Mean Square Error: " + str(math.sqrt(mean_squared_error(y_test, y_hat_svm_best))))
print("R2: " + str(r2_score(y_test, y_hat_svm_best)))


# In[ ]:


##3.5 Conclusion
##The best model with the target metric Root Mean Square Error is the Random Forest with depth 2.

