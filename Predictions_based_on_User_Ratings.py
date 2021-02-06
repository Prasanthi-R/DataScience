#!/usr/bin/env python
# coding: utf-8

# In[512]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import math
#%matplotlib inline
sns.set_style("whitegrid")
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

raw_data = pd.read_csv("/Users/avanish/Desktop/Coefficient_Assignment/predictions_by_rating.csv")


# In[513]:


raw_data.head()


# In[514]:


raw_data.describe()


# In[515]:


raw_data.info()


# In[516]:


raw_data.isnull().sum()


# In[517]:


sns.pairplot(raw_data)


# In[518]:


sns.countplot(raw_data["Type"],palette='plasma')


# In[519]:


sns.countplot(raw_data["Year"],palette='magma')


# In[520]:


sns.distplot(raw_data["Year"])


# In[521]:


raw_data['Rating'].hist(bins=50)
plt.show()


# In[522]:


sns.boxplot(raw_data["Rating"],palette='BuGn')


# In[523]:


sns.distplot(raw_data["Price"])


# In[524]:


sns.boxplot(raw_data["Price"],palette='OrRd')


# In[525]:


sns.distplot(raw_data["Reviews"])


# In[526]:


sns.boxplot(raw_data["Price"],palette='PuRd')


# In[527]:


raw_data.describe()


# In[528]:


raw_data.info()


# In[529]:


raw_data.drop("Name", axis=1, inplace=True)


# In[530]:


raw_data.info()


# In[531]:


dummies=pd.get_dummies(raw_data, drop_first=True, columns=['Type'])
dummies.head()


# In[532]:


dummies.drop("Author", axis=1, inplace=True)


# In[533]:


dummies.columns


# In[534]:


plt.figure(figsize=(9,7))
correlation = dummies.corr()
sns.heatmap(correlation,xticklabels=correlation.columns,yticklabels=correlation.columns,annot=True)
plt.show()


# In[535]:


y=dummies['Rating']


# In[536]:


X=dummies.drop(["Rating"], axis = 1)


# In[537]:


y


# In[538]:


X


# In[539]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)


# In[540]:


lmodel=LinearRegression()


# In[541]:


lmodel.fit(X_train,y_train)


# In[542]:


y_pred=lmodel.predict(X_test)


# In[543]:


y_pred


# In[544]:


print(y_pred.max())


# In[545]:


print(y_pred.min())


# In[546]:


plt.scatter(y_test, y_pred)


# In[547]:


print("Mean Absolute Error: " + str(mean_absolute_error(y_test, y_pred)))
print("Mean Squared Error: " + str(mean_squared_error(y_test, y_pred)))
print("Root Mean Squareed Error: " + str(math.sqrt(mean_squared_error(y_test, y_pred))))
print("R2: " + str(r2_score(y_test, y_pred)))


# In[548]:


reg_tree = DecisionTreeRegressor(max_depth=2)
reg_tree.fit(X, y)
y_regtree_pred = reg_tree.predict(X_test)


# In[549]:


print("Decision Tree with depth 2:")
print("Mean Absolute Error: " + str(mean_absolute_error(y_test, y_regtree_pred)))
print("Mean Square Error: " + str(mean_squared_error(y_test, y_regtree_pred)))
print("Root Mean Square Error: " + str(math.sqrt(mean_squared_error(y_test, y_regtree_pred))))
print("R2: " + str(r2_score(y_test, y_regtree_pred)))


# In[550]:


rfr_tree = RandomForestRegressor(max_depth=2, random_state=0)
rfr_tree.fit(X, y)
y_rfr_pred = rfr_tree.predict(X_test)


# In[551]:


print("Random Forest with depth 2:")
print("Mean Absolute Error: " + str(mean_absolute_error(y_test, y_rfr_pred)))
print("Mean Squared Error: " + str(mean_squared_error(y_test, y_rfr_pred)))
print("Root Mean Squared Error: " + str(math.sqrt(mean_squared_error(y_test, y_rfr_pred))))
print("R2: " + str(r2_score(y_test, y_rfr_pred)))

