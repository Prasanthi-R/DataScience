#!/usr/bin/env python
# coding: utf-8

# In[44]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import math
sns.set_style("whitegrid")
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


# In[4]:


data=pd.read_csv('/Users/avanish/Desktop/Coefficient_Assignment/Insurance_cost_prediction.csv')
data.head()


# In[5]:


data.describe()


# In[6]:


sns.distplot(data['Age'])


# In[7]:


sns.scatterplot(x="Age", y="BMI", hue='Gender',data=data)


# In[8]:


sns.boxplot(x='Gender',y='Age',data=data)


# In[9]:


f= plt.figure(figsize=(12,5))

ax=f.add_subplot(121)
sns.distplot(data[(data.Gender == 'male')]["Age"],color='b',ax=ax)
ax.set_title('Distribution of ages of male')

ax=f.add_subplot(122)
sns.distplot(data[(data.Gender == 'female')]['Age'],color='r',ax=ax)
ax.set_title('Distribution of ages of female')


# In[10]:


sns.boxplot(x='Smoker',y='Age',data=data)


# In[11]:


f= plt.figure(figsize=(12,5))

ax=f.add_subplot(121)
sns.distplot(data[(data.Smoker == 'yes')]["Age"],ax=ax)
ax.set_title('Distribution of ages of smoker')

ax=f.add_subplot(122)
sns.distplot(data[(data.Smoker == 'no')]['Age'],ax=ax)
ax.set_title('Distribution of ages of non smoker')


# In[12]:


sns.catplot(x="Smoker", kind="count",hue = 'Gender',palette='magma',data=data)


# In[13]:


sns.boxplot(x='Smoker',y='Price',data=data)


# In[14]:


f= plt.figure(figsize=(12,5))

ax=f.add_subplot(121)
sns.distplot(data[(data.Smoker == 'yes')]['Price'],ax=ax)
ax.set_title('Distribution of charges of smoker')

ax=f.add_subplot(122)
sns.distplot(data[(data.Smoker == 'no')]['Price'],ax=ax)
ax.set_title('Distribution of charges of non smoker')


# In[15]:


sns.boxplot(x='Smoker',y='Price',data=data[(data.Age>=18)&(data.Age<=22)])


# In[16]:


sns.scatterplot(x="Age", y="Price", data=data[data.Smoker=='yes'],color='purple')


# In[17]:


sns.scatterplot(x="Age", y="Price", data=data[data.Smoker=='no'])


# In[18]:


sns.boxplot(x='Gender',y='BMI',data=data)


# In[19]:


f= plt.figure(figsize=(12,5))

ax=f.add_subplot(121)
sns.distplot(data[(data.Gender == 'male')]["BMI"],color='b',ax=ax)
ax.set_title('Distribution of bmi of male')

ax=f.add_subplot(122)
sns.distplot(data[(data.Gender == 'female')]['BMI'],color='r',ax=ax)
ax.set_title('Distribution of bmi of female')


# In[20]:


f= plt.figure(figsize=(12,5))

ax=f.add_subplot(121)
sns.distplot(data[(data.Smoker == 'yes')]["BMI"],color='b',ax=ax)
ax.set_title('Distribution of bmi of smoker')

ax=f.add_subplot(122)
sns.distplot(data[(data.Smoker == 'no')]['BMI'],color='r',ax=ax)
ax.set_title('Distribution of bmi of non smoker')


# In[21]:


sns.catplot(x="Geography", kind="count",palette='plasma',data=data)


# In[22]:


sns.catplot(x="Geography", kind="count",hue = 'Gender',palette='viridis',data=data)


# In[23]:


sns.lmplot(x="BMI", y="Price",data=data)


# In[24]:


sns.catplot(x="Children", kind="count",palette='rainbow',data=data)


# In[25]:


sns.lmplot(x="Children", y="Price",data=data)


# In[26]:


sns.lmplot(x="Children", y="Price", hue='Smoker',data=data)


# In[27]:


plt.figure(figsize=(10,8))
correlation = data.corr()
sns.heatmap(correlation,xticklabels=correlation.columns,yticklabels=correlation.columns,annot=True)
plt.show()


# In[28]:


le = LabelEncoder()
le.fit(data.Gender.drop_duplicates()) 
data.Gender = le.transform(data.Gender)


# In[29]:


le.fit(data.Smoker.drop_duplicates()) 
data.Smoker = le.transform(data.Smoker)


# In[30]:


le.fit(data.Geography.drop_duplicates()) 
data.region = le.transform(data.Geography)


# In[31]:


data.head()


# In[32]:


x = data.drop(['Price','Geography'], axis = 1)
y = data.Price


# In[58]:


x_train,x_test,y_train,y_test = train_test_split(x,y, random_state = 0)


# In[59]:


lre = linear_model.LinearRegression()
lre.fit(x_train,y_train)


# In[60]:


y_pred=lre.predict(x_test)


# In[61]:


lre.score(x_test,y_test)


# In[53]:


print("Mean Absolute Error: " + str(mean_absolute_error(y_test, y_pred)))
print("Mean Squared Error: " + str(mean_squared_error(y_test, y_pred)))
print("Root Mean Squareed Error: " + str(math.sqrt(mean_squared_error(y_test, y_pred))))
print("R2: " + str(r2_score(y_test, y_pred)))


# In[78]:


reg_tree = DecisionTreeRegressor(max_depth=2)
reg_tree.fit(x, y)
y_regtree_pred = reg_tree.predict(x_test)


# In[79]:


print("Decision Tree with depth 2:")
print("Mean Absolute Error: " + str(mean_absolute_error(y_test, y_regtree_pred)))
print("Mean Square Error: " + str(mean_squared_error(y_test, y_regtree_pred)))
print("Root Mean Square Error: " + str(math.sqrt(mean_squared_error(y_test, y_regtree_pred))))
print("R2: " + str(r2_score(y_test, y_regtree_pred)))


# In[64]:


Rf = RandomForestRegressor(n_estimators = 100,
                              criterion = 'mse',
                              random_state = 1,
                              n_jobs = -1)
Rf.fit(x_train,y_train)
Rf_train_pred = Rf.predict(x_train)
Rf_test_pred = Rf.predict(x_test)


r2_score(y_test,Rf_test_pred)


# In[ ]:




