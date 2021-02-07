#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import math
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
sns.set_style("whitegrid")
import warnings
warnings.filterwarnings('ignore')


# In[3]:


dataset = pd.read_csv('/Users/avanish/Desktop/Coefficient_Assignment/Game_Sales.csv')


# In[4]:


dataset.info()


# In[5]:


dataset.head()


# In[6]:


dataset.describe()


# In[7]:


dataset.isnull().values.any()


# In[8]:


print(dataset['Rank'].isnull().values.any())
print(dataset['Name'].isnull().values.any())
print(dataset['Platform'].isnull().values.any())
print(dataset['Year'].isnull().values.any())
print(dataset['Genre'].isnull().values.any())
print(dataset['Publisher'].isnull().values.any())
print(dataset['NA_Sales'].isnull().values.any())
print(dataset['EU_Sales'].isnull().values.any())
print(dataset['JP_Sales'].isnull().values.any())
print(dataset['Other_Sales'].isnull().values.any())
print(dataset['Global_Sales'].isnull().values.any())


# In[9]:


print(dataset['Year'].isnull().sum())
print(dataset['Publisher'].isnull().sum())


# In[10]:


dataset = dataset.dropna(axis=0, subset=['Year','Publisher'])


# In[11]:


dataset.isnull().values.any()


# In[28]:


x = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values
print(x[0])
print(y)


# In[33]:


corr_mat = dataset.corr()
top_corr_features = corr_mat.index
plt.figure(figsize=(10,10))
#Plotting heat map
hp=sns.heatmap(dataset[top_corr_features].corr(),annot=True,linewidths=.5)
plt.show()  


# In[34]:


x = dataset.iloc[:,6:-1].values
print(x[0])


# In[35]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)


# In[37]:


print(x_train)
print(x_test)
print(y_train)
print(y_test)


# In[38]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)


# In[39]:


y_pred = lr.predict(x_test)


# In[40]:


from sklearn.metrics import r2_score
r2_lr = r2_score(y_test,y_pred)
print(r2_lr)


# In[42]:


from sklearn.neighbors import KNeighborsRegressor
knn_range = range(1,11,1)
scores_list = []
for i in knn_range:
    knn_reg = KNeighborsRegressor(n_neighbors=i)
    knn_reg.fit(x_train,y_train)
    y_pred = knn_reg.predict(x_test)
    scores_list.append(r2_score(y_test,y_pred))
plt.plot(knn_range,scores_list,linewidth=2,color='green')
plt.xticks(knn_range)
plt.xlabel('No. of neighbors')
plt.ylabel('r2 score of KNN')
plt.show()    


# In[43]:


knn_reg = KNeighborsRegressor(n_neighbors=7)
knn_reg.fit(x_train,y_train)
y_pred = knn_reg.predict(x_test)
r2_knn = r2_score(y_test,y_pred)
print(r2_knn)


# In[44]:


from sklearn.tree import DecisionTreeRegressor
regressor_Tree = DecisionTreeRegressor(random_state=0)
regressor_Tree.fit(x_train,y_train)


# In[45]:


y_pred = regressor_Tree.predict(x_test)


# In[46]:


r2_tree = r2_score(y_test,y_pred)
print(r2_tree)


# In[47]:


from sklearn.ensemble import RandomForestRegressor
forestRange=range(50,500,50)
scores_list=[]
for i in forestRange: 
    regressor_Forest = RandomForestRegressor(n_estimators=i,random_state=0)
    regressor_Forest.fit(x_train,y_train)
    y_pred = regressor_Forest.predict(x_test)
    scores_list.append(r2_score(y_test,y_pred))
plt.plot(forestRange,scores_list,linewidth=2,color='maroon')
plt.xticks(forestRange)
plt.xlabel('No. of trees')
plt.ylabel('r2 score of Random Forest Reg.')
plt.show()  


# In[48]:


regressor_Forest = RandomForestRegressor(n_estimators=100,random_state=0)
regressor_Forest.fit(x_train,y_train)
y_pred = regressor_Forest.predict(x_test)
r2_forest = r2_score(y_test,y_pred)
print(r2_forest)


# In[49]:


from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(np.reshape(y_train,(len(y_train),1)))
y_test = sc_y.transform(np.reshape(y_test,(len(y_test),1)))


# In[50]:


print(x_train)
print(x_test)
print(y_test)
print(y_train)


# In[51]:


from sklearn.svm import SVR
regressor_SVR = SVR(kernel='linear')
regressor_SVR.fit(x_train,y_train)


# In[52]:


y_pred = regressor_SVR.predict(x_test)


# In[53]:


r2_linearSVR = r2_score(y_test,y_pred)
print(r2_linearSVR)


# In[54]:


from sklearn.svm import SVR
regressor_NonLinearSVR = SVR(kernel='rbf')
regressor_NonLinearSVR.fit(x_train,y_train)


# In[55]:


y_pred = regressor_NonLinearSVR.predict(x_test)


# In[56]:


r2_NonlinearSVR = r2_score(y_test,y_pred)
print(r2_NonlinearSVR)


# In[59]:


labelList = ['Multiple Linear Reg.','K-NearestNeighbors','Decision Tree','Random Forest',
             'Linear SVR','Non-Linear SVR']
mylist = [r2_lr,r2_knn,r2_tree,r2_forest,r2_linearSVR,r2_NonlinearSVR]
for i in range(0,len(mylist)):
    mylist[i]=np.round(mylist[i]*100,decimals=3)
print(mylist)


# In[63]:


plt.figure(figsize=(14,8))
ax = sns.barplot(x=labelList,y=mylist)
plt.yticks(np.arange(0, 101, step=10))
plt.title('r2 score comparison among different regression models',fontweight='bold')
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    ax.annotate('{:.4f}%'.format(height), (x +0.25, y + height + 0.8))
plt.show()


# In[ ]:




