#!/usr/bin/env python
# coding: utf-8

# In[70]:


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


# In[71]:


data = pd.read_csv(r"/Users/avanish/Desktop/Coefficient_Assignment/Gender_Prediction.csv")
data.info()


# In[72]:


data.head()


# In[73]:


data.describe()


# In[74]:


data.isnull().sum()


# In[75]:


sns.pairplot(data)


# In[76]:


sns.catplot(x="Ethnicity", kind="count",data=data)


# In[77]:


sns.set(rc={'figure.figsize':(10.7,9.7)})
sns.countplot(x = 'Education_Level', data = data, hue='Gender')
plt.title('Degree Comparison')
locs, labels = plt.xticks()
plt.setp(labels, rotation=20)


# In[78]:


sns.catplot(x="Lunch_Type", kind="count",data=data,palette='plasma')


# In[79]:


plt.figure(figsize=(30,10))
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9,
                      wspace=0.5, hspace=0.2)
plt.subplot(141)
plt.title('Test Preparation Course',fontsize = 20)
data['Preparation_cource_details'].value_counts().plot.pie(autopct="%1.1f%%")


# In[80]:


sns.distplot(data['Maths_Score'])


# In[81]:


sns.distplot(data['Reading_Score'])


# In[82]:


sns.distplot(data['Writing_Score'])


# In[83]:


correlation = data.corr()
sns.heatmap(correlation,xticklabels=correlation.columns,yticklabels=correlation.columns,annot=True)


# In[84]:


data = pd.get_dummies(data, 
                           columns = ["Ethnicity","Lunch_Type","Education_Level","Preparation_cource_details"]
                           )
data.info()


# In[85]:


le = LabelEncoder()
le.fit(data.Gender.drop_duplicates()) 
y = le.transform(data.Gender)


# In[86]:


data.head()


# In[87]:


data.drop(["Gender"],axis = 1 , inplace = True)
data.columns


# In[88]:


x = data.astype(int)


# In[89]:


x


# In[90]:


x_train , x_test , y_train, y_test = train_test_split(x,y,test_size = 0.2 , random_state = 42)


# In[91]:


lr = LogisticRegression()
lr.fit(x_train,y_train)
print("Logistic Regression Score : ",lr.score(x_test,y_test))
lr_score = lr.score(x_test,y_test)


# In[92]:


knn_list = []
for each in range(1,50):
    knn = KNeighborsClassifier(n_neighbors = each)
    knn.fit(x_train,y_train)
    knn_score = knn.score(x_test,y_test)
    knn_list.append(knn_score)
knn_score = np.max(knn_list)
print("K-Nearest Neighbor Score : ",knn_score)


# In[93]:


svm = SVC()
svm.fit(x_train,y_train)
svm_score = svm.score(x_test,y_test)
print("Support Vector Model Score : ", svm_score)


# In[94]:


score_list = []
for each in range (1,50):
    rf = RandomForestClassifier(n_estimators = each,random_state = 7,criterion="gini")
    rf.fit(x_train,y_train)
    score_list.append(rf.score(x_test,y_test))
    
rfc_score = np.max(score_list)
print("RFC Score : ",rfc_score)


# In[95]:


model_scores = {"Support Vector Machine " : svm_score,
          "Logistic Regression" : lr_score,
          "Random Forest Classifier" : rfc_score,
          "K-Nearest Neighbor" : knn_score
          }


# In[96]:


model_scores


# In[ ]:




