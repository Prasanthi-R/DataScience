#!/usr/bin/env python
# coding: utf-8

# In[136]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import math


# In[138]:


df=pd.read_csv('/Users/avanish/Desktop/Coefficient_Assignment/Customer_churn.csv')
df.head()


# In[139]:


df.describe()


# In[140]:


df.info()


# In[141]:


sns.countplot(data=df,x='Attrition_Flag')


# In[142]:


sns.boxplot(data=df,x='Attrition_Flag',y='Customer_Age')


# In[143]:


sns.countplot(data=df,x='Gender',hue='Attrition_Flag')


# In[144]:


sns.catplot(x="Gender", kind="count",hue = 'Attrition_Flag',palette='viridis',data=df)


# In[145]:


fig, ax = plt.subplots(1,1 , figsize=(7, 5))
tmp = df['Education_Level'].value_counts().sort_index()[::-1]

ax.bar(tmp.index, tmp)
ax.set_xticklabels(tmp.index, rotation=45)
ax.set_title('Education_Level', loc='left', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()


# In[146]:


plt.figure(figsize=(13,5))
sns.countplot(data=df,x='Education_Level',hue='Attrition_Flag')


# In[147]:


sns.countplot(data=df,x='Dependent_count')


# In[148]:


sns.countplot(data=df,x='Dependent_count',hue='Attrition_Flag')


# In[149]:


sns.countplot(data=df,x='Marital_Status')


# In[150]:


plt.figure(figsize=(9,5))
sns.countplot(data=df,x='Card_Category',hue='Attrition_Flag')


# In[151]:


plt.figure(figsize=(9,5))
sns.countplot(data=df,x='Attrition_Flag',hue='Dependent_count')


# In[152]:


sns.distplot(df['Months_on_book'])


# In[153]:


sns.boxplot(data=df,x='Attrition_Flag',y='Months_on_book')


# In[154]:


df_categorical=df.loc[:,df.dtypes==np.object]
df_categorical = df_categorical[['Gender', 'Education_Level', 'Marital_Status', 'Income_Category','Card_Category','Attrition_Flag']]
df_categorical.head()


# In[155]:


df_numerical=df.loc[:,df.dtypes!=np.object]
df_numerical['Attrition_Flag']=df.loc[:,'Attrition_Flag']
oh=pd.get_dummies(df_numerical['Attrition_Flag'])
df_numerical=df_numerical.drop(['Attrition_Flag'],axis=1)
df_numerical=df_numerical.drop(['Customer_Id'],axis=1)
df_numerical=df_numerical.join(oh)
df_numerical.head()


# In[156]:


from scipy import stats
num_corr=df_numerical.corr()
plt.figure(figsize=(16, 6))
mask = np.triu(np.ones_like(num_corr, dtype=np.bool))
num_heatmap = sns.heatmap(num_corr, mask=mask, vmin=-1, vmax=1, annot=True, cmap='BrBG')
num_heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12);


# In[157]:


fig, ax=plt.subplots(ncols=2,figsize=(15, 5))

heatmap = sns.heatmap(num_corr[['Existing Customer']].sort_values(by='Existing Customer', ascending=False), ax=ax[0],vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Features Correlating with Existing Customers', fontdict={'fontsize':18}, pad=16);
heatmap = sns.heatmap(num_corr[['Attrited Customer']].sort_values(by='Attrited Customer', ascending=False), ax=ax[1],vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Features Correlating with Attrited Customers', fontdict={'fontsize':18}, pad=16);

fig.tight_layout(pad=5)


# In[158]:


df_model=df
df_model=df_model.drop(['Customer_Id','Credit_Limit','Customer_Age','Avg_Open_To_Buy','Months_on_book','Dependent_count'],axis=1)
df_model.head()


# In[159]:


df_model['Attrition_Flag'] = df_model['Attrition_Flag'].map({'Existing Customer': 1, 'Attrited Customer': 0})
df_oh=pd.get_dummies(df_model)
df_oh['Attrition_Flag'] = df_oh['Attrition_Flag'].map({1: 'Existing Customer', 0: 'Attrited Customer'})
list(df_oh.columns)


# In[160]:


from sklearn.model_selection import train_test_split


# In[161]:


X = df_oh.loc[:, df_oh.columns != 'Attrition_Flag']
y = df_oh['Attrition_Flag']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[162]:


from sklearn.ensemble import RandomForestClassifier

rfc_model = RandomForestClassifier()
rfc_model.fit(X_train, y_train)


# In[163]:


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
svm_model = make_pipeline(StandardScaler(), SVC(gamma='auto'))
svm_model.fit(X_train, y_train)


# In[164]:


from sklearn.ensemble import GradientBoostingClassifier
gb_clf=GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=42)
gb_clf.fit(X_train, y_train)


# In[165]:


y_pred_rfc=rfc_model.predict(X_test)
y_pred_svm=svm_model.predict(X_test)
y_pred_gb=gb_clf.predict(X_test)


# In[166]:


from sklearn.metrics import classification_report, recall_score, precision_score, f1_score
print('Random Forest Classifier')
print(classification_report(y_test, y_pred_rfc))
print('------------------------')
print('Support Vector Machine')
print(classification_report(y_test, y_pred_svm))
print('------------------------')
print('Gradient Boosting')
print(classification_report(y_test, y_pred_gb))

