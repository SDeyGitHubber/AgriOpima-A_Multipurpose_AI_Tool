#!/usr/bin/env python
# coding: utf-8

# # Pretext
# In a world ever more interconnected, the marriage of machine learning and agriculture gains profound significance. With our global population swelling, the study of crop yield becomes paramount. The intricate dance of weather patterns, chemical treatments, and historical insights determines agricultural success. Unlocking this puzzle holds the key to food security and resilience against climate flux.     
# 
# This project squarely addresses this challenge by deploying machine learning to predict the top 10 most-consumed crops worldwide. These essential staples like corn, wheat, and rice, form the bedrock of human sustenance. By harnessing the power of regression techniques, we develop a path to foresee yields, empowering farmers, as they can optimise resources.
# 
# In this notebook, I have aimed to provide deep, insightful analysis, and visualisation, unveiling meaningful information that will be of significant value, and have also finally built a regression model, that can predict yields on unseen test data, with a whopping R2 score of about 0.972!

# <div style="background-color: #fce5cd; padding: 20px; border-radius: 10px;">
#     <p style="font-size: 18px; text-align: center;"><em>Your support fuels inspiration!</em></p>
#     <p style="font-size: 16px; text-align: center;">🌟 If this Notebook has intrigued you, I humbly invite you to join in celebrating the journey. An upvote is a resounding cheer, a way to say "Bravo!"</p>
#     <p style="font-size: 16px; text-align: center;">Let's build a connection – a bridge for ideas to meet up. Your engagement encourages, and I extend a warm welcome to connect.</p>
#     <p style="font-size: 16px; text-align: center;">Together, let's revel in the journey of exploration. 🚀🤝</p>
# </div>
# 
# 

# # Loading and reading the datasets
# 
# After importing required libraries, crops yield of ten most consumed crops around the world was downloaded from FAO webiste.The collected data include country, item, year starting from 1961 to 2016 and yield value

# In[1]:


#importing the basic libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#ingesting the datasets 
pest_df=pd.read_csv('/kaggle/input/crop-yield-prediction-dataset/pesticides.csv')
rain_df=pd.read_csv('/kaggle/input/crop-yield-prediction-dataset/rainfall.csv')
temp_df=pd.read_csv('/kaggle/input/crop-yield-prediction-dataset/temp.csv')
yield_df=pd.read_csv('/kaggle/input/crop-yield-prediction-dataset/yield.csv')
data_frames=[pest_df,rain_df,temp_df,yield_df]

#reading a sample set of rows for all the ingested datasets 
for df in data_frames:
    print('A sample set of rows for {} is:\n'.format(df))
    print(df.sample(6))


# # Performing Feature Engineering and Null value imputation

# In[2]:


# Dropping unnecessary columns, as they wont be of anu help to find patterns in our
# data
pest_df=pest_df.drop(['Unit','Domain','Element','Item'],axis=1)
yield_df=yield_df.drop(['Domain Code','Domain','Area Code','Element Code','Item Code','Year Code','Unit'],axis=1)
pest_df.head()


# In[3]:


yield_df.head()


# In[4]:


yield_df.columns


# In[5]:


rain_df.rename(columns = {' Area':'Area'},inplace = True)
for df in data_frames:
    print(df.columns)


# In[6]:


temp_df.rename(columns = {'year':'Year','country':'Area'},inplace = True)
temp_df.columns


# In[7]:

# Merging our datasets into 1 single dataframe
yield_df_df=pd.read_csv('/kaggle/input/crop-yield-prediction-dataset/yield_df.csv')
pr=pd.merge(pest_df,rain_df,on=['Year','Area'])
prt=pd.merge(pr,temp_df,on=['Year','Area'])
prty=pd.merge(yield_df,prt,on=['Year','Area'])
print(prty.columns)
prty.sample(10)


# In[8]:


prty.rename(columns={'Value_y':'pesticides_tonnes','Value_x':'hg/ha_yield'},inplace=True)
prty=prty.drop('Element',axis=1)
yield_df_df=yield_df_df.drop('Unnamed: 0',axis=1)
print(yield_df_df.columns)
print(prty.columns)


# ### Descriptive Data Analysis

# In[9]:


yield_df_df.shape,prty.shape


# In[10]:


prty.describe()


# In[11]:


prty.info()


# **Here, the data for our rainfall is of Object data type but is desired in integer format**

# In[12]:


prty.isnull().sum().sum()/prty.shape[0]


# **Since an extremely small fragment of our dataset contains null values( around 0.02 percent), we can consider dropping those rows**

# In[13]:


prty=prty.dropna()
prty.isnull().sum()


# In[14]:


#converting the datatype of rainfall data to a desirable type
prty['average_rain_fall_mm_per_year'] = prty['average_rain_fall_mm_per_year'].replace('..',np.nan)
prty['average_rain_fall_mm_per_year'] = prty['average_rain_fall_mm_per_year'].astype('float')
prty=prty.dropna()
#filtering out the numerical and categorical columns, that will be useful for our 
#feature scaling and EDA
num_cols = [i for i in prty.columns if (prty[i].dtype == 'float64' or prty[i].dtype == 'int64')]
cat_cols = [i for i in prty.columns if (i not in num_cols) and i != 'hg/ha_yield']
print(num_cols)
print(cat_cols)


# # Exploratory Data Analysis
# Here onwards, we have some detailed visual analysis of several patterns in our data. We have used multiple data analytics and visualization techniques and plotted several insightful results! Stay glued for the upcoming, intriguing insights !

# In[15]:


plt.figure(figsize=(12, 16))
for i, col in enumerate(num_cols, 1):
    plt.subplot(3, 2, i)
    sns.boxplot(data=prty[col])
    plt.title(f'Box Plot of {col}')
    plt.ylabel('Value')

plt.tight_layout()
plt.show()


# In[16]:


plt.figure(figsize=(10,6))
sns.countplot(x='Item',data=prty)
plt.title('Countplot of Item vs its count')
plt.xlabel('Crops',fontsize=18,loc='center')
plt.ylabel('Count',fontsize=18,loc='center')
plt.xticks(rotation=45)
plt.show()


# In[17]:


fig, axes = plt.subplots(3, 1, figsize=(18, 22))
sns.lineplot(x = "pesticides_tonnes", y = "hg/ha_yield", hue = "Item", data = prty, ax=axes[0], legend = True)
axes[0].tick_params(axis='x', rotation=45)
axes[0].set_ylabel('Average Yield')
axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

sns.lineplot(x = "average_rain_fall_mm_per_year", y = "hg/ha_yield", hue = "Item", data = prty, ax=axes[1], legend = True)
axes[1].tick_params(axis='x', rotation=45)
axes[1].set_ylabel('Average Yield')
axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

sns.lineplot(x = "avg_temp", y = "hg/ha_yield", hue = "Item", data = prty, ax=axes[2], legend = True)
axes[2].tick_params(axis='x', rotation=45)
axes[2].set_ylabel('Average Yield')
axes[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()


# In[18]:


df=prty.copy()
df['yield_rainfall_ratio'] = df['hg/ha_yield'] / df['average_rain_fall_mm_per_year']

top_10_crops = df.groupby('Item')['yield_rainfall_ratio'].mean().sort_values(ascending=False).head(10).index

# Filter the data to only include the top 10 crops
top_10_data = df[df['Item'].isin(top_10_crops)]

sns.barplot(data=top_10_data, x='Item', y='yield_rainfall_ratio', order=top_10_crops)
plt.xlabel('Crops',fontsize=18)
plt.ylabel('Yield/Average Rainfall',fontsize=18)
plt.xticks(rotation=45)
plt.show()


# In[19]:


df=prty.copy()
df['yield_rainfall_ratio'] = df['hg/ha_yield'] / df['average_rain_fall_mm_per_year']

top_10_countries = df.groupby('Area')['yield_rainfall_ratio'].mean().sort_values(ascending=False).head(10).index
top_10_data = df[df['Area'].isin(top_10_countries)]

sns.boxplot(data=top_10_data, x='Area', y='yield_rainfall_ratio', order=top_10_countries)
plt.xlabel('Countries',fontsize=18)
plt.ylabel('Yield/Average Rainfall',fontsize=18)
plt.xticks(rotation=45)
plt.show()


# In[20]:


plt.figure(figsize=(7,7))
sns.heatmap(prty.corr(), annot=True,linewidth=.5,cmap='crest')
plt.show()


# In[21]:


fig, axes = plt.subplots(3, 1, figsize=(18, 22))

sns.scatterplot(x = "pesticides_tonnes", y = "hg/ha_yield", hue = "Item", data = prty, ax=axes[0], legend = True)
axes[0].tick_params(axis='x', rotation=45)
axes[0].set_ylabel('Average Yield')
axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

sns.scatterplot(x = "average_rain_fall_mm_per_year", y = "hg/ha_yield", hue = "Item", data = prty, ax=axes[1], legend = True)
axes[1].tick_params(axis='x', rotation=45)
axes[1].set_ylabel('Average Yield')
axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

sns.scatterplot(x = "avg_temp", y = "hg/ha_yield", hue = "Item", data = prty, ax=axes[2], legend = True)
axes[2].tick_params(axis='x', rotation=45)
axes[2].set_ylabel('Average Yield')
axes[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()


# In[22]:


prty.columns


# # Data Preprocessing

# In[23]:


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# First, we have performed one hot encoding on the categorical features, and then
# performed standard scaling on all the features
prty=prty.drop('Year',axis=True)
X,y=prty.drop('hg/ha_yield',axis=1),prty['hg/ha_yield']
X = pd.get_dummies(X,columns = cat_cols, drop_first = True)
scaler = StandardScaler()
X=scaler.fit_transform(X)


# In[24]:


X


# In[25]:


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.20,shuffle=True)
print(X_train.shape,X_test.shape)


# # Building our Initial model

# In[26]:


from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression,Lasso
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

poly=PolynomialFeatures(degree=2,order='C',include_bias=True)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
lin_reg=LinearRegression()
lin_reg.fit(X_train_poly,y_train)
y_pred=lin_reg.predict(X_test_poly)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)


# # Trying multiple models and Hyperparameter Tuning 

# In[27]:


model_names=['svm_regressor','random_forest_regressor','lasso_regressor','XGBoost_Regressor']

svr=SVR(kernel='rbf', gamma='auto')
random_forest=RandomForestRegressor()
lasso_regressor = Lasso(alpha=1.0, random_state=42,max_iter=3000)
xgb_regressor = xgb.XGBRegressor(n_estimators=100, random_state=42)

models = [svr, random_forest, lasso_regressor, xgb_regressor]

model_params = [
    {},  # SVR doesn't require hyperparameters here
    {'n_estimators': [10, 50, 100]},  # RandomForestRegressor parameters
    {'alpha': [0.1, 1.0, 10.0]},  # Lasso parameters
    {'n_estimators': [50, 100, 200], 'learning_rate': [0.1, 0.5, 1.0]}  # XGBRegressor parameters
]


# In[28]:


# Here, we have performed hyperparameter tuning on multiple regression models
# to finally find out the best model

scores = []
best_estimators = {}

for name, model, params in zip(model_names, models, model_params):
#     pipe = make_pipeline(StandardScaler(), model)
    clf = GridSearchCV(model, params, cv=5, return_train_score=False)
    clf.fit(X_train, y_train)
    scores.append({
        'model': name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    best_estimators[model] = clf.best_estimator_

res = pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])
res


# In[29]:


best_model=xgb.XGBRegressor(learning_rate=0.5,n_estimators=200)
best_model.fit(X_train,y_train)
y_pred=best_model.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("R-squared:", r2)


# In[30]:


#plotting the results of our model, against the original results
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.7, color='blue', label='Predicted')
sns.scatterplot(x=y_test, y=y_test, alpha=0.7, color='red', label='Actual')
plt.xlabel("Actual Values (y_test)")
plt.ylabel("Predicted Values (y_pred)")
plt.title("Actual vs. Predicted Values")
plt.grid(True)
plt.legend()
plt.show()

