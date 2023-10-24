#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing the basic libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#ingesting the datasets 
df=pd.read_csv('/kaggle/input/crop-recommendation-dataset/Crop_recommendation.csv')

print('A sample set of rows for dataframe is:\n')
print(df.sample(6))


# In[2]:


df.isnull().sum()


# In[3]:


df.info()


# In[4]:


df.describe()


# In[5]:


plt.style.use('ggplot')
sns.set_palette("husl")
for i in df.columns[:-1]:
    fig,ax=plt.subplots(1,3,figsize=(18,4))
    sns.histplot(data=df,x=i,kde=True,bins=40,ax=ax[0])
    sns.violinplot(data=df,x=i,ax=ax[1])
    sns.boxplot(data=df,x=i,ax=ax[2])
    plt.suptitle(f'Visualizing {i}',size=20)


# In[6]:


grouped = df.groupby(by='label').mean().reset_index()
grouped


# In[7]:


print(f'********************************')
for i in grouped.columns[1:]:
    print(f'Top 5 Most {i} requiring crops:')
    print(f'********************************')
    for j ,k in grouped.sort_values(by=i,ascending=False)[:5][['label',i]].values:
        print(f'{j} --> {k}')
    print(f'********************************')


# In[8]:


print(f'********************************')
for i in grouped.columns[1:]:
    print(f'Top 5 Least {i} requiring crops:')
    print(f'********************************')
    for j ,k in grouped.sort_values(by=i)[:5][['label',i]].values:
        print(f'{j} --> {k}')
    print(f'********************************')


# In[9]:


figure = plt.figure(figsize=(12, 6))
sns.heatmap(df.corr(),annot=True)


# In[10]:


from sklearn.decomposition import PCA
import plotly.express as px

pca=PCA(n_components=2)
df_pca=pca.fit_transform(df.drop(['label'],axis=1))
df_pca=pd.DataFrame(df_pca)
fig = px.scatter(x=df_pca[0],y=df_pca[1],color=df['label'],title="Decomposed using PCA")
fig.show()


# In[11]:


fig = px.scatter(x=df['N'],y=df['P'],color=df['label'],title="Nitrogen VS Phosphorus")
fig.show()


# In[12]:


fig = px.scatter(x=df['P'],y=df['K'],color=df['label'],title="Phosphorus VS Potassium")
fig.show()


# In[13]:


# #would be required in future to get the names of crops back from encoded form
# names = df['label'].unique()
# from sklearn.preprocessing import LabelEncoder
# encoder=LabelEncoder()
# df['label']=encoder.fit_transform(df['label'])
# df.sample(5)


# In[14]:


X=df.drop(['label'],axis=1)
y=df['label']
#Splitting into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,
                                shuffle = True, random_state = 42,stratify=y)


# In[15]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_train=pd.DataFrame(X_train,columns=X.columns)
X_test=scaler.transform(X_test)
X_train.head()


# In[16]:


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

model=GaussianNB()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
score=accuracy_score(y_test,y_pred)
score


# In[17]:


from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV

model_names = ['svm', 'random_forest', 'logistic_regression', 'gradient_boosting', 'gaussian_nb']

# Initialize classifiers
svm = SVC()
random_forest = RandomForestClassifier()
logistic_regression = LogisticRegression(solver='liblinear', multi_class='auto')
gradient_boosting = GradientBoostingClassifier()
gaussian_nb = GaussianNB()

# List of classifiers
models = [svm, random_forest, logistic_regression, gradient_boosting, gaussian_nb]

# Hyperparameter search spaces for each classifier
model_params = [
    {'C': [1, 10, 100, 0.1], 'kernel': ['rbf', 'linear'], 'gamma': ['scale', 'auto']},
    {'n_estimators': [1, 5, 10]},
    {'C': [1, 5, 10], 'penalty': ['l1', 'l2', 'elasticnet']},
    {'n_estimators': [50, 100, 200], 'learning_rate': [0.1, 0.5, 1.0]},
    {'var_smoothing': [0.4, 0.8, 1e-9]}
]


# In[18]:


# Perform grid search for each classifier
for name, model, params in zip(model_names, models, model_params):
    grid_search = GridSearchCV(model, params, cv=5)
    grid_search.fit(X_train, y_train)  # Replace X_train and y_train with your data
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    print(f"Best parameters for {name}: {best_params}")
    print(f"Best cross-validation score for {name}: {best_score}")


# In[19]:


from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

best_model=GaussianNB(var_smoothing=1e-9)
best_model.fit(X_train,y_train)
y_pred=best_model.predict(X_test)
score=accuracy_score(y_pred,y_test)
print('The accuracy score of the Model is {}\n\n'.format(score))
report=classification_report(y_pred,y_test)
print('The classification Report is')
print(report)


# In[20]:


import numpy as np
from sklearn.preprocessing import label_binarize

n_classes=len(np.unique(y_test))
y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
y_pred_bin = label_binarize(y_pred, classes=np.unique(y_test))

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
plt.xticks(np.arange(n_classes), np.unique(y_test))
plt.yticks(np.arange(n_classes), np.unique(y_test))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Generate ROC curve and calculate AUC for multiclass classification
if n_classes > 2:
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_bin[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=2, label='ROC curve class %d (area = %0.2f)' % (i, roc_auc[i]))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    
    # Position the legend outside the plot and adjust its position
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.show()

