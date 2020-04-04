#!/usr/bin/env python
# coding: utf-8

# # Daniel Jaso
# # HS 608 - Computer Science for Health Informatics
# # Project 3: Scientific Programming Exploratory Analysis and Machine Learning 

# ## Data Preprocessing 

# In[46]:


import pandas as pd
import numpy as np
import seaborn as sns
from pandas.plotting import scatter_matrix
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer, MinMaxScaler
df = pd.read_csv("/Users/student1/Desktop/SVM_diabetes/data/diabetes_explore.csv")


# In[26]:


df_ml = df.copy()
df_ml[['pres','plas','skin','insu','mass']] = df_ml[['pres','plas','skin','insu','mass']].replace(0,np.NaN)
df_ml.isnull().sum(axis=0) ## looking for null values within dataset 


# ### Drop columns skin, insulin, class (target variable) and age level (since this data is already five on the age level and this might cause overfitting on the model.)

# In[27]:


X = df_ml.drop(labels=['skin','insu','class', 'age_level'], axis=1) 
y = df_ml['class']
X.isnull().sum(axis=0) ## looking for null values within dataset 


# ## Training and Testing
# ### - Context of this data is a classification problem in order to identify whether or not a person will have diabetes or not based upon these 6 columns/features 
# 
# ### - Thus this is a classification problem under the supervised learning approach that will be taken, the ML used will be Support Vector Machine (SVM)
# 
# ### - SVM works for both classification and regression problems. The overview of how this algorithm works is that the data used on the training data set will be used to find the optimal hyperplanes that can be used to classify new data points (test data).

# In[28]:


from sklearn.preprocessing import Imputer, MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score


# In[29]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1,stratify=y)
print(len(X_train),len(y_train))
print(len(X_test),len(y_test))


# ## Imputation
# #### By making use of mean strategy 

# In[30]:


imp_x = Imputer(missing_values='NaN', strategy='mean', axis=0)
X_train = imp_x.fit_transform(X_train)# # fit AND transform training set
X_test = imp_x.transform(X_test) # transform test set on scale fitted to training set


# ## Scaling Data using MinMax Scaler from Sklearn
# ###   X (new) = X(i) - X(min) / X(max) - X(min)
# ### Previously displayed, we observed on the boxplots that our 6 features (pregnancy, plasma (glucose), diastolic blood pressure, mass (BMI), diabetes pedigree function and age) we have outliers that might affect our classifier model 
# 

# In[31]:


print (X_test) #compare before/after scaling
min_max_scaler = MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X_train)# fit AND transform training set
X_test_minmax = min_max_scaler.transform(X_test)# test set transform only, no fit
X_test_minmax


# In[32]:


svc = SVC(kernel='rbf', class_weight='balanced', cache_size=1000, probability=True, gamma=1, C=1) 
print (svc) # calls SVC __str__ to view all the attibutes, including the default params you used 
clf = svc.fit(X_train_minmax, y_train) # trains the classifier on the training set
y_pred_minmax = svc.predict(X_test_minmax) # tests the classifier on the test set
pTot = accuracy_score(y_test, y_pred_minmax)
print ("Prediction accuracy: ",pTot)


# In[33]:


tn, fp, fn, tp = confusion_matrix(y_test, y_pred_minmax, labels=[0,1]).ravel()
print(tn, fp, fn, tp)
cm = confusion_matrix(y_test, y_pred_minmax)
print (cm)
report = classification_report(y_test, y_pred_minmax)
print (report) #for each class prints: precision  recall  f1-score   support


# In[34]:


kernels = ['rbf', 'linear', "poly"]


# In[35]:


for kernel in kernels:
    svc = SVC(kernel=kernel, class_weight='balanced', cache_size=1000, probability=True, gamma='scale') 
    print (svc) #  view all the attibutes of SVC used 
    clf = svc.fit(X_train_minmax, y_train) # trains the classifier on the training set
    y_pred_minmax = svc.predict(X_test_minmax) # tests the classifier on the test set
    pTot = accuracy_score(y_test, y_pred_minmax)
    print ("Prediction accuracy: ",pTot)
    cm = confusion_matrix(y_test, y_pred_minmax)
    print (cm)
    report = classification_report(y_test, y_pred_minmax)
    print (report) #for each class prints: precision  recall  f1-score   support


# ## Paramater Tunning
# ### SVM parameter tunning with rbf (Radial Basis Function)/Gaussian Kernel, Linear, and Polynomial  kernels while using different paramters within our grid search
# ###  - C (regularization term): tells the SVM optimization how much you want to avoid miss classifying each training example. Thus, if the C is higher, the optimization will choose smaller margin hyperplane, so training data miss classification rate will be lower.
# ### - Gamma: The gamma parameter defines how far the influence of a single training example reaches. Thus, the  gamma parameter will consider only points close to the plausible hyperplane and low Gamma will consider points at greater distance.
# ### - Degree(relevant for poly kernel)

# In[36]:


C_range = 10.0 ** np.arange(-4, 4)
gamma_range = [.0001,.001,.01, .1, 1, 10, 100,1000,10000]
print (gamma_range)
param_grid = dict(gamma=gamma_range, C=C_range)
param_grid 


# ### Using RBF Kernel

# In[37]:


grid = GridSearchCV(SVC(kernel='rbf',cache_size=1000, probability=True), 
                    param_grid=param_grid, cv = 5) 
grid.fit(X_train_minmax, y_train)
best_C = grid.best_estimator_.C
best_gamma = grid.best_estimator_.gamma
print ("The best C and gamma for rbf is: %.5f, %.5f " % (best_C, best_gamma))
grid.best_estimator_


# In[38]:


best_predict_minmax = grid.best_estimator_.predict(X_test_minmax)
pTot = accuracy_score(y_test, best_predict_minmax)
print("Prediction accuracy: ",pTot)
cm = confusion_matrix(y_test, best_predict_minmax)
print (cm)
report = classification_report(y_test, best_predict_minmax)
print (report) 


# In[39]:


probas_ = svc.fit(X_train_minmax, y_train).predict_proba(X_test_minmax)
fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])  # use the probs of class (patient w diabetes vs no diabetes)
roc_auc = auc(fpr, tpr)
print ("AUC using predict_proba", roc_auc)

#get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc, lw=3, color ="#0000ff", marker='s',markerfacecolor="red", markersize=2) 
plt.plot([0, 1], [0, 1], 'k--') 

# Set x and y ranges, labels, title and legend
plt.xlim([-0.005, 1.0])  
plt.ylim([0.0, 1.005])   
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


# ### Using Linear Kernel

# In[40]:


grid = GridSearchCV(SVC(kernel='linear',cache_size=1000, probability=True), 
                    param_grid=param_grid, cv = 5) 
grid.fit(X_train_minmax, y_train)
best_C = grid.best_estimator_.C
best_gamma = grid.best_estimator_.gamma
print ("The best C and gamma for linear is: %.5f, %.5f " % (best_C, best_gamma))
grid.best_estimator_


# In[41]:


best_predict_minmax = grid.best_estimator_.predict(X_test_minmax)
pTot = accuracy_score(y_test, best_predict_minmax)
print("Prediction accuracy: ",pTot)
cm = confusion_matrix(y_test, best_predict_minmax)
print (cm)
report = classification_report(y_test, best_predict_minmax)
print (report) 


# In[42]:


probas_ = svc.fit(X_train_minmax, y_train).predict_proba(X_test_minmax)
fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])  # use the probs of class (patient w diabetes vs no diabetes)
roc_auc = auc(fpr, tpr)
#get_ipython().run_line_magic('matplotlib', 'inline')
print ("AUC using predict_proba", roc_auc)
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc, lw=3, color ="#0000ff", marker='s',markerfacecolor="red", markersize=2) 
plt.plot([0, 1], [0, 1], 'k--') 

plt.xlim([-0.005, 1.0])  
plt.ylim([0.0, 1.005])   
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


# In[43]:


C_range = 10.0 ** np.arange(-2, 4)
gamma_range = [.001,.01, .1, 1, 10]
print (gamma_range)
param_grid = dict(gamma=gamma_range, C=C_range)
param_grid 


# ### Using Polynomial Kernel
# #### Following block of code commented out since the gridsearch on the poly kernel took longer than the previous ones
# #### C = .10000 and gamma of 1 were the best estimator obtained

# In[44]:


svc = SVC(kernel='poly', class_weight='balanced', cache_size=1000, probability=True, degree=3, C=0.10000, gamma=1) 
print (svc) # calls SVC __str__ to view all the attibutes, including the default params you used 
clf = svc.fit(X_train_minmax, y_train) # trains the classifier on the training set
y_pred_minmax = svc.predict(X_test_minmax) # tests the classifier on the test set
pTot = accuracy_score(y_test, y_pred_minmax)
print ("Prediction accuracy: ",pTot)
cm = confusion_matrix(y_test, y_pred_minmax)
print (cm)
report = classification_report(y_test, y_pred_minmax)
print (report) #for each class prints: precision  recall  f1-score   support


# In[45]:


probas_ = svc.fit(X_train_minmax, y_train).predict_proba(X_test_minmax)
fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])  # use the probs of class (patient w diabetes vs no diabetes)
roc_auc = auc(fpr, tpr)
#get_ipython().run_line_magic('matplotlib', 'inline')
print ("AUC using predict_proba", roc_auc)
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc, lw=3, color ="#0000ff", marker='s',markerfacecolor="red", markersize=2) 
plt.plot([0, 1], [0, 1], 'k--') 

plt.xlim([-0.005, 1.0])  
plt.ylim([0.0, 1.005])   
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


# # Summary
# 
# ### From the screenshots displayed below, we can see that the model that had the highest accuracy was the RBF kernel. Although, accuracy is not everything in order to evaluate which model is better, rather we need to see at metrics that incorporate precision, recall (sensitivity). F1-score measure since the formula account for both of these metrics. 
# 
# ### F1 = (2 * Recall * Precision)/(Recall + Precision)
# 
# ### This measure tells us a mid point with these metric, also the fact that we can to account for how well our model classify our True Negatives and True Positives that are accounted on the ROC curve. Thus, for both F1 and ROC curve for the Poly kernel seems to be the approriate model that meets both conditions.
# 
# #### Note: classification report spits out macro avg and weighted avg, we want to make reference of the weighted avg because weighted avg takes into accoutn for the imbalance data on our target variable

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# In[ ]:




