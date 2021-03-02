#!/usr/bin/env python
# coding: utf-8

# # 1 . Inspecting transfusion.data file

# In[1]:


# Print out the first 5 lines from the transfusion.data file
get_ipython().system('head -n 5 datasets/transfusion.data')


# # 2. Loading Blood Donations dataset

# In[3]:


#Import Pandas
import pandas as pd


#Read the Dataset
transfusion = pd.read_csv('transfusion.data')

#Print out the first rows of the Dataset
transfusion.head(5)


# # 3. Inspecting transfusion dataframe

# In[42]:


# Print a concise summary of transfusion DataFrame
transfusion.info()


# # 4. Creating Target Column
# 

# In[43]:


# Rename target column as 'target' for brevity 
transfusion.rename(
    columns={'whether he/she donated blood in March 2007':'target'},
    inplace=True
)


# Print out the First 2 rows
transfusion.head(2)


# # 5. Checking Target Incidence
# 

# In[44]:


# Print target incidence proportions, rounding output to 3 decimal places
transfusion.target.value_counts(normalize=True)


# # 6. Splitting transfusion ino train and test datasets

# In[45]:


## Import train_test_split method
from sklearn.model_selection import train_test_split

# Split transfusion DataFrame into
# X_train, X_test, y_train and y_test datasets,
# stratifying on the `target` column
X_train, X_test, y_train, y_test = train_test_split(
    transfusion.drop(columns='target'),
    transfusion.target,
    test_size=0.25,
    random_state=42,
    stratify=transfusion.target
)


# Print out the first 2 rows of X_train
X_train.head(2)


# # 7. Selecting Model using TPOT

# In[46]:


# Import TPOTClassifier and roc_auc_score
from tpot import TPOTClassifier
from sklearn.metrics import roc_auc_score

# Instantiate TPOTClassifier
tpot = TPOTClassifier(
    generations=5,
    population_size=20,
    verbosity=2,
    scoring='roc_auc',
    random_state=42,
    disable_update_check=True,
    config_dict='TPOT light'
)
tpot.fit(X_train, y_train)

# AUC score for tpot model
tpot_auc_score = roc_auc_score(y_test, tpot.predict_proba(X_test)[:, 1])
print(f'\nAUC score: {tpot_auc_score:.4f}')

# Print best pipeline steps
print('\nBest pipeline steps:', end='\n')
for idx, (name, transform) in enumerate(tpot.fitted_pipeline_.steps, start=1):
    # Print idx and transform
    print(f'{idx}. {transform}')


# # 8. Checking the Variance

# In[47]:


# X_train's variance, rounding the output to 3 decimal places
X_train.var().round(3)


# # 9. Log Normalization

# In[48]:


# Import numpy
import numpy as np



#Copy X_train and X_test into X_train_normed and X_test_normed
X_train_normed, X_test_normed = X_train.copy(), X_test.copy()

# Specify which column to normalize
col_to_normalize = 'Monetary (c.c. blood)'

# Log normalization
for df_ in [X_train_normed, X_test_normed]:
    # Add log normalized column
    df_['monetary_log'] = np.log(df_[col_to_normalize])
    # Drop the original column
    df_.drop(columns=col_to_normalize, inplace=True)

# Check the variance for X_train_normed
X_train_normed.var().round(3)


# # 10 . Training Linear Regression Model

# In[49]:


# Importing module
from sklearn import linear_model


#Instantiate LogisticRegression
logreg = linear_model.LogisticRegression(
    solver='liblinear',
    random_state=42
)

#Train the Model
logreg.fit(X_train_normed,y_train)


# AUC score for tpot model
logreg_auc_score = roc_auc_score(y_test, logreg.predict_proba(X_test_normed)[:, 1])
print(f'\nAUC score: {logreg_auc_score:.4f}')


# # 11. Conclusion

# In[50]:





# Import itemgetter
from operator import itemgetter


# Sort models based on their AUC score from highest to lowest
sorted(
    [('tpot', tpot_auc_score), ('logreg', logreg_auc_score)],
    key=itemgetter(1),
    
)


# In[51]:


import pickle


# In[54]:


pickle.dump(logreg,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))


# In[ ]:




