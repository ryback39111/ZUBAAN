#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 

df = pd.read_csv('Healthy.csv')
df.sample(5)


# In[2]:


df.isna().sum()


# In[3]:


cols = ['Healthy Dog temperature', 'Healthy Dog Heart Rate bpm', 'Healthy Dog Oxygen Level (%)']
df["Healthy"].value_counts() 


# In[4]:


import matplotlib.pyplot as plt

df["Healthy"].value_counts().plot(kind="pie",autopct="%1.1f%%")
plt.show()



# In[5]:


import os
import pandas
import numpy
import pickle
import pefile
import sklearn.ensemble as ek
from sklearn.feature_selection import SelectFromModel
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn import svm
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score



# In[6]:


X = df.drop(["Healthy"],axis=1).values   
Y = df["Healthy"].values


# In[7]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=0.29, stratify = Y)


# In[8]:


from tensorflow.keras.layers import Input, Dense, Activation,Dropout
from tensorflow.keras.models import Model

input_layer = Input(shape=(X.shape[1],))
dense_layer_1 = Dense(15, activation='relu')(input_layer)
dense_layer_2 = Dense(10, activation='relu')(dense_layer_1)
output = Dense(Y.shape[0], activation='sigmoid')(dense_layer_2)

model = Model(inputs=input_layer, outputs=output)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])


# In[9]:


print(model.summary())


# In[10]:


history = model.fit(X_train, y_train, epochs=300)


# In[11]:


score = model.evaluate(X_test, y_test, verbose=1)

print("Test Score:", score[0])
print("Test Accuracy:", score[1])


# In[12]:


from sklearn.metrics import mean_squared_error
from math import sqrt

pred_train = model.predict(X_train).sum(axis=1)
print(np.sqrt(mean_squared_error(y_train,pred_train)))

pred = model.predict(X_test).sum(axis=1)
print(np.sqrt(mean_squared_error(y_test,pred)))


# In[13]:


from sklearn.metrics import classification_report

y_pred=model.predict(X_test)
y_pred=np.argmax(y_pred,axis=1)
accuracy_score(y_pred,y_test)
print(classification_report(y_pred,y_test))
confusion_matrix(y_pred,y_test)


# In[14]:


model.save('final_model.h5')


# In[ ]:




