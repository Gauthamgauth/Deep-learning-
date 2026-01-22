#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np 
import pandas as pd 

import os 
for dirname,_, filename in os.walk("/kaggle/input"):
    for filename in filesname:
        print(os.path.join(dirname.filename))


# In[5]:


get_ipython().system('pip install pandas')


# In[6]:


import pandas as pd 


# In[7]:


import sys
get_ipython().system('{sys.executable} -m pip install pandas')


# In[1]:


import pandas as pd 


# In[8]:


df = pd.read_csv("bank datasets.zip")


# In[5]:


df.head()


# In[9]:


df.info()


# In[10]:


df.duplicated().sum()


# In[11]:


df["Exited"].value_counts()


# In[12]:


df["Geography"].value_counts()


# In[13]:


df["Gender"].value_counts()


# In[14]:


df.drop(columns=["RowNumber","CustomerId","Surname"],inplace=True)


# In[15]:


df.head()


# In[16]:


# using OHE
df = pd.get_dummies(df,columns=["Geography","Gender"],drop_first=True)


# In[17]:


df


# In[36]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)


# In[21]:


import sys
get_ipython().system('{sys.executable} -m pip install scikit-learn')


# In[1]:


import sys
get_ipython().system('{sys.executable} -m pip install pandas numpy scikit-learn matplotlib seaborn')


# In[18]:


X= df.drop(columns=["Exited"])
y = df["Exited"]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)


# In[37]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[41]:


X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[42]:


X_train_scaled


# In[43]:


X_test_scaled


# In[20]:


import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import Dense


# In[21]:


model = Sequential()


# In[30]:


model.add(Dense(3,activation="sigmoid",input_dim = 11))
model.add(Dense(1,activation = "sigmoid"))


# In[31]:


model.summary()


# In[49]:


model.compile (loss="binary_crossentropy",optimizer="Adam")


# In[50]:


model.fit(X_train_scaled,y_train,epochs=10)


# In[55]:


model.layers[0].get_weights()


# In[57]:


y_log=model.predict(X_test_scaled)


# In[ ]:


y_pred = np.


# In[62]:


y_pred = np.where(y_log>0.5,1,0)


# In[63]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)

