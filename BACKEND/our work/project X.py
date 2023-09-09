#!/usr/bin/env python
# coding: utf-8

# In[35]:


# import the libraries

import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import keras as kr
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('bmh')


# In[2]:


# get the data online from yahoo api
df = web.DataReader('AAPL' ,data_source = 'yahoo' , start= '2013-01-01' ,end='2021-01-01')
df1 = web.DataReader('TSLA' ,data_source = 'yahoo' , start= '2013-01-01' ,end='2021-01-01')


# In[3]:


df


# In[4]:


df.describe()


# In[5]:


df.shape


# In[36]:


plt.figure(figsize = (16,9))
plt.title('APPLE close price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Price', fontsize=18)


# In[37]:


df1.shape


# In[38]:


plt.figure(figsize = (16,9))
plt.title('TESLA close price History')
plt.plot(df1['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Price', fontsize=18)


# In[39]:


# create a new datafram with close price only
data = df.filter(['Close'])
#convert the data dataframe to numpy array
dataset= data.values
#get the number of rows to train the model 
training_data_len =math.ceil(len(dataset) * .8)

training_data_len


# In[40]:


#scale the data 
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)


# In[41]:


scaled_data


# In[42]:


train_data = scaled_data[0:training_data_len , :]


# In[43]:


# split the data into x_train data sets
x_train = []
y_train = []


# In[44]:


for i in range (100, len(train_data) ):
    x_train.append(train_data[i-100:i, 0])
    y_train.append(train_data[i, 0])
    if i <= 100 :
        print(x_train)
        print(y_train)
                  


# In[45]:


# convert the x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)


# In[46]:


x_train.shape


# In[47]:


x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_train.shape


# In[48]:


#build the LSTM model
model = kr.Sequential()


# In[49]:


model.add(kr.layers.LSTM(50, return_sequences=True, input_shape= (x_train.shape[1], 1) ) )
model.add(kr.layers.LSTM(50, return_sequences=False))
model.add(kr.layers.Dense(25))
model.add(kr.layers.Dense(1))


# In[50]:


# compile the model
model.compile(optimizer= 'adam', loss= 'mean_squared_error')


# In[51]:


# train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)


# In[52]:


# create the testing data set 
#create a new array containing scaled value from index 1543 to 2003
test_data = scaled_data[training_data_len - 100: , :]
# create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]
for i in range (100, len(test_data)):
    x_test.append(test_data[i-100:i, 0])


# In[53]:


# convert the data to a numpy array 
x_test = np.array(x_test)


# In[54]:


x_test.shape


# In[55]:


# Reshape the data 
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


# In[56]:


# forcasting  , analysis from historical data 
# flactionation 
# change inflation rate, technology 


# In[57]:


x_test.shape


# In[58]:


# get the models predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)


# In[59]:


df.describe()


# In[60]:


# get the root mean squared error (RMSE
rmse= np.sqrt(np.mean(predictions -y_test)**2)
rmse


# In[ ]:





# In[61]:


#plot the data 
train = data[:training_data_len]
valid = data[training_data_len:]
valid['predictions']= predictions
#visualize the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('close price USD ', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close','predictions']])
plt.legend(['Train','Val','predictions'], loc='lower right')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




