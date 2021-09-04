
#pip install whatthelang

import numpy as np 
import pandas as pd 
from textblob import TextBlob
import matplotlib.pyplot as plt

import os
file = 'C:/Users/Kshama/Downloads/tweets/tweets.csv'
df = pd.read_csv(file, sep=';',nrows=1300)
print(df.head())

from google.colab import files
uploaded = files.upload()

import io
df2 = pd.read_csv(io.BytesIO(uploaded['Bitcoin_tweets.csv']))
#pip install whatthelang

import numpy as np 
import pandas as pd 
from textblob import TextBlob
import matplotlib.pyplot as plt

import os
file = 'C:/Users/Kshama/Downloads/tweets/tweets.csv'
df = pd.read_csv(file, sep=';',nrows=1300)
print(df.head())

from google.colab import files
uploaded = files.upload()

import io
df2 = pd.read_csv(io.BytesIO(uploaded['Bitcoin_tweets.csv']))

#pip install textBlob

print(df2.head())
twe=df2[["date","text"]]

twe.to_csv('twe.csv', index=False)

from google.colab import files
files.download('twe.csv')

import csv
from whatthelang import WhatTheLang
wtl = WhatTheLang()
L=[]
for row in twe['text']:
    L.append(wtl.predict_lang(str(row)))
        
twe['lang'] = L
twe.head()

twe1 = twe[twe["lang"] == 'en']
twe1.head()

import nltk
import re
from nltk.corpus import stopwords

def text_cleaning(text):
    forbidden_words = set(stopwords.words('english'))
    text = ' '.join(text.split('.'))
    text = re.sub('\/',' ',text)
    text = text.strip('\'"')
    text = re.sub(r'@([^\s]+)',r'\1',text)
    text = re.sub(r'\\',' ',text)
    text = text.lower()
    text = re.sub('[\s]+', ' ', text)
    text = re.sub(r'#([^\s]+)', r'\1', text)
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))',' ',text)
    text = re.sub(r'((http)\S+)','',text)
    text = re.sub(r'\s+', ' ', re.sub('[^A-Za-z]', ' ', text.strip().lower())).strip()
    text = re.sub(r'\W+', ' ', text.strip().lower()).strip()
    text = [word for word in text.split() if word not in forbidden_words]
    return ' '.join(text)

twe['text'] = twe['text'].apply(lambda text: text_cleaning(text))
twe.sample(3)

twe1 = twe.dropna()
print(twe1)

nltk.download('stopwords')

twe.sample(10)

twe1.to_csv('twe1.csv', index=False)
from google.colab import files
files.download('twe1.csv')

from textblob import TextBlob

def sentiment(txt):
    return TextBlob(txt).sentiment.polarity

twe1['sentiment'] = twe1['text'].apply(lambda txt: sentiment(txt))      # new column of sentiment

twe1.sample(10)

twe1.to_csv('twe1.csv', index=False)
from google.colab import files
files.download('twe1.csv')

twe1.head()

import io
df2 = pd.read_csv(io.BytesIO(uploaded['sentiTwet.csv']))

df2['date'] = pd.to_datetime(df2['date'])

df2.info()

twe1.resample('H', on='date').sentiment.mean()



twe1.resample('60min', base=00, label='right')['date'].first()

#VADER

from time import time
import pytz
import pandas as pd
import numpy as np
import re
import sys
import csv

import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

twe1 = pd.read_csv('D:/590 ML/Project/Data/tweets.csv/sentiTwet.csv', skiprows=0)
twe1.columns #Index(['date', 'text', 'lang', 'sentiment'], dtype='object')

import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()
compound = []
for text in twe1['text']:
    vs = sid.polarity_scores(text)
    compound.append(vs["compound"])
twe1["compound"] = compound

twe1.columns #Index(['date', 'text', 'lang', 'sentiment', 'compound'], dtype='object')
twe1.sample(10)

twe1.to_csv(r'D:/590 ML/Project/Data/tweets.csv/sentiTwet_blob_vader_1.csv', index=False, header=True)
######################################################
twe2 = pd.read_csv('D:/590 ML/Project/Data/tweets.csv/sentiTwet_blob_vader_1.csv', skiprows=0)

#twe2.drop_duplicates(inplace=True)
twe2.drop_duplicates(subset=['date'], inplace=True)

date_new =[]
for date in twe2['date']:
    date = pd.to_datetime(date, errors='coerce') # , format='%m/%d/%y %H:%M')#.tz_localize(tz=None)
    date_new.append(date)
twe2['date'] = date_new

twe2['date'].fillna(method='ffill', inplace=True)
twe2.head()
twe2.tail()

twe2.info()
twe2 = twe2.groupby([pd.Grouper(key='date', freq='H')]).sum().reset_index()

twe2.to_csv(r'D:/590 ML/Project/Data/tweets.csv/sentiTwet_blob_vader_hourly_timestamp_sum.csv', index=False, header=True)

#preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
import plotly.express as px
from itertools import product
import warnings
import statsmodels.api as sm
plt.style.use('seaborn-darkgrid')

bitstamp = pd.read_csv("D:/590 ML/Project/Data/bitstampUSD/gemini_BTCUSD_1hr.csv")
bitstamp.head()
bitstamp.info()

# Converting the Timestamp column from string to datetime
#bitstamp['Date'] = [datetime.fromtimestamp(x) for x in bitstamp['Date']]
bitstamp.drop_duplicates(subset=['Date'], inplace=True)
date_new =[]
for date in bitstamp['Date']:
    date = pd.to_datetime(date, errors='coerce') # , format='%m/%d/%y %H:%M')#.tz_localize(tz=None)
    date_new.append(date)
bitstamp['Date'] = date_new

bitstamp = bitstamp.sort_values(by = 'Date')
bitstamp = bitstamp[(bitstamp.Date >= '2/5/2021 10:00') & (bitstamp.Date <= '7/5/2021 23:00')]

bitstamp.columns #Index(['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume'], dtype='object')
bitstamp = bitstamp[['Date','Close']]

bitstamp.to_csv(r'D:/590 ML/Project/Data/bitstampUSD/hourly_rate_short.csv', index=False, header=True)

bitstamp.head()
bitstamp.tail()


#MERGE
import pandas as pd

from google.colab import files
uploaded = files.upload()

import io
df2 = pd.read_csv(io.BytesIO(uploaded['sentiTwet_blob_vader_hourly_timestamp_sum.csv']))

import io
df3 = pd.read_csv(io.BytesIO(uploaded['hourly_rate_short (1).csv']))

merged = df2.merge(df3, on='date')

new_merged=merged[["sentiment","compound","Close"]]

new_merged.to_csv('premodel.csv', index=False)
from google.colab import files
files.download('premodel.csv')


#EXPLORATORY DATA ANALYSIS

import json
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tnrange, tqdm_notebook, tqdm
import glob
from datetime import datetime, timedelta

sr = pd.read_csv('D:/590 ML/Project/Data/tweets.csv/score_rate.csv')

fig, ax1 = plt.subplots(figsize=(20,10))
ax1.set_title("Crypto currency evolution compared to twitter sentiment", fontsize=18)
ax1.tick_params(labelsize=14)
ax2 = ax1.twinx()
ax3 = ax1.twinx()
ax1.plot_date(sr.date, sr.sentiment, 'g-')
ax2.plot_date(sr.date, sr.compound, 'b-')
ax3.plot_date(sr.date, sr.Close, 'r-')
ax1.set_ylabel("TextBlob Sentiment", color='g', fontsize=16)
ax2.set_ylabel(f"Vader Compound", color='b', fontsize=16)
ax3.set_ylabel("Hourly Closing Price", color='r', fontsize=16)
plt.show()

#Data Normalization
sentiments = []
d_sentiment = max(sr.sentiment.max(), abs(sr.sentiment.min()))
for sentiment in sr['sentiment']:
    sentiment = sentiment/d_sentiment
    sentiments.append(sentiment)
sr['sentiment']=sentiments

compounds = []
d_compound = max(sr.compound.max(), abs(sr.compound.min()))
for compound in sr['compound']:
    compound = compound/d_compound
    compounds.append(compound)
sr['compound']=compounds

Closes = []
d_Close = max(sr.Close.max(), abs(sr.Close.min()))
for Close in sr['Close']:
    Close = Close/d_Close
    Closes.append(Close)
sr['Close']=Closes

fig, ax1 = plt.subplots(figsize=(20,10))
ax1.set_title("Normalized Crypto currency evolution compared to normalized twitter sentiment", fontsize=18)
ax1.tick_params(labelsize=14)
ax2 = ax1.twinx()
ax3 = ax1.twinx()
ax1.plot_date(sr.date, sr.sentiment, 'g-')
ax2.plot_date(sr.date, sr.compound, 'b-')
ax3.plot_date(sr.date, sr.Close, 'r-')
ax1.set_ylabel("TextBlob Sentiment", color='g', fontsize=16)
ax2.set_ylabel(f"Vader Compound", color='b', fontsize=16)
ax3.set_ylabel("Hourly Closing Price", color='r', fontsize=16)
plt.show()


# Derivative
sentiments = []
for sentiment in sr['sentiment']:
    sentiment = pd.Series(np.gradient(sr.sentiment), sr.date, name='slope')
    sentiments.append(sentiment)
sr['sentiment']=sentiments

compounds = []
for compound in sr['compound']:
    compound = pd.Series(np.gradient(sr.compound), sr.date, name='slope')
    compounds.append(compound)
sr['compound']=compounds

Closes = []
for Close in sr['Close']:
    Close = pd.Series(np.gradient(sr.Close), sr.date, name='slope')
    Closes.append(Close)
sr['Close']=Closes

fig, ax1 = plt.subplots(figsize=(20,10))
ax1.set_title("Derivative of crypto currency and sentiment's score", fontsize=18)
ax1.tick_params(labelsize=14)
ax2 = ax1.twinx()
ax3 = ax1.twinx()
ax1.plot_date(sr.date, sentiment, 'g-')
ax2.plot_date(sr.date, compound, 'b-')
ax3.plot_date(sr.date, Close, 'r-')
ax1.set_ylabel("TextBlob Sentiment's derivative", color='g', fontsize=16)
ax2.set_ylabel("Vader Compound's derivative'", color='b', fontsize=16)
ax3.set_ylabel("Bitcoin close Price's derivative'", color='b', fontsize=16)
plt.show()

#MODELING

# In[121]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error


# In[264]:


df = pd.read_csv('D:/590 ML/Project/Data/premodel.csv')
df.head()


# In[265]:


values = df.values
cols = df.columns.tolist()
cols = cols[-1:] + cols[:-1]
df = df[cols]
df = df[['sentiment', 'compound', 'Close']]
df.head()


# In[266]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(df.values)


# In[267]:


n_hours = 1 #adding 1 hours lags creating number of observations 
n_features = 2 #Features in the dataset.
n_obs = n_hours*n_features


# In[268]:


from math import sqrt
from numpy import concatenate
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot

from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# In[269]:


reframed = series_to_supervised(scaled, n_hours, 1)
reframed.head()


# In[270]:


reframed.drop(reframed.columns[-4], axis=1)
reframed.head()


# In[271]:


values = reframed.values
n_train_hours = 2500
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
train.shape


# In[272]:


# split into input and outputs
train_X, train_y = train[:, :n_obs], train[:, -n_features]
test_X, test_y = test[:, :n_obs], test[:, -n_features]


# In[273]:


# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


# In[274]:


# design network
model = Sequential()
model.add(LSTM(3, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=10, batch_size=2, validation_data=(test_X, test_y), verbose=2, shuffle=True)


# In[275]:


plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()


# In[276]:


# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], n_hours* n_features,))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, -4:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, -4:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]







mape = (mean_absolute_percentage_error(test_y, yhat))
print('Test MAPE: %.3f' % mape)







