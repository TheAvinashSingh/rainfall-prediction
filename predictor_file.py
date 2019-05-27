import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from math import sqrt
import sys

data = pd.read_csv('Uttarakhand.csv', sep=',')
data = data.drop(['Unnamed: 0'], axis=1)
mon= {'jan':9, 'feb':10, 'mar':11, 'apr':12, 'may':1, 'jun':2, 'jul':3, 'aug':4, 'sep':5, 'oct':6, 'nov':7,'dec':8}

def predictor(city='Almora', time=2):

    global predictions
    location = city
    data_filtered = data[data.Area == location]
    months_in_year = 12
    X=data_filtered['Rain'].values
    validation_size=12
    start = int(len(X)*0.3)
    train_size = int(len(X) - validation_size)
    prev = X[(train_size - 12 + time):]
    train, test = X[start:train_size], X[train_size:]
    history =[x for x in train]
    predictions = list()
    temp_modl = SARIMAX(train, order=(0,1,1), seasonal_order=(1,1,1,12),enforce_stationarity=False, enforce_invertibility=False)
    model_fit = temp_modl.fit(disp=0)
    yhat = float(model_fit.forecast()[0])
    for i in range(0, 12 - time):
        predictions.append(prev[i])
    predictions.append(yhat)
    history.append(test[0])
    for i in range(1, time):
        mdl = SARIMAX(history, order=(0,1,1), seasonal_order=(1,1,1,12),enforce_stationarity=False, enforce_invertibility=False)
        model_fitt= mdl.fit(disp=0)
        yhat = model_fitt.forecast()[0]
        predictions.append(yhat)
        obs = test[i]
        history.append(obs)
    return yhat
    mse = mean_squared_error(test,predictions)
    rmse = sqrt(mse)
