import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from math import sqrt
import warnings
import sys
from tqdm import tqdm
sns.set()
warnings.filterwarnings('ignore')

data = pd.read_csv('Uttarakhand.csv', sep=',')

data = data.drop(['Unnamed: 0'], axis=1)

mon= {'jan':11, 'feb':12, 'mar':1, 'apr':2, 'may':3, 'jun':4, 'jul':5, 'aug':6, 'sep':7, 'oct':8, 'nov':9,'dec':10}
months_dict_full = {11 : 'January', 12: 'February', 1: 'March', 2:'April', 3: 'May', 4:'June', 5:'July', 6:'August', 7:'September', 8:'October', 9:'November', 10: 'December'}

def predictor(city='Chamoli', time=4):
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
    print('\nPrediction for the Month: %s is: %.3f\n' % (months_dict_full[time], yhat))
    print('Expected is: %.2f' % obs)
    mse = mean_squared_error(test,predictions)
    rmse = sqrt(mse)
    print('RMSE is %.3f' % rmse)
predictor('Garhwal', mon['apr'])
