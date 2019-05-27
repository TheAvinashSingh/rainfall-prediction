from flask import Flask, render_template, request, url_for
from predictor_file import predictor
app = Flask(__name__)

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from math import sqrt
import sys

data = pd.read_csv('Uttarakhand.csv', sep=',')
data = data.drop(['Unnamed: 0'], axis=1)
mon= {'jan':9, 'feb':10, 'mar':11, 'apr':12, 'may':1, 'jun':2, 'jul':3, 'aug':4, 'sep':5, 'oct':6, 'nov':7,'dec':8}

@app.route('/home', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        city = request.form['SelectCity']
        month = request.form['SelectMonth']
        predicted = predictor(city, mon[month])
        #Main Code Here

        #Code Ends Here
        return render_template('index.html', city=city, month=month, predicted=predicted)
    return render_template('index.html')

@app.route('/harvesting')
def harvesting():
    return render_template('harvesting.html')

if __name__ == '__main__':
    app.run(debug= True)
