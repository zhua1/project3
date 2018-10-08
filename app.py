# -*- coding: utf-8 -*-
import pandas as pd
from flask import Flask, jsonify, render_template 
from yahoofinancials import YahooFinancials
import numpy as np
from datetime import date
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# identify how many days into the future do we want to predict
future = int(30)
# identify the stocks
tickers = ['AAPL', 'DIS', 'TXN', 'CAT', 'NVDA', 'V', 'CMCSA', 'LMT', 'GOOG', 'TSLA']
names_dict = {'AAPL':'Apple', 'DIS':'Disney', 'TXN':'Texas Instruments', 'CAT':'Caterpilla', 'NVDA':'Nivdia', 'V':'Visa', 'CMCSA':'Comcast', 'LMT':'Lockheed Martin', 'GOOG':'Google', 'TSLA':'Tesla'}
sent_dict = {'AAPL':'Apple', 
'DIS':'Disney', 
'TXN':"Texas Instruments Inc. (stock ticker: TI) is an American technology company that designs and manufactures semiconductors and various integrated circuits, which it sells to electronics designers and manufacturers globally. Headquartered in Dallas, Texas, United States, TI is one of the top ten semiconductor companies worldwide, based on sales volume. Texas Instruments' focus is on developing analog chips and embedded processors, which accounts for more than 80% of their revenue. TI also produces TI digital light processing (DLP) technology and education technology products including calculators, microcontrollers and multi-core processors. To date, TI has more than 43,000 patents worldwide.", 
'CAT':'Caterpilla', 
'NVDA':'Nivdia', 
'V': 'Visa',
'CMCSA':"Comcast (CMSCA) is the world's largest broadcasting and cable corporation. Comcast is the largest cable television and internet provider services. The company began in 1963 in Tupelo, Mississippi and went public in 1972 with an initial price of $7 per share. Comcast's stock price has risen steadily since it was initially offered and peaked for $42 a share in February 2018.", 
'LMT':'Lockheed Martin', 
'GOOG':'Google', 
'TSLA':'Tesla'}
# identify the date interval
date1 = '2016-01-01'
date2 = str(date.today()) 

# adjclose is the same as close
# initialize empty list to append
ti = []
acc = []
pred = []
act = []
for ticker in tickers:
    dat = pd.DataFrame()
    yahoo_financials = YahooFinancials(ticker)
    result = yahoo_financials.get_historical_price_data(date1, date2, 'daily')
    df = pd.DataFrame(data=result[ticker]['prices'])
    df['ticker'] = ticker
    df.drop(columns=['close', 'date'], inplace=True)
    df.rename(columns={'formatted_date':'date'}, inplace=True)
    dat = pd.concat([dat, df], ignore_index=True)
    dat = dat[['ticker', 'date', 'open', 'adjclose', 'low', 'high', 'volume']]
    dat = dat[['adjclose']]
    # predicting the last n days
    act.append(dat['adjclose'].values[-future:])
    dat['prediction'] = dat[['adjclose']].shift(-future)
    
    # prepare X and y
    X = np.array(dat.drop(['prediction'], axis=1))
    X = preprocessing.scale(X)
    # set forecast to the last n rows
    X_forecast = X[-future:] 
    # get rid of the last n nan rows
    X = X[:-future]
    # get y
    y = np.array(dat['prediction'])
    y = y[:-future]
    
    # train test split 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
    
    # linear Regression
    clf = LinearRegression()
    clf.fit(X_train,y_train)
    # Testing
    accuracy = clf.score(X_test, y_test)
    # print(f"Model accuracy for {ticker}: {accuracy}")
    forecast = clf.predict(X_forecast)
    
    ti.append(ticker)
    acc.append(accuracy)
    pred.append(forecast)
    
df = pd.DataFrame(pred).T
df.columns = ti
dates = pd.date_range(end=pd.datetime.today(), periods=future).date.astype('str').tolist()
df['Date'] = dates

actual = pd.DataFrame(act).T
actual.columns = ti
dates = pd.date_range(end=pd.datetime.today(), periods=future).date.astype('str').tolist()
actual['Date'] = dates

acc_dic = {}
for i in range(10):
    acc_dic[tickers[i]] = acc[i]

print('data ready!')
    
# 0. Create app
app = Flask(__name__)

# 1. prediction
@app.route('/')
def Prediction():
    prediction_table = []
    for i in range(df.shape[0]):
        dic = {}
        for j in df.columns:
            dic[j] = df[j][i]
        prediction_table.append(dic)

    actual_table = []
    for i in range(actual.shape[0]):
        dic = {}
        for j in actual.columns:
            dic[j] = actual[j][i]
        actual_table.append(dic)
    return render_template("index.html", data = prediction_table, acc = acc_dic, act = actual_table, names = names_dict, sent = sent_dict)

# 2. Run App
if __name__ == '__main__':
    app.run(port = 5000, debug=True)