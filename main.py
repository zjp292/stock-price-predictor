import matplotlib as plt
import pandas as pd
import yfinance as yf

'''
TODO - Create docker of sql to get data 
'''


ticker = 'AAPL'

aapl_data = yf.download(ticker, start='2020-01-01', end='2022-01-01')

aapl_data.to_csv('aapl_data.csv')
print(aapl_data.head())
print(aapl_data[('Adj Close', 'AAPL')])
