import matplotlib.pyplot as plt
import pandas as pd
from tiingo import TiingoClient
import os
from dotenv import load_dotenv
'''
TODO - Create docker of sql to get data 
'''
load_dotenv()
API_KEY = os.getenv('TIINGO_API_KEY')

config = {
    'api_key': API_KEY,
    'session': True
}

client = TiingoClient(config)
ticker = 'AAPL'


if not os.path.isfile('aapl_hist_data.csv'): 
    aapl_hist_data = client.get_dataframe(ticker, '2020-01-01')
    aapl_hist_data.index = pd.to_datetime(aapl_hist_data.index).tz_localize(None)
    aapl_hist_data.to_csv('aapl_hist_data.csv', index=True)

apple_stock_df = pd.read_csv('aapl_hist_data.csv')
apple_stock_df = apple_stock_df.iloc[365:]

seven_day_rolling_avg = apple_stock_df['close'].rolling(window=7).mean()
thirty_day_rolling_avg = apple_stock_df['close'].rolling(window=30).mean()



print(apple_stock_df['close'].rolling(window=7).mean())

plt.plot(apple_stock_df['date'], apple_stock_df['close'])
plt.plot(apple_stock_df['date'], seven_day_rolling_avg)
plt.plot(apple_stock_df['date'], thirty_day_rolling_avg)
plt.xlabel('time')
plt.ylabel('closing price')
plt.title('aaple stock')
plt.show()



# aapl_data = yf.download(ticker, start='2020-01-01', end='2022-01-01')

# aapl_data.to_csv('aapl_data.csv')
# print(aapl_data.head())
# print(aapl_data[('Adj Close', 'AAPL')])
