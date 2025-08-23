from models.data_processing import DataProcessing
# from models.neural_models import LSTM
from utils import graphing

# get stock data (either csv or ticker)

# call data_processing.py to process the data

# call model based on user input
# show linear reg for free
# show random forest for free
# LSTM for paid


data_processor = DataProcessing("MSFT")
print(data_processor.data.head())

list_of_columns = [
    "close",
    "SMA_20",
    # "SMA_50",
    "EMA_20",
    # "EMA_50",
    "RSI",
    # "MACD",
    # "MACD_Signal",
    "Bollinger_Upper",
    "Bollinger_Lower",
]
graphing.GraphingUtils.plot_candlestick_chart(data_processor.data[-350:])