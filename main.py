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

graphing.GraphingUtils.plot_candlestick_chart(data_processor.data[-350:])
