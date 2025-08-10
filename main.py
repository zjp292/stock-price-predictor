from models.data_processing import DataProcessing

# get stock data (either csv or ticker)

# call data_processing.py to process the data

# call model based on user input
# show linear reg for free
# show random forest for free
# LSTM for paid

data_processor = DataProcessing("MSFT")
data = data_processor.fetch_data()
print(data.head())
# data_processor.process_data()
