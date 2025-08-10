from dotenv import load_dotenv
import os
import pandas as pd
from tiingo import TiingoClient


class DataProcessing:
    def __init__(self, ticker):
        load_dotenv()

        API_KEY = os.getenv("TIINGO_API_KEY")
        config = {"api_key": API_KEY, "session": True}

        self.ticker = ticker
        self.client = TiingoClient(config)

    def fetch_data(self, start_date="1990-01-01"):
        if not os.path.isfile(f"{self.ticker}.csv"):
            print(f"Fetching data for {self.ticker}...")
            hist_data = self.client.get_dataframe(self.ticker, start_date)
            hist_data.index = pd.to_datetime(hist_data.index).tz_localize(None)
            hist_data.to_csv(f"{self.ticker}.csv", index=True)
        else:
            print(f"Data for {self.ticker} already exists.")

        return pd.read_csv(f"{self.ticker}.csv", index_col=0)

    def process_data(self):
        # Implement data processing logic here
        pass
