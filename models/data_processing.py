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
        self.data = None

        self.fetch_data()
        self.add_tech_indicators()

    def fetch_data(self, start_date="1990-01-01"):
        if not os.path.isfile(f"{self.ticker}.csv"):
            print(f"Fetching data for {self.ticker}...")
            hist_data = self.client.get_dataframe(self.ticker, start_date)
            hist_data.index = pd.to_datetime(hist_data.index).tz_localize(None)
            hist_data.to_csv(f"{self.ticker}.csv", index=True)
        else:
            print(f"Data for {self.ticker} already exists.")

        self.data = pd.read_csv(f"{self.ticker}.csv", index_col=0)

    def add_tech_indicators(self):
        if self.data is not None:
            self.data["SMA_20"] = self.data["close"].rolling(window=20).mean()
            self.data["SMA_50"] = self.data["close"].rolling(window=50).mean()
            self.data["EMA_20"] = self.data["close"].ewm(span=20, adjust=False).mean()
            self.data["EMA_50"] = self.data["close"].ewm(span=50, adjust=False).mean()
            self.data["RSI"] = self.compute_rsi(self.data["close"])
            self.data["MACD"], self.data["MACD_Signal"] = self.compute_macd(
                self.data["close"]
            )

            self.data["Bollinger_Upper"], self.data["Bollinger_Lower"] = (
                self.compute_bollinger_bands(self.data["close"])
            )

        else:
            raise ValueError("Data not fetched. Call fetch_data() first.")

    def compute_rsi(self, series, period=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def compute_macd(self, series, short_window=12, long_window=26, signal_window=9):
        short_ema = series.ewm(span=short_window, adjust=False).mean()
        long_ema = series.ewm(span=long_window, adjust=False).mean()
        macd = short_ema - long_ema
        signal = macd.ewm(span=signal_window, adjust=False).mean()
        return macd, signal

    def compute_bollinger_bands(self, series, window=20, num_std_dev=2):
        rolling_mean = series.rolling(window=window).mean()
        rolling_std = series.rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std_dev)
        lower_band = rolling_mean - (rolling_std * num_std_dev)
        return upper_band, lower_band
    
    