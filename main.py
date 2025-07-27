import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tiingo import TiingoClient
import os
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import seaborn

load_dotenv()
API_KEY = os.getenv("TIINGO_API_KEY")

config = {"api_key": API_KEY, "session": True}

client = TiingoClient(config)
ticker = "AAPL"

if not os.path.isfile("aapl_hist_data.csv"):
    print("not executing")
    aapl_hist_data = client.get_dataframe(ticker, "2020-01-01")
    aapl_hist_data.index = pd.to_datetime(aapl_hist_data.index).tz_localize(None)
    aapl_hist_data.to_csv("aapl_hist_data.csv", index=True)

apple_stock_df = pd.read_csv("aapl_hist_data.csv")
apple_stock_df = apple_stock_df.iloc[365:]  # this is to remove the covid outlier data

apple_stock_df["seven_day_rolling_avg"] = (
    apple_stock_df["close"].rolling(window=7).mean()
)
apple_stock_df["thirty_day_rolling_avg"] = (
    apple_stock_df["close"].rolling(window=30).mean()
)
apple_stock_df["return"] = apple_stock_df["close"].pct_change()
apple_stock_df["SMA"] = apple_stock_df["close"].rolling(window=5).mean()
apple_stock_df["EMA"] = apple_stock_df["close"].ewm(span=5).mean()
apple_stock_df["CMA"] = apple_stock_df["close"].expanding().mean()
apple_stock_df["lag1"] = apple_stock_df["close"].shift(1)
apple_stock_df["target"] = apple_stock_df["lag1"] > apple_stock_df["close"].astype(int)

features = ["SMA", "EMA", "CMA", "open", "high", "low", "close", "volume"]

apple_stock_df = apple_stock_df.dropna()

plt.plot(apple_stock_df["date"], apple_stock_df["close"], label="Closing Price")
plt.plot(apple_stock_df["date"], apple_stock_df["SMA"], label="SMA")
plt.plot(apple_stock_df["date"], apple_stock_df["EMA"], label="EMA")
plt.plot(apple_stock_df["date"], apple_stock_df["CMA"], label="CMA")
plt.xlabel("time")
plt.ylabel("closing price")
plt.title("aaple stock")
plt.legend()
plt.show()

X = apple_stock_df[["CMA", "SMA", "EMA"]]
y = apple_stock_df["close"]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# linear regression
linReg = LinearRegression(fit_intercept=True)
linReg.fit(x_train, y_train)
print(linReg.score(X, y))  # 0.995254691136916

predictions = linReg.predict(x_test)

plt.scatter(y_test, predictions)
plt.show()

plt.hist(y_test - predictions)
plt.show()

print(metrics.mean_absolute_error(y_test, predictions))  # 1.5091531233628197
print(metrics.mean_squared_error(y_test, predictions))  # 3.4416421437571643
print(np.sqrt(metrics.mean_squared_error(y_test, predictions)))  # 1.8551663385683679

# random forest classifier
x, y = apple_stock_df[features], apple_stock_df["target"]
x, y = x[:-1], y[:-1]
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=False, test_size=0.2)


randForClf = RandomForestClassifier(
    n_estimators=100, min_samples_split=100, random_state=1
)
randForClf.fit(x_train, y_train)

y_pred = randForClf.predict(x_test)
print("accuracy: ", metrics.precision_score(y_test, y_pred))
test_res = x_test.copy()

test_res["Actual"] = y_test.values
test_res["Predicted"] = y_pred
test_res = test_res.reset_index()

plt.figure(figsize=(15, 5))
plt.plot(test_res.index, test_res["Actual"], label="Actual", marker="o")
plt.plot(test_res.index, test_res["Predicted"], label="Predicted", marker="x")
plt.xlabel("Index")
plt.ylabel("Target (1 = price dropped)")
plt.title("Random Forest Classifier: Actual vs Predicted")
plt.legend()
plt.grid(True)
plt.show()


metrics.ConfusionMatrixDisplay.from_estimator(randForClf, x_test, y_test)
plt.show()
