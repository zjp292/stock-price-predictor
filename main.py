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
ticker = "VOO"

if not os.path.isfile("sp500.csv"):
    print("not executing")
    aapl_hist_data = client.get_dataframe(ticker, "1999-01-01")
    aapl_hist_data.index = pd.to_datetime(aapl_hist_data.index).tz_localize(None)
    aapl_hist_data.to_csv("s&p500_data.csv", index=True)

sp500_df = pd.read_csv("sp500.csv", index_col=0)

sp500_df["seven_day_rolling_avg"] = sp500_df["Close"].rolling(window=7).mean()
sp500_df["thirty_day_rolling_avg"] = sp500_df["Close"].rolling(window=30).mean()
sp500_df["return"] = sp500_df["Close"].pct_change()
sp500_df["Tomorrow"] = sp500_df["Close"].shift(-1)
sp500_df["SMA"] = sp500_df["Close"].rolling(window=5).mean()
sp500_df["EMA"] = sp500_df["Close"].ewm(span=5).mean()
sp500_df["CMA"] = sp500_df["Close"].expanding().mean()
sp500_df["lag1"] = sp500_df["Close"].shift(1)
sp500_df["Target"] = (sp500_df["Tomorrow"] > sp500_df["Close"]).astype(int)
sp500_df = sp500_df.loc["1990-01-01":].copy()

print(sp500_df)
# features = ["SMA", "EMA", "CMA", "open", "high", "low", "close", "volume"]

sp500_df = sp500_df.dropna()

# plt.plot(sp500_df["date"], sp500_df["close"], label="Closing Price")
# plt.plot(sp500_df["date"], sp500_df["SMA"], label="SMA")
# plt.plot(sp500_df["date"], sp500_df["EMA"], label="EMA")
# plt.plot(sp500_df["date"], sp500_df["CMA"], label="CMA")
# plt.xlabel("time")
# plt.ylabel("closing price")
# plt.title("aaple stock")
# plt.legend()
# plt.show()

X = sp500_df[["CMA", "SMA", "EMA"]]
y = sp500_df["Close"]

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
# x, y = sp500_df[features], sp500_df["target"]
# x, y = x[:-1], y[:-1]
# x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=False, test_size=0.2)


randForClf = RandomForestClassifier(
    n_estimators=100, min_samples_split=100, random_state=1
)
# train = sp500_df.iloc[:-100]
# test = sp500_df.iloc[-100:]
predictors = ["Close", "Volume", "Open", "High", "Low"]
# randForClf.fit(train[predictors], train["target"])

# y_pred = randForClf.predict(x_test)

# predictions = randForClf.predict(test[predictors])
# predictions = pd.Series(predictions, index=test.index)
# print("accuracy: ", metrics.precision_score(test["target"], predictions))

# combined = pd.concat([test["target"], predictions], axis=1)
# print(combined.plot())


def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:, 1]
    preds[preds >= 0.6] = 1
    preds[preds < 0.6] = 0
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined


def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []

    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i : (i + step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)

    return pd.concat(all_predictions)


# predictions = backtest(sp500_df, randForClf, predictors)
# print(predictions["Predictions"].value_counts())
# print(metrics.precision_score(predictions["Target"], predictions["Predictions"]))
# print(predictions["Target"].value_counts() / predictions.shape[0])


horizons = [2, 5, 60, 250, 1000]
new_predictors = []

for horizon in horizons:
    rolling_avgs = sp500_df.rolling(horizon).mean()

    ratio_column = f"Close_Ratio_{horizon}"
    sp500_df[ratio_column] = sp500_df["Close"] / rolling_avgs["Close"]
    trend_column = f"Trend_Column_{horizon}"
    sp500_df[trend_column] = sp500_df.shift(1).rolling(horizon).sum()["Target"]

    new_predictors += [ratio_column, trend_column]

sp500_df = sp500_df.dropna()
print(sp500_df)
model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)

predictions = backtest(sp500_df, model, new_predictors)
print(predictions["Predictions"].value_counts())
print(metrics.precision_score(predictions["Target"], predictions["Predictions"]))
