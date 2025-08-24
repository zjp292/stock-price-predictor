import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


class GraphingUtils:
    @staticmethod
    def plot_candlestick_chart(df):
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02)

        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name="Candlesticks",
            ),
            row=1,
            col=1,
        )

        if df["SMA_10"].notnull().any():
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["SMA_10"],
                    mode="lines",
                    name="SMA_10",
                    line_color="purple",
                ),
                row=1,
                col=1,
            )

        if df["SMA_20"].notnull().any():
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["SMA_20"],
                    mode="lines",
                    name="SMA_20",
                    line_color="orange",
                ),
                row=1,
                col=1,
            )

        if df["SMA_50"].notnull().any():
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["SMA_50"],
                    mode="lines",
                    name="SMA_50",
                    line_color="green",
                ),
                row=1,
                col=1,
            )

        if df["SMA_200"].notnull().any():
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["SMA_200"],
                    mode="lines",
                    name="SMA_200",
                    line_color="red",
                ),
                row=1,
                col=1,
            )

        volume_colors = np.where(df["close"] >= df["open"], "green", "red")

        fig.add_trace(
            go.Bar(
                x=df.index, y=df["volume"], name="Volume", marker_color=volume_colors
            ),
            row=2,
            col=1,
        )

        if df["Vol_EMA_22"].notnull().any():
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["Vol_EMA_22"],
                    mode="lines",
                    name="Vol_EMA_22",
                ),
                row=2,
                col=1,
            )
        fig.update_yaxes(fixedrange=True)

        fig.show()

    @staticmethod
    def plot_price_history(df, list_of_columns):
        for column in list_of_columns:
            plt.plot(df.index, df[column], label=column)

        plt.xlabel("time")
        plt.ylabel("closing price")
        plt.title("Stock Price History")
        plt.legend()
        plt.show()

    @staticmethod
    def plot_predictions(y_test, predictions):
        plt.scatter(y_test, predictions)
        plt.xlabel("True Values")
        plt.ylabel("Predictions")
        plt.title("True vs Predicted Values")
        plt.show()

    @staticmethod
    def plot_residuals(y_test, predictions):
        plt.hist(y_test - predictions)
        plt.xlabel("Residuals")
        plt.title("Distribution of Residuals")
        plt.show()
