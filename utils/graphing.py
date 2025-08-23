import matplotlib.pyplot as plt
import mplfinance as mpf

class GraphingUtils:
    @staticmethod
    def plot_candlestick_chart(df):
        addplots = [mpf.make_addplot(df['Vol_EMA_22'], panel=1, color='orange', width=1.0, ylabel='Volume EMA 22')]
        mpf.plot(df, type='candle', style='charles', title='Candlestick Chart', ylabel='Price', mav=(10, 20, 50, 200), volume=True, addplot=addplots, returnfig=True)
        plt.show()


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
