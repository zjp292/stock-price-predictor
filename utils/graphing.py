import matplotlib.pyplot as plt


class GraphingUtils:

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
