import pandas as pd
import numpy as np

import utils

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from tqdm import tqdm

from keras import Sequential


class Montecarlo:

    def __init__(self, train_data: pd.DataFrame, test_data: pd.DataFrame, portfolios: list[int],
                 scaler: StandardScaler | MinMaxScaler = None, prices: bool = True, log: bool = False):

        self.train_data = train_data
        self.test_data = test_data
        self.scaler = scaler
        self.prices = prices
        self.data_are_scaled: bool = False
        self.scaled_train: pd.DataFrame | None = None
        self.scaled_test: pd.DataFrame | None = None
        self.n_sim: int = 100
        self.n_epochs: int = 20
        self.forecast_horizon: int = 1
        self.look_back: int = 252
        self.batch_size: int = 32
        self.portfolios = portfolios
        self.log = log
        self.selected_stocks_dict: dict = dict()
        self.loss_dict: dict = dict()
        self.stocks_df: pd.DataFrame | None = None
        self.portfolios_composition: dict[str, list[str]] | None = None

    def scale_data(self) -> None:
        print(f"Scaling data with {type(self.scaler)}")
        self.scaler.fit(self.train_data)
        self.scaled_train = pd.DataFrame(self.scaler.transform(self.train_data),
                                         index=self.train_data.index, columns=self.train_data.columns)
        self.scaled_test = pd.DataFrame(self.scaler.transform(self.test_data),
                                        index=self.test_data.index, columns=self.test_data.columns)
        self.data_are_scaled = True

    def get_info(self) -> str:
        result = f"n_sim: {self.n_sim}\nn_epochs: {self.n_epochs}\nforecast_horizon: {self.forecast_horizon}\n" \
                f"look_back: {self.look_back}\nbatch_size: {self.batch_size}\nlog: {self.log}\n" \
                 f"data_are_scaled: {self.data_are_scaled}"
        return result

    def fit(self, model_initializer: callable, n_sim: int = 100, n_epochs: int = 20, forecast_horizon: int = 1,
            look_back: int = 252, batch_size: int = 32, verbose:int = 0) -> None:

        # Update attributes
        self.n_sim = n_sim
        self.n_epochs = n_epochs
        self.forecast_horizon = forecast_horizon
        self.look_back = look_back
        self.batch_size = batch_size

        # Initialize results
        selected_stocks_dict = dict()
        loss_dict = dict()

        # Split sequence
        if self.data_are_scaled:
            x_train, y_train = utils.split_sequence(self.scaled_train, look_back, forecast_horizon)
        else:
            x_train, y_train = utils.split_sequence(self.train_data, look_back, forecast_horizon)

        # Set Start
        forecast_start = y_train[-look_back:]

        for i in tqdm(range(1, n_sim + 1)):
            # Initialize model
            model: Sequential = model_initializer()

            # Train model
            history = model.fit(x_train, y_train, verbose=verbose, epochs=n_epochs, batch_size=batch_size)

            # Save history
            loss_dict[i] = history.history["loss"]

            # Compute forecast
            forecast_predictions: np.ndarray = utils.forecast(model, forecast_start, look_back,
                                                  self.scaled_train.shape[1], len(self.scaled_test))

            forecast_predictions = forecast_predictions.reshape(forecast_predictions.shape[0],
                                                                self.scaled_train.shape[1])

            # Inverse transform
            data_forecast: np.ndarray = self.scaler.inverse_transform(forecast_predictions)
            data_forecast: pd.DataFrame = pd.DataFrame(data_forecast, index=self.test_data.index,
                                                         columns=self.train_data.columns)

            # Select stocks
            util_df_forecast: pd.DataFrame = pd.concat([self.train_data[-1:], data_forecast])
            selected_top_portfolios: dict = utils.get_ranking(util_df_forecast, self.portfolios, self.prices, self.log)
            selected_stocks_dict[i] = selected_top_portfolios

        self.selected_stocks_dict, self.loss_dict =  selected_stocks_dict, loss_dict
        self.stocks_df = self.get_stocks_df()

    def get_stocks_df(self) -> pd.DataFrame:
        result: list = []
        max_portfolio_size: int = max(self.portfolios)
        for sim in range(1, len(self.selected_stocks_dict) + 1):
            top_stocks: list[str] = self.selected_stocks_dict[sim][f"Top {max_portfolio_size}"]
            df = pd.DataFrame(top_stocks, index=list(range(max_portfolio_size))).T
            result.append(df)
        result: pd.DataFrame = pd.concat(result, ignore_index=True)
        return result

    def get_portfolios_compositions(self) -> dict[str, list[str]]:
        result = dict()

        all_stocks = set()
        for i in range(self.stocks_df.shape[0]):
            for j in range(self.stocks_df.shape[1]):
                all_stocks.add(self.stocks_df.iloc[i, j])

        for i in range(self.stocks_df.shape[1]):
            util_dict = dict()

            for stock in all_stocks:
                util_dict[stock] = 0

            for j in range(i):
                val_counts: pd.Series = self.stocks_df.iloc[:, j].value_counts()
                for s_name in val_counts.index:
                    util_dict[s_name] = util_dict[s_name] + val_counts[s_name]

            stocks_top_i: list[str] = list(pd.Series(util_dict).sort_values(ascending=False)[:i + 1].index)
            result[f"Top {i + 1}"] = stocks_top_i
        self.portfolios_composition = result
        return result

