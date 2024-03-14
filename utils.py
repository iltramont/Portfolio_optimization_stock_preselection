import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas_market_calendars as mcal
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from tqdm import tqdm

from keras import Sequential

from astropy.stats import biweight_location
from keras.layers import *
import quantstats as qs

#################
# Basic functions
#################


def price_to_returns(df: pd.DataFrame, log=False, drop_na=False) -> pd.DataFrame:
    """
    :param df: starting DataFrame with prices.
    :param log: if True, reurns log-returns. Default=False
    :param drop_na: if True, drop all rows with np.NAN values
    :return: DataFrame with returns.
    """
    if log:
        result = np.log(df / df.shift(1))
    else:
        result = (df - df.shift(1)) / df.shift(1)
    if drop_na:
        result = result.dropna()
    return result


def cumulative_returns_from_series(series: pd.Series, log=True, starting_capital=1, sub_first_row=False) -> pd.Series:
    """
    WORKS ALSO ON DATAFRAMES
    :param series: pandas series of returns.
    :param log: True if returns are in logarithmic form.
    :param starting_capital: starting capital. Default=1.
    :param sub_first_row: used if you have first row of NaN
    :return: pd.series of cumulative returns.
    """
    if log:
        result = starting_capital * np.exp(series.cumsum())
    else:     
        result = starting_capital * (1 + series).cumprod()

    if sub_first_row:
        result.iloc[0] = starting_capital

    return result


def from_cum_ret_to_price(df: pd.DataFrame, starting_prices):
    """
    Start from cumulative returns to compute the price trajectory
    """
    if len(df.columns) != len(starting_prices):
        return False
    else:
        X = df.copy()
        for col, starting_price in zip(X.columns, starting_prices):
            X[col] = starting_price * X[col]
        return X


def price_to_cumulative_returns(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    return x / x.iloc[0]    
    

def setup_tables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Used to change index with dates. Sort df by date.
    :param df: starting dataframe
    :return: df with changed column
    """
    x = df.copy()
    col_name = x.columns[0]
    x.rename(columns={col_name: "Date"}, inplace=True)
    x["Date"] = pd.to_datetime(x["Date"], format="%Y%m%d")

    x.sort_values(by=["Date"], inplace=True)
    x.reset_index(drop=True, inplace=True)
    x.set_index("Date", drop=True, inplace=True)
    return x


def split_data(data, train_size: float = 0.8):
    i = int(len(data) * train_size)
    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        train = data.iloc[: i]
        test = data.iloc[i:]
    else:
        train = data[:i]
        test = data[i:]
    return train, test


def apply_tukey(prices: pd.DataFrame, c: float = 9.0, window: int = 4) -> pd.DataFrame:

    def _tukey_loc(series):
        return biweight_location(series, c=c)

    return prices.rolling(window=window).apply(_tukey_loc)


################
# Time functions
################


def select_time_slice(df: pd.DataFrame, start: int = 20020102, end: int = 20191013) -> pd.DataFrame:
    """
    :param df: dataframe to slice
    :param start: starting day
    :param end: ending day
    :return: slice of starting df
    """
    x = df.copy()
    start = pd.to_datetime(start, format="%Y%m%d")
    end = pd.to_datetime(end, format="%Y%m%d")
    x = x.loc[x.index <= end]
    x = x.loc[x.index >= start]
    return x


def get_full_time_stock_in_period(comp_df: pd.DataFrame) -> list:
    """
    :param comp_df: dataframe used to specify if a stock is in the market index or not
    :return: list of stock names
    """
    sum_index_series = comp_df.sum(axis=0)
    objective_value = len(comp_df)
    result = []
    for stock in sum_index_series.index:
        if sum_index_series[stock] == objective_value:
            result.append(stock)
    return result


def get_trading_dates(start_period: str = "2013-01-01", end_period: str = "2017-12-31", market: str = "EUREX") -> pd.DatetimeIndex:
    calendar = mcal.get_calendar(market)
    schedule = calendar.schedule(start_date=start_period, end_date=end_period)
    dates = mcal.date_range(schedule, frequency="1D")
    return dates.strftime('%Y-%m-%d')


#####################
# Portfolio functions
####################

# TODO
def compute_market_returns(composition: pd.DataFrame, capitalization: pd.DataFrame, returns: pd.DataFrame, log=True) -> pd.Series:
    """
    Compute market return as the cap-weighted return of all the stocks.
    :param composition:
    :param capitalization:
    :param prices: price df for all the stocks composing the index []; for your period of interest

    :--RETURN: series with returns of the index (weighted by market capitalization)
    """
    # Compute weights
    weights = capitalization * composition
    weights = weights / weights.sum(axis=1).values.reshape((-1, 1))
    weights.fillna(0, inplace=True)

    weights = weights.loc[returns.index[0]:, :] #starting date of weights = starting date of returns

    weighted_returns = weights * returns
    result = pd.Series(weighted_returns.sum(axis=1), index=weights.index, name="SX5E_returns")
    return result


def get_ranking(predictions, N: list, prices: bool, log=True, sub_first_row=False):
    """
    Considering the df of predictions:
    1) Calculate the cumulative returns for each stock (for the considered period)
    2) Calculate the ranking in descending order for the cumulative returns
    3) Select the top stocks among the ranking (for all top Ns)
    """
    if prices:
        cum_returns = price_to_cumulative_returns(predictions)
    else:
        cum_returns = cumulative_returns_from_series(predictions, log=log, starting_capital=1, sub_first_row=sub_first_row)

    # Select last row
    final_return = cum_returns.iloc[-1]
    # List of stocks ordered by return
    ranking = (final_return.sort_values(ascending=False)).index
    portfolios = {f'Top {i}': list(ranking[:i]) for i in N}

    # Return len(N) lists with names of the top stocks according to the model's ranking
    # basically the stocks composing each portfolio with N stocks
    return portfolios


def calc_portfolios(assets: dict, test_ret, log=True):
    """
    This function basically extract the returns of the stocks 'chosen' from our models.
    Then compute the cumulative returns series for each strategy (top 5, top 7, top 10)

    INPUT:
    - assets : dictionary with key(=name of the portfolio), value (= list with names of assets in the portfolio)
    - test_returns : series containing the returns dataframe of all assets in the index

    RETURN-- portfolios; dictionary containing returns, cumulative returns and total performance for each portfolio
    """

    portfolios_returns = {}
    portfolios_perf = {}

    # Calculate the portfolios performance (equal weight portfolio on the top stocks from the previous ranking)
    for key, choices in assets.items():
        # Select stock from true dataframe
        returns = test_ret[choices]
        # Number of selected stocks
        n_assets = len(choices)

        # Compute daily cum returns for the stocks in the portfolio
        port_stocks_cum_ret = cumulative_returns_from_series(returns, log=log, sub_first_row=False)
        # Divide in order to have equal weight
        port_stocks_cum_ret = port_stocks_cum_ret / n_assets

        # Compute portfolio cumulative returns
        portfolio_cum_returns = port_stocks_cum_ret.sum(axis=1)

        x = price_to_returns(portfolio_cum_returns, log=log)
        # Substitute first value with the average of the first row of test returns
        x.iloc[0] = returns.iloc[0].mean()

        # store in the dictionary the series and the portfolio performance
        portfolios_returns[key] = x
        portfolios_perf[key] = 100 * (portfolio_cum_returns.iloc[-1] - 1)

    return portfolios_perf, portfolios_returns

# TODO
def plot_portfolios(portfolios_returns: dict, benchmark_returns, renderer=False):

    traces = []

    benchmark_perf = (1 + benchmark_returns).cumprod()
    
    traces.append(go.Scatter(x=benchmark_perf.index, y=benchmark_perf.values, mode='lines', name='SX5E performance'))
    
    for key, series in portfolios_returns.items():

        series_perf = (1 + series).cumprod()

        traces.append(go.Scatter(x=series_perf.index, y=series_perf, mode='lines', name=key))
        
    layout = go.Layout(title='Cumulative returns : Top N Portfolios vs SX5E (benchmark)',
                   xaxis=dict(title='Date'),
                   yaxis=dict(title='Values'), width=1000)
        
    fig = go.Figure(data=traces, layout=layout)
    
    if renderer:
        fig.show(renderer='svg')
    else:
        fig.show()


def split_sequence(sequence, look_back, forecast_horizon):
    """
    functions for training and evaluation - training DL.ipynb
    alternative to keras' TimeSeriesGenerator
    """
    X, y = list(), list()
    for i in range(len(sequence)):
        lag_end = i + look_back
        forecast_end = lag_end + forecast_horizon
        if forecast_end > len(sequence):
            break
        seq_x, seq_y = sequence[i:lag_end], sequence[lag_end:forecast_end]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def forecast(model, start, look_back, n_stocks, forecast_steps: int, verbose=0) -> np.ndarray:
    """
    start must be of shape (look_back, 1, n_stocks)
    """
    seq = list(start)
    for i in range(forecast_steps):
        x = np.array(seq[-look_back:])
        x = x.reshape(1, look_back, n_stocks)
        pred = model.predict(x, verbose=verbose)[0]
        seq.append(pred)
    return np.array(seq[look_back:])


def montecarlo(model_init: callable, train_data: pd.DataFrame, test_data: pd.DataFrame, scaler_type: str,
               look_back: int, forecast_horizon: int, portfolios: list[int], log: bool, prices: bool, n_sim: int,
               train_epochs: int = 20, batch_size: int = 32, verbose: int = 0) -> (dict, dict):

    # Initialize results
    selected_stocks_dict = dict()
    loss_dict = dict()

    # Scale data
    scaler = None
    if scaler_type.lower() == "minmaxscaler":
        scaler = MinMaxScaler()
    if scaler_type.lower() == "standardscaler":
        scaler = StandardScaler()
    train_data_scaled: np.ndarray = scaler.fit_transform(train_data)

    # Split data
    x_train, y_train = split_sequence(train_data_scaled, look_back=look_back, forecast_horizon=forecast_horizon)
    forecast_start = y_train[-look_back:]

    for i in tqdm(range(1, n_sim+1)):

        # Initialize model
        model: Sequential = model_init()

        # Train model
        history = model.fit(x_train, y_train, verbose=verbose, epochs=train_epochs, batch_size=batch_size)
        # Save history
        loss_dict[i] = history.history["loss"]

        # Compute forecast
        forecast_predictions = forecast(model, forecast_start, look_back, train_data.shape[1], len(test_data))
        forecast_predictions = forecast_predictions.reshape(forecast_predictions.shape[0], train_data.shape[1])

        # Inverse transform
        prices_forecast: np.ndarray = scaler.inverse_transform(forecast_predictions)
        prices_forecast: pd.DataFrame = pd.DataFrame(prices_forecast, index=test_data.index, columns=train_data.columns)

        # Select stocks
        util_df_forecast: pd.DataFrame = pd.concat([train_data[-1:], prices_forecast])
        selected_top_portfolios: dict = get_ranking(util_df_forecast, portfolios, prices, log)
        selected_stocks_dict[i] = selected_top_portfolios

    return selected_stocks_dict, loss_dict


def from_selected_stocks_dict_to_dataframe(selected_stocks_dict: dict, portfolios: list[int]) -> pd.DataFrame:
    result: list = []
    max_portfolio_size: int = max(portfolios)
    for sim in range(1, len(selected_stocks_dict) + 1):
        top_stocks: list[str] = selected_stocks_dict[sim][f"Top {max_portfolio_size}"]
        df = pd.DataFrame(top_stocks, index=list(range(max_portfolio_size))).T
        result.append(df)
    result: pd.DataFrame = pd.concat(result, ignore_index=True)
    return result


def compute_portfolios_from_montecarlo_dataframe(montecarlo_df: pd.DataFrame) -> dict[str, list[str]]:
    result = dict()

    all_stocks = set()
    for i in range(montecarlo_df.shape[0]):
        for j in range(montecarlo_df.shape[1]):
            all_stocks.add(montecarlo_df.iloc[i, j])

    for i in range(montecarlo_df.shape[1]):
        util_dict = dict()

        for stock in all_stocks:
            util_dict[stock] = 0

        for j in range(i):
            val_counts: pd.Series = montecarlo_df.iloc[:, j].value_counts()
            for s_name in val_counts.index:
                util_dict[s_name] = util_dict[s_name] + val_counts[s_name]

        stocks_top_i: list[str] = list(pd.Series(util_dict).sort_values(ascending=False)[:i+1].index)
        result[f"Top {i+1}"] = stocks_top_i

    return result


