import numpy as np
import pandas as pd
import optuna
from itertools import combinations
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, MACD
from ta.volatility import BollingerBands, AverageTrueRange

# Clase para almacenar información de las posiciones
class Position:
    def __init__(self, ticker: str, price: float, n_shares: int, timestamp: float, short: bool = False):
        self.ticker = ticker
        self.price = price
        self.n_shares = n_shares
        self.timestamp = timestamp
        self.short = short  # Indica si es una posición corta

def apply_indicators(data, indicators, params):
    if "RSI" in indicators:
        data["rsi"] = RSIIndicator(data["Close"], window=params["rsi_window"]).rsi().fillna(0)
    if "SMA" in indicators:
        data["sma"] = SMAIndicator(data["Close"], window=params["sma_window"]).sma_indicator().fillna(0)
    if "MACD" in indicators:
        macd = MACD(data["Close"], window_slow=params["slow_period"], window_fast=params["fast_period"],
                    window_sign=params["signal_period"])
        data["macd_diff"] = macd.macd_diff().fillna(0)
    if "Bollinger" in indicators:
        bb = BollingerBands(data["Close"], window=params["bollinger_window"])
        data["bollinger_h"] = bb.bollinger_hband_indicator().fillna(0)
        data["bollinger_l"] = bb.bollinger_lband_indicator().fillna(0)
    if "ATR" in indicators:
        atr = AverageTrueRange(data["High"], data["Low"], data["Close"], window=params["atr_window"])
        data["atr"] = atr.average_true_range().fillna(0)
    return data

def position_size(atr_value, capital, risk_per_trade=0.002):  # Riesgo reducido al 0.2%
    if atr_value == 0:
        return 1
    # Calcular tamaño de posición basado en el riesgo por operación (0.2% del capital)
    return max(int((capital * risk_per_trade) / atr_value), 1)

def update_portfolio_value(active_positions, current_price, capital, COM):
    """
    Actualiza el valor del portafolio considerando el capital disponible y las posiciones activas.
    """
    portfolio_value = capital  # Capital disponible
    for position in active_positions:
        if position.short:
            # En posiciones cortas, el valor aumenta si el precio cae
            position_value = (position.price - current_price) * position.n_shares * (1 - COM)
        else:
            # En posiciones largas, el valor aumenta si el precio sube
            position_value = current_price * position.n_shares * (1 - COM)
        portfolio_value += position_value
    return portfolio_value

# Cargar los datos desde el CSV y detectar si es Apple o Bitcoin
def load_data(file_path):
    data = pd.read_csv(file_path)
    if 'bitcoin' in file_path.lower():
        return data, 'BTC'
    return data, 'AAPL'

def backtest(data, indicators, params, ticker):
    capital = 1_000_000  # Initial capital
    blocked_capital = 0  # Capital blocked by short positions
    COM = 0.125 / 100  # Commission rate (0.125%)
    MIN_CAPITAL_THRESHOLD = 500_000  # Minimum capital threshold for new trades
    RISK_PER_TRADE = 0.002  # Risk per trade (0.2%)
    active_positions = []
    portfolio_value = []  # Track portfolio value at every time step
    num_long_wins, num_long_losses = 0, 0
    num_short_wins, num_short_losses = 0, 0

    # Apply indicators to dataset
    data = apply_indicators(data, indicators, params)

    # Simulate backtest
    for i, row in data.iterrows():
        # Handle closing of active positions
        for position in active_positions.copy():
            if not position.short:
                # Handle long positions
                trailing_stop = max(position.price * (1 - params["sl"]),
                                    row.Close * (1 - params.get("trailing_sl", 0.02)))
                if row.Close > position.price * (1 + params["tp"]):  # Take-profit
                    capital += row.Close * position.n_shares * (1 - COM)
                    active_positions.remove(position)
                    num_long_wins += 1
                elif row.Close < trailing_stop:  # Stop-loss or trailing stop
                    capital += row.Close * position.n_shares * (1 - COM)
                    active_positions.remove(position)
                    num_long_losses += 1
            if position.short:
                # Handle short positions
                trailing_stop = min(position.price * (1 + params["sl"]),
                                    row.Close * (1 + params.get("trailing_sl", 0.02)))
                if row.Close < position.price * (1 - params["tp"]):  # Take-profit for short
                    profit = (position.price - row.Close) * position.n_shares * (1 - COM)
                    capital += profit
                    blocked_capital -= position.price * position.n_shares
                    active_positions.remove(position)
                    num_short_wins += 1
                elif row.Close > trailing_stop:  # Stop-loss or trailing stop for short
                    loss = (row.Close - position.price) * position.n_shares * (1 + COM)
                    capital -= loss
                    blocked_capital -= position.price * position.n_shares
                    active_positions.remove(position)
                    num_short_losses += 1

        # Ensure portfolio value is tracked at every step
        current_portfolio_value = update_portfolio_value(active_positions, row.Close, capital, COM)
        portfolio_value.append(current_portfolio_value)

        # Check if new positions can be opened
        if len(active_positions) >= params["max_active_positions"]:
            continue  # Skip if max active positions are reached

        # Define the number of shares based on the ticker
        n_shares = params["n_shares"] if ticker == 'AAPL' else float(params["n_shares"])

        # Señal RSI
        if "RSI" in indicators:
            if row["rsi"] < params["rsi_lower"]:  # Señal de compra
                cost = row.Close * params["n_shares"] * (1 + COM)
                if capital - blocked_capital > cost and capital >= MIN_CAPITAL_THRESHOLD:
                    capital -= cost
                    active_positions.append(
                        Position(ticker="AAPL", price=row.Close, n_shares=params["n_shares"], timestamp=row.name,
                                 short=False))
            elif row["rsi"] > params["rsi_upper"]:  # Señal de venta (corto)
                cost = row.Close * params["n_shares"]
                if capital - blocked_capital >= cost:
                    blocked_capital += cost
                    active_positions.append(
                        Position(ticker="AAPL", price=row.Close, n_shares=params["n_shares"], timestamp=row.name,
                                 short=True))

        # Señal MACD
        if "MACD" in indicators:
            if row["macd_diff"] > 0:  # Señal de compra
                cost = row.Close * params["n_shares"] * (1 + COM)
                if capital - blocked_capital > cost and capital >= MIN_CAPITAL_THRESHOLD:
                    capital -= cost
                    active_positions.append(
                        Position(ticker="AAPL", price=row.Close, n_shares=params["n_shares"], timestamp=row.name,
                                 short=False))
            elif row["macd_diff"] < 0:  # Señal de venta (corto)
                cost = row.Close * params["n_shares"]
                if capital - blocked_capital >= cost:
                    blocked_capital += cost
                    active_positions.append(
                        Position(ticker="AAPL", price=row.Close, n_shares=params["n_shares"], timestamp=row.name,
                                 short=True))

        # Señal Bollinger Bands
        if "Bollinger" in indicators:
            if row["bollinger_l"] == 1:  # Señal de compra (banda inferior)
                cost = row.Close * params["n_shares"] * (1 + COM)
                if capital - blocked_capital > cost and capital >= MIN_CAPITAL_THRESHOLD:
                    capital -= cost
                    active_positions.append(
                        Position(ticker="AAPL", price=row.Close, n_shares=params["n_shares"], timestamp=row.name,
                                 short=False))
            elif row["bollinger_h"] == 1:  # Señal de venta (corto) (banda superior)
                cost = row.Close * params["n_shares"]
                if capital - blocked_capital >= cost:
                    blocked_capital += cost
                    active_positions.append(
                        Position(ticker="AAPL", price=row.Close, n_shares=params["n_shares"], timestamp=row.name,
                                 short=True))

        # Señal SMA (Simple Moving Average)
        if "SMA" in indicators:
            if row["Close"] > row["sma"]:  # Señal de compra
                cost = row.Close * params["n_shares"] * (1 + COM)
                if capital - blocked_capital > cost and capital >= MIN_CAPITAL_THRESHOLD:
                    capital -= cost
                    active_positions.append(
                        Position(ticker="AAPL", price=row.Close, n_shares=params["n_shares"], timestamp=row.name,
                                 short=False))
            elif row["Close"] < row["sma"]:  # Señal de venta (corto)
                cost = row.Close * params["n_shares"]
                if capital - blocked_capital >= cost:
                    blocked_capital += cost
                    active_positions.append(
                        Position(ticker="AAPL", price=row.Close, n_shares=params["n_shares"], timestamp=row.name,
                                 short=True))

        # Ajuste del tamaño de posición basado en ATR
        if "ATR" in indicators:
            atr_value = row["atr"]
            shares_per_trade = position_size(atr_value, capital - blocked_capital, RISK_PER_TRADE)

            if atr_value > 0:
                if capital - blocked_capital > row.Close * shares_per_trade * (1 + COM):
                    capital -= row.Close * shares_per_trade * (1 + COM)
                    active_positions.append(
                        Position(ticker="AAPL", price=row.Close, n_shares=shares_per_trade, timestamp=row.name,
                                 short=False))

        # Actualizar el valor del portafolio
        current_portfolio_value = update_portfolio_value(active_positions, row.Close, capital, COM)
        portfolio_value.append(current_portfolio_value)


    for position in active_positions.copy():
        if position.short:
            profit = (position.price - data.iloc[-1]['Close']) * position.n_shares * (1 - COM)
            capital += profit
            blocked_capital -= position.price * position.n_shares
        else:
            capital += data.iloc[-1]['Close'] * position.n_shares * (1 - COM)
        active_positions.remove(position)

        # Add the final portfolio value
    portfolio_value[-1] = update_portfolio_value(active_positions, data.iloc[-1]['Close'], capital, COM)

    return capital, portfolio_value, {'long_win': num_long_wins, 'long_loss': num_long_losses,
                                      'short_win': num_short_wins, 'short_loss': num_short_losses}

# Cálculo del Sharpe Ratio
def calculate_sharpe_ratio(returns, risk_free_rate=0.01):
    excess_returns = np.mean(returns) - risk_free_rate
    std_dev = np.std(returns)
    if std_dev == 0:
        return 0
    return excess_returns / std_dev

# Cálculo del Max Drawdown
def calculate_max_drawdown(portfolio_values):
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - peak) / peak
    return np.min(drawdown)