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
