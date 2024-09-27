import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from backtesting import run_optimization_and_backtest, backtest, apply_indicators, calculate_sharpe_ratio, \
    calculate_max_drawdown


def buy_and_hold_evolution(data, initial_capital, is_crypto=False):
    """
    Calcula cómo hubiera cambiado el valor del portafolio si se hubiera invertido todo el capital al inicio.
    :param data: Datos de mercado con precios.
    :param initial_capital: Capital inicial disponible.
    :param is_crypto: Indica si se permite comprar fracciones de acciones.
    :return: Lista con la evolución del valor del portafolio.
    """
    buy_price = data.iloc[0]["Close"]

    # Para criptomonedas (Bitcoin), se permiten fracciones de acciones
    n_shares = initial_capital / buy_price if is_crypto else int(initial_capital / buy_price)

    # Calcular la evolución del portafolio Buy & Hold para todo el dataset
    portfolio_values = data["Close"] * n_shares

    # Asegurarte de que se calculen todos los valores hasta el final
    print(f"Buy & Hold Length: {len(portfolio_values)}")

    return portfolio_values


def run_for_ticker(ticker, train_file, test_file, is_crypto=False):
    """
    Ejecuta la optimización y el backtest para un ticker específico (Apple o Bitcoin).
    :param ticker: Ticker del activo (AAPL, BTC)
    :param train_file: Archivo CSV del dataset de entrenamiento.
    :param test_file: Archivo CSV del dataset de prueba.
    :param is_crypto: Indica si es un activo de criptomoneda (para usar fracciones de acciones).
    """
    # Cargar el dataset de entrenamiento
    train_data = pd.read_csv(train_file)
    train_data = train_data.copy().dropna()

    # Ejecutar la optimización y el backtest usando el dataset de entrenamiento
    best_combination, best_params, final_capital, portfolio_value, win_loss_ratio, sharpe_ratio, max_drawdown = run_optimization_and_backtest(
        train_data, ticker)

    # Imprimir resultados de la optimización
    print(f"Best Combination of Indicators for {ticker}: {best_combination}")
    print(f"Best Parameters for {ticker}: {best_params}")
    print(f"Final Portfolio Value on Train Data for {ticker}: ${final_capital:,.2f}")
    print(f"Sharpe Ratio for {ticker}: {sharpe_ratio}")
    print(f"Max Drawdown for {ticker}: {max_drawdown}")
    print(f"Win-Loss Ratio for {ticker}: {win_loss_ratio}")

    # Calcular la evolución de Buy & Hold en el dataset de entrenamiento
    buy_and_hold_train = buy_and_hold_evolution(train_data, initial_capital=1_000_000, is_crypto=is_crypto)
    print(f"Buy & Hold Final Value on Train Data for {ticker}: ${buy_and_hold_train.iloc[-1]:,.2f}")
    print(f"Train Portfolio Value Length: {len(portfolio_value)}")
    print(f"Buy & Hold Train Value Length: {len(buy_and_hold_train)}")

    # Cargar el dataset de prueba
    test_data = pd.read_csv(test_file)
    test_data = test_data.copy().dropna()

    # Aplicar la mejor estrategia al dataset de prueba
    test_data = apply_indicators(test_data, best_combination, best_params)

    # Ejecutar el backtest con la mejor estrategia en el dataset de prueba
    final_capital_test, portfolio_value_test, win_loss_ratio_test = backtest(test_data, best_combination, best_params,
                                                                             ticker)

    # Calcular métricas de rendimiento en el dataset de prueba
    sharpe_ratio_test = calculate_sharpe_ratio(np.diff(portfolio_value_test) / portfolio_value_test[:-1])
    max_drawdown_test = calculate_max_drawdown(portfolio_value_test)

    # Calcular la evolución de Buy & Hold en el dataset de prueba
    buy_and_hold_test = buy_and_hold_evolution(test_data, initial_capital=1_000_000, is_crypto=is_crypto)
    print(f"Final Portfolio Value on Test Data for {ticker}: ${final_capital_test:,.2f}")
    print(f"Sharpe Ratio on Test Data for {ticker}: {sharpe_ratio_test}")
    print(f"Max Drawdown on Test Data for {ticker}: {max_drawdown_test}")
    print(f"Buy & Hold Final Value on Test Data for {ticker}: ${buy_and_hold_test.iloc[-1]:,.2f}")
    print(f"Test Portfolio Value Length: {len(portfolio_value_test)}")
    print(f"Buy & Hold Test Value Length: {len(buy_and_hold_test)}")

    # ======= Primera gráfica =======
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(portfolio_value)), portfolio_value, label=f'{ticker} Train Portfolio Value', color='blue')
    plt.plot(np.arange(len(buy_and_hold_train)), buy_and_hold_train, label=f'{ticker} Buy & Hold Train Value',
             color='green', linestyle='--')
    plt.title(f'Portfolio Value Over Time for {ticker} (Train)')
    plt.xlabel('Time Steps')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.show()

    # ======= Segunda gráfica =======
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(portfolio_value_test)), portfolio_value_test, label=f'{ticker} Test Portfolio Value',
             linestyle='--', color='orange')
    plt.plot(np.arange(len(buy_and_hold_test)), buy_and_hold_test, label=f'{ticker} Buy & Hold Test Value', color='red',
             linestyle='--')
    plt.title(f'Portfolio Value Over Time for {ticker} (Test)')
    plt.xlabel('Time Steps')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.show()

    # ======= Tercera gráfica =======
    train_time_steps = np.arange(len(portfolio_value))  # Eje de tiempo para entrenamiento
    test_time_steps = np.arange(len(portfolio_value),
                                len(portfolio_value) + len(portfolio_value_test))  # Eje de tiempo para prueba

    plt.figure(figsize=(10, 6))
    plt.plot(train_time_steps, portfolio_value, label=f'{ticker} Train Portfolio Value', color='blue')
    plt.plot(test_time_steps, portfolio_value_test, label=f'{ticker} Test Portfolio Value', linestyle='--',
             color='orange')
    plt.plot(np.arange(len(buy_and_hold_train)), buy_and_hold_train, label=f'{ticker} Buy & Hold Train Value',
             color='green', linestyle='--')
    plt.plot(np.arange(len(buy_and_hold_test)) + len(portfolio_value), buy_and_hold_test,
             label=f'{ticker} Buy & Hold Test Value', color='red', linestyle='--')
    plt.title(f'Portfolio Value Over Time for {ticker} (Train and Test)')
    plt.xlabel('Time Steps')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.show()

# Ejecutar para Apple
#run_for_ticker('AAPL', 'aapl_5m_train.csv', 'aapl_5m_test.csv')

# Ejecutar para Bitcoin (se permite la compra de fracciones de acciones)
run_for_ticker('BTC', 'btc_project_train.csv', 'btc_project_test.csv', is_crypto=True)