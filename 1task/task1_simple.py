import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Загрузка данных из CSV файла
file_path = '/workspace/task-1-stocks.csv'  # укажите путь к вашему файлу
stock_data = pd.read_csv(file_path, skiprows=1, header=None)

# Расчет средних доходностей и ковариационной матрицы
returns = stock_data.pct_change().dropna()  # расчет доходностей для каждого актива
mean_returns = returns.mean().values * 100  # средняя доходность (в процентах)
cov_matrix = returns.cov().values * 10000  # ковариационная матрица (в процентах)

# Параметры задачи
target_risk = 0.2  # целевой риск, который не должен быть превышен
n_assets = len(mean_returns)  # количество активов

# Функция для оптимизации (максимизация доходности)
def portfolio_optimization(weights):
    portfolio_return = np.dot(mean_returns, weights)  # ожидаемая доходность (в процентах)
    return -portfolio_return  # так как мы минимизируем, инвертируем для максимизации доходности

# Ограничение на риск
def risk_constraint(weights):
    portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return target_risk - portfolio_risk  # возвращаем разницу между целевым риском и текущим риском

# Условия оптимизации: суммы весов должны быть равны 1
constraints = [
    {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},  # сумма весов равна 1
    {'type': 'ineq', 'fun': risk_constraint}  # риск не должен превышать target_risk
]

# Ограничения на веса акций (от 0 до 1 для каждой акции)
bounds = [(0, 1) for _ in range(n_assets)]
initial_weights = np.array([1 / n_assets] * n_assets)  # начальные веса для равномерного распределения

# Решение задачи оптимизации
result = minimize(portfolio_optimization, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

# Результаты
optimal_weights = result.x
portfolio_return = np.dot(mean_returns, optimal_weights)
portfolio_risk = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))

print("Оптимизированный портфель:")
print("Риск портфеля:", portfolio_risk, "%")
print("Ожидаемая доходность:", portfolio_return, "%")
print("Веса акций в портфеле:", optimal_weights)
