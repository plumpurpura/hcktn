import numpy as np
import pandas as pd
import cirq
from scipy.optimize import minimize

# Загрузка данных из CSV файла
file_path = '/workspace/task-1-stocks.csv'  # Укажите путь к вашему файлу
stock_data = pd.read_csv(file_path, skiprows=1, header=None)

# Расчет доходностей и ковариационной матрицы
returns = stock_data.pct_change().dropna()
mean_returns = returns.mean().values * 100
cov_matrix = returns.cov().values * 10000

# Параметры
target_risk = 0.2
target_return_range = (0.04, 0.05)
n_assets = 16  # Количество активов для кубитов
n_ansatzes = 3  # Количество квантовых анзатцев для усреднения

# Различные квантовые анзатцы
def create_quantum_ansatz(params, n_assets, ansatz_type):
    qubits = [cirq.GridQubit(0, i) for i in range(n_assets)]
    circuit = cirq.Circuit()

    # Первый тип анзатца: линейная запутанность
    if ansatz_type == 1:
        for i, qubit in enumerate(qubits):
            circuit.append(cirq.rx(params[i])(qubit))
        for i in range(n_assets - 1):
            circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))

    # Второй тип анзатца: кольцевая запутанность
    elif ansatz_type == 2:
        for i, qubit in enumerate(qubits):
            circuit.append(cirq.ry(params[i])(qubit))
        for i in range(n_assets):
            circuit.append(cirq.CNOT(qubits[i], qubits[(i + 1) % n_assets]))

    # Третий тип анзатца: двойные вращения и кросс-CNOT
    elif ansatz_type == 3:
        for i, qubit in enumerate(qubits):
            circuit.append(cirq.rz(params[i])(qubit))
            circuit.append(cirq.rx(params[n_assets + i])(qubit))
        for i in range(0, n_assets - 1, 2):
            circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))

    return circuit, qubits

# Целевая функция с усреднением по нескольким анзатцам
def quantum_portfolio_objective(params):
    portfolio_returns = []
    portfolio_risks = []

    # Запускаем несколько квантовых схем с разными анзатцами
    for ansatz_type in range(1, n_ansatzes + 1):
        circuit, qubits = create_quantum_ansatz(params, n_assets, ansatz_type)
        simulator = cirq.Simulator()
        result = simulator.simulate(circuit)

        # Рассчитываем вектор состояния и веса
        state_vector = np.abs(result.final_state_vector) ** 2
        weights = state_vector[:n_assets] / np.sum(state_vector[:n_assets])

        # Расчет доходности и риска портфеля для данного анзатца
        portfolio_return = np.dot(mean_returns[:n_assets], weights)
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix[:n_assets, :n_assets], weights)))

        portfolio_returns.append(portfolio_return)
        portfolio_risks.append(portfolio_risk)

    # Усредненные значения доходности и риска
    avg_portfolio_return = np.mean(portfolio_returns)
    avg_portfolio_risk = np.mean(portfolio_risks)

    # Ограничения по риску и доходности
    penalty = max(0, avg_portfolio_risk - target_risk) ** 2 * 100
    return_penalty = 0
    if not target_return_range[0] <= avg_portfolio_return <= target_return_range[1]:
        return_penalty = abs(avg_portfolio_return - target_return_range[1]) ** 2 * 100

    # Цель: максимизировать доходность с учетом ограничений на риск и доходность
    return -avg_portfolio_return + penalty + return_penalty

# Начальные параметры для вращений
initial_params = np.random.rand(2 * n_assets) * np.pi

# Оптимизация параметров
result = minimize(quantum_portfolio_objective, initial_params, method='COBYLA')

# Финальные вычисления для оптимизированного портфеля
optimal_params = result.x
portfolio_returns = []
portfolio_risks = []

# Запускаем симуляции с оптимизированными параметрами
for ansatz_type in range(1, n_ansatzes + 1):
    circuit, qubits = create_quantum_ansatz(optimal_params, n_assets, ansatz_type)
    simulator = cirq.Simulator()
    final_result = simulator.simulate(circuit)

    # Рассчитываем итоговые веса
    state_vector = np.abs(final_result.final_state_vector) ** 2
    optimal_weights = state_vector[:n_assets] / np.sum(state_vector[:n_assets])

    portfolio_returns.append(np.dot(mean_returns[:n_assets], optimal_weights))
    portfolio_risks.append(np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix[:n_assets, :n_assets], optimal_weights))))

# Итоговые усредненные значения доходности и риска
avg_portfolio_return = np.mean(portfolio_returns)
avg_portfolio_risk = np.mean(portfolio_risks)

print("\nКвантовый оптимизированный портфель:")
print("Средний риск портфеля:", avg_portfolio_risk, "%")
print("Средняя ожидаемая доходность:", avg_portfolio_return, "%")
print("Веса акций в портфеле (по последнему анзатцу):", optimal_weights)
