import csv

def read_matrix(filename):
    """Читает матрицу стоимостей из CSV-файла."""
    with open(filename, 'r',encoding='utf8') as file:
        reader = csv.reader(file)
        header = next(reader)
        matrix = {}
        for row in reader:
            node = row[0]
            matrix[node] = {header[i]: int(row[i]) if row[i] != '-' else float('inf') for i in range(1, len(row))}
        return matrix, header

def read_nodes(filename):
    """Читает список групп с количеством людей и точками маршрута."""
    with open(filename, 'r',encoding='utf8') as file:
        reader = csv.reader(file)
        nodes = [(row[0], int(row[1])) for row in reader]
        return nodes

def find_best_route(matrix, nodes, bus_capacity, max_length):
    """Ищет оптимальные маршруты для автобусов."""
    routes = []
    remaining_nodes = nodes.copy()
    while remaining_nodes:
        current_route = ['Вокзал']
        current_capacity = bus_capacity
        current_length = 0
        while current_length < max_length and current_capacity > 0 and remaining_nodes:
            best_node = None
            min_cost = float('inf')
            for i, (node, group_size) in enumerate(remaining_nodes):  # Перебираем группы по индексу
                if current_capacity >= group_size and matrix[current_route[-1]][node] < min_cost:
                    best_node = node
                    min_cost = matrix[current_route[-1]][node]
                    best_node_index = i  # Сохраняем индекс выбранной группы
            if best_node:
                current_route.append(best_node)
                current_capacity -= remaining_nodes.pop(best_node_index)[1]  # Используем сохраненный индекс
                current_length += min_cost
            else:
                current_length += matrix[current_route[-1]]['Вокзал']  # возвращение на вокзал
                current_route.append('Вокзал')
        routes.append(current_route)
    return routes

def print_routes(routes):
    """Выводит полученные маршруты."""
    for i, route in enumerate(routes):
        print(f"Маршрут автобуса {i + 1}: {' -> '.join(route)}")

# Загрузка данных
matrix, header = read_matrix('task-2-adjacency_matrix.csv')
nodes = read_nodes('task-2-nodes.csv')

# Настройка параметров
bus_capacity = 10
max_length = 15

# Поиск оптимальных маршрутов
routes = find_best_route(matrix, nodes, bus_capacity, max_length)

# Вывод полученных маршрутов
print_routes(routes)
