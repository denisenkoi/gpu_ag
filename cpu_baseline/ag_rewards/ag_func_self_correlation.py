import numpy as np

def calc_consistency(tvt, curve):
    # Шаг 1: Разбиваем диапазон curve на 20 интервалов
    min_curve, max_curve = curve.min(), curve.max()
    intervals = np.linspace(min_curve, max_curve, 21)

    # Словарь для хранения результатов
    interval_tvt_counts = {}

    # Шаг 2: Интерполяция и подсчет для каждого интервала
    for i in range(len(intervals) - 1):
        lower_bound = intervals[i]
        upper_bound = intervals[i + 1]

        # Находим индексы, где значения curve попадают в текущий интервал
        indices = np.where((curve >= lower_bound) & (curve <= upper_bound))

        # Интерполированные значения tvt для текущего интервала
        relevant_tvt = tvt[indices]

        # Шаг 3: Подсчет уникальных значений с учетом разницы в 0.5
        unique_tvt = set()
        for value in relevant_tvt:
            # Округляем до ближайшего значения с шагом 0.5 и добавляем в множество
            rounded_value = round(value * 2) / 2
            unique_tvt.add(rounded_value)

        # Сохраняем количество уникальных значений для интервала
        interval_tvt_counts[(lower_bound, upper_bound)] = len(unique_tvt)

    return interval_tvt_counts


def group_values(values, threshold):
    grouped = []
    temp_group = []
    first_group_idx = 0

    for value in sorted(values):
        if not temp_group or value - temp_group[0] <= threshold:
            temp_group.append(value)
        else:
            grouped.append(temp_group)
            temp_group = [value]

    if temp_group:
        grouped.append(temp_group)

    return grouped


def find_intersections(tvt,
                       curve,
                       well,
                       max_delta_tvt,
                       num_intervals,
                       start_idx=0):

    target_values = np.linspace(well.min_curve, well.max_curve, num_intervals)

    # Используем словарь для хранения результатов
    intersections = {val: [] for val in target_values}

    if start_idx is None:
        print('start_idx is None')
    # Поиск пересечений и соответствующих значений tvt, начиная с start_idx
    for i in range(start_idx + 1, len(curve)):
        for target in target_values:
            if (curve[i-1] - target) * (curve[i] - target) <= 0:
                interp_tvt = np.interp(target, [curve[i-1], curve[i]], [tvt[i-1], tvt[i]])
                intersections[target].append(interp_tvt)

    # Группировка и подсчет уникальных значений tvt для каждого пересечения
    intersections_count = 0
    for target in intersections:
        grouped_tvt = group_values(intersections[target], max_delta_tvt)
        intersections[target] = 0 if len(grouped_tvt) < 2 else len(grouped_tvt)
        intersections_count += intersections[target]

    # Сохранение результатов в класс Well
    well.intersections = intersections
    well.intersections_count = intersections_count

    return intersections, intersections_count