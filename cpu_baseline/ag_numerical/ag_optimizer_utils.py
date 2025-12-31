"""
Общие утилиты для алгоритмов оптимизации интерпретации
"""
import numpy as np


def calculate_optimization_bounds(segments, angle_range, accumulative=True):
    """
    Рассчитывает границы (bounds) для оптимизации сдвигов сегментов

    Args:
        segments: Список сегментов для оптимизации
        angle_range: Диапазон углов для ограничений (в градусах)
        accumulative: Флаг накопительного расчета диапазона (True - как в визуальном оптимизаторе,
                      False - независимый расчет для каждого сегмента)

    Returns:
        list: Список кортежей (lower_bound, upper_bound) для каждого сегмента
    """
    bounds = []
    vertical_range = 0

    for segment in segments:
        segment_len = segment.end_vs - segment.start_vs

        if accumulative:
            # Накопительный расчет диапазона - более поздние сегменты имеют больший диапазон
            vertical_range += segment_len * np.tan(np.radians(angle_range))
        else:
            # Независимый расчет диапазона для каждого сегмента
            vertical_range = segment_len * np.tan(np.radians(angle_range))

        lower_bound = segment.end_shift - vertical_range
        upper_bound = segment.end_shift + vertical_range
        bounds.append((lower_bound, upper_bound))

    return bounds


def calculate_angle_bounds(segments, angle_range, normalize_factor=10.0, center_angles=None):
    """
    Рассчитывает границы для оптимизации УГЛОВ (не shift'ов)

    Args:
        segments: Список сегментов
        angle_range: Макс угол в градусах (например 10.0)
        normalize_factor: Множитель нормализации (10.0 = переменная 0.1 соответствует углу 1°)
        center_angles: Опциональный список центров bounds для каждого сегмента (в градусах).
                       Если None, центр = 0. Если float, применяется ко всем сегментам.

    Returns:
        list: Список кортежей (lower_bound, upper_bound) для нормализованных углов
    """
    normalized_range = angle_range / normalize_factor
    bounds = []

    for i, seg in enumerate(segments):
        if center_angles is None:
            center = 0.0
        elif isinstance(center_angles, (int, float)):
            center = center_angles / normalize_factor
        else:
            center = center_angles[i] / normalize_factor if i < len(center_angles) else 0.0

        bounds.append((center - normalized_range, center + normalized_range))

    return bounds


def angles_to_shifts(normalized_angles, segments, normalize_factor=10.0):
    """
    Конвертирует нормализованные углы в end_shift'ы (кумулятивно)

    Args:
        normalized_angles: Массив нормализованных углов от оптимизатора
        segments: Список сегментов
        normalize_factor: Множитель денормализации (10.0 = 0.1 -> 1°)

    Returns:
        list: end_shift для каждого сегмента
    """
    shifts = []
    current_shift = segments[0].start_shift if segments else 0.0

    for i, (norm_angle, segment) in enumerate(zip(normalized_angles, segments)):
        angle_deg = norm_angle * normalize_factor
        angle_rad = np.radians(angle_deg)
        segment_len = segment.end_vs - segment.start_vs
        delta_shift = segment_len * np.tan(angle_rad)
        current_shift = current_shift + delta_shift
        shifts.append(current_shift)

    return shifts


def collect_optimization_stats(result, well, typewell, segments, pearson_power, mse_power):
    """
    Собирает статистику по результатам оптимизации

    Args:
        result: Результат оптимизации (объект от scipy.optimize)
        well: Объект скважины
        typewell: Объект опорной скважины
        segments: Список сегментов
        pearson_power: Степень для корреляции Пирсона
        mse_power: Степень для MSE

    Returns:
        dict: Словарь со статистикой
    """
    stats = {
        'success': getattr(result, 'success', True),
        'message': getattr(result, 'message', 'Optimization completed'),
        'n_iterations': getattr(result, 'nit', 0),
        'n_function_evaluations': getattr(result, 'nfev', 0),
        'final_fun': getattr(result, 'fun', 0),
        'method': getattr(result, 'method', 'unknown'),
        'segments_count': len(segments),
        'pearson_power': pearson_power,
        'mse_power': mse_power,
        'shifts': list(result.x) if hasattr(result, 'x') else []
    }

    return stats