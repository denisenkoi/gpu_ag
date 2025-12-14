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