import numpy as np
from scipy.stats import pearsonr
import csv
from scipy.spatial.distance import cosine
from sklearn.metrics import mean_squared_error
from ag_rewards.ag_func_self_correlation import find_intersections
from copy import deepcopy


class DummyLock:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass


def linear_interpolation(x1, y1, x2, y2, x):
    """Линейная интерполяция между двумя точками (x1, y1) и (x2, y2) для x."""
    return y1 if x1 == x2 else y1 + (y2 - y1) * (x - x1) / (x2 - x1)


# Функция для вычисления косинусного сходства
def cosine_similarity(a, b):
    return 1 - cosine(a, b)


# Функция для вычисления корреляции Пирсона
def pearson_correlation(a, b):
    corr, _ = pearsonr(a, b)
    return corr


# Функция для вычисления суммы абсолютных расхождений
def absolute_difference(a, b):
    return np.sum(np.abs(a - b)) / len(a)


# Функция для получения срезов well_data и synth_curve
def get_segment_data(well_data, start_idx, end_idx):
    a = well_data.value[start_idx:end_idx + 1]
    return a


def calc_self_correlation(tvt,
                          curve,
                          range_md,
                          save_to_file=False,
                          filename="output.csv",
                          self_correlation_vertical_step=0.006,  # анализ корреляции на себя с шагом 0.01 метра
                          normalize=True):
    min_points_for_self_correlation = 100
    # Удаляем NaN значения
    valid_indices = ~np.isnan(tvt)
    tvt = tvt[valid_indices]
    curve = curve[valid_indices]

    # Создаем равномерную сетку для TVT
    uniform_tvt = np.arange(np.min(tvt), np.max(tvt), self_correlation_vertical_step / range_md if normalize else self_correlation_vertical_step)

    # Используем np.digitize для определения, в какой интервал попадает каждое значение
    bins = uniform_tvt - 0.5 * self_correlation_vertical_step / range_md
    indices = np.digitize(tvt, bins)

    # Собираем значения для каждого интервала
    all_selected_values = [curve[indices == i] for i in range(len(uniform_tvt))]

    # Вычисляем стандартное отклонение и одновременно считаем количество парных точек
    std_values = []
    paired_points = 0
    for values in all_selected_values:
        if len(values) > 1:
            std_values.append(np.std(values))
            paired_points += len(values)
        else:
            std_values.append(np.nan)

    # Сохраняем в CSV файл, если требуется
    if save_to_file:
        max_values = max(len(values) for values in all_selected_values)
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['TVT', 'STD'] + [f'value{i+1}' for i in range(max_values)]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for tvt_val, std_val, selected_values in zip(uniform_tvt, std_values, all_selected_values):
                row = {'TVT': tvt_val, 'STD': std_val}
                for i, value in enumerate(selected_values):
                    row[f'value{i+1}'] = value
                writer.writerow(row)

    # Если все значения std_values являются NaN, возвращаем NaN, False и 0
    if paired_points < min_points_for_self_correlation or np.all(np.isnan(std_values)):
        return np.nan, 0

    # Возвращаем среднее стандартное отклонение по всем точкам, True и количество парных точек
    return np.nanmean(std_values), paired_points


def correct_self_correlation_values(mean_self_correlation_intersect,
                                    num_points,
                                    self_correlation_mult,
                                    best_corr,
                                    num_try_with_intersection):
    if num_points == 0 and mean_self_correlation_intersect == float('inf'):
        # если нет пересечений и не было в прошлых попытках
        self_correlation_mult = 100
    elif num_points > 0 and mean_self_correlation_intersect == float('inf'):
        # если есть пересечение и в прошлых попытках не было ни одного пересечения т. е. это первое
        mean_self_correlation_intersect = self_correlation_mult
        # print(self_correlation_mult)
        best_corr = best_corr * self_correlation_mult if best_corr is np.nan else best_corr
        # print(f'Best correlation changed to {best_corr}')
        num_try_with_intersection = 1
        # домножаем корреляцию на корреляцию на себя, чтобы механизм не начал "убегать" от корреляции на себя
    elif num_points == 0 and mean_self_correlation_intersect != float('inf'):
        # если нет пересечений но они были в прошлых попытках, используем среднее self_correlation
        self_correlation_mult = mean_self_correlation_intersect
    elif num_points > 0 and mean_self_correlation_intersect != float('inf'):
        # если есть пересечение и они были раньше, обновляем среднее
        num_try_with_intersection += 1
        mean_self_correlation_intersect = mean_self_correlation_intersect \
                                                + (self_correlation_mult - mean_self_correlation_intersect) \
                                                / (num_try_with_intersection)
    return best_corr,\
           num_try_with_intersection,\
           mean_self_correlation_intersect,\
           self_correlation_mult\


def calculate_correlation(well,
                          self_corr_start_idx,
                          start_idx,
                          end_idx,
                          mean_self_correlation_intersect,
                          num_try_with_intersection,
                          best_corr,
                          pearson_power,
                          mse_power,
                          num_intervals_self_correlation,
                          sc_power,
                          min_pearson_value,
                          lock=None):

    self_correlation_power = 2
    num_points_power = 0.7
    max_delta_tvt = 0.15 / well.md_range
    self_correlation_enable = False

    if np.any(np.isnan(well.synt_curve[start_idx: end_idx + 1])):
        mse = 1
        pearson = 0
    else:
        mse = mean_squared_error(well.value[start_idx: end_idx + 1],
                                 well.synt_curve[start_idx: end_idx + 1])

        x = well.value[start_idx: end_idx + 1]
        y = well.synt_curve[start_idx: end_idx + 1]

        const_x = (x == x[0]).all()
        const_y = (y == y[0]).all()

        if not const_x and not const_y:
            pearson = pearson_correlation(well.value[start_idx: end_idx + 1],
                                          well.synt_curve[start_idx: end_idx + 1])
        else:
            pearson = 0

        pearson = max(pearson, min_pearson_value)

    if self_correlation_enable:
        self_correlation, num_points = calc_self_correlation(
            well.tvt[self_corr_start_idx: end_idx + 1],
            well.value[self_corr_start_idx: end_idx + 1],
            well.md_range,
            normalize=well.normalized
        )
    else:
        self_correlation, num_points = 1, 1

    # consistency_factor = calc_consistency(well.tvt[interpretation_start_idx: end_idx + 1], well.curve[interpretation_start_idx: end_idx + 1])
    # consistency_factor = consistency_factor

    intersections_mult = 1
    intersections_count = 1
    intersections, intersections_count = find_intersections(
        well.tvt[self_corr_start_idx: end_idx + 1],
        well.value[self_corr_start_idx: end_idx + 1],
        well,
        max_delta_tvt=max_delta_tvt,
        num_intervals=num_intervals_self_correlation,
        start_idx=start_idx
    )
    intersections_mult = 1 / sc_power ** intersections_count
    # intersections_mult = 1


    # приводим все метрики к сопоставимому виду, пусть для идеального решения величина стремится к бесконечности
    pearson_mult = 1 / (1 - pearson) if pearson > 0 else 1
    mse_mult = 1 / mse
    self_correlation_mult = pow((1 / self_correlation), self_correlation_power)\
                            * pow(num_points, num_points_power)

    corr = pow(pearson_mult, pearson_power) \
           * pow(mse_mult, mse_power) \
           * self_correlation_mult \
           * intersections_mult \
           / 1e5

    return corr,\
           best_corr,\
           self_correlation,\
           mean_self_correlation_intersect,\
           num_try_with_intersection, \
           pearson, \
           num_points, \
           mse, \
           intersections_mult, \
           intersections_count


def objective_function_optimizer(shifts,
                                 well,
                                 typewell,
                                 self_corr_start_idx,
                                 segments,
                                 pearson_power,
                                 mse_power,
                                 num_intervals_self_correlation,
                                 sc_power,
                                 angle_range,
                                 angle_sum_power,
                                 min_pearson_value,
                                 tvd_to_typewell_shift=0.0,
                                 visualizer=None):
    """
    Целевая функция для оптимизации сдвигов сегментов с дополнительными штрафами за углы

    Args:
        shifts: Список значений сдвигов для конца каждого сегмента
        well: Объект скважины
        typewell: Объект опорной скважины
        self_corr_start_idx: Начальный индекс для корреляции
        segments: Список сегментов для оптимизации
        pearson_power, mse_power, num_intervals_self_correlation, sc_power: Параметры для расчета метрики
        angle_range: Максимально допустимый угол сегмента относительно горизонтали
        angle_sum_power: Степень штрафа за сумму углов между сегментами
        min_pearson_value: Минимальное допустимое значение для корреляции Пирсона
        visualizer: Опциональный объект визуализатора для отображения промежуточных результатов

    Returns:
        float: Значение целевой функции для минимизации
    """
    # Проверяем, нужно ли остановить оптимизацию (для визуализатора)
    if visualizer and getattr(visualizer, 'should_stop', False):
        return float('inf')

    # Проверяем, находимся ли мы в режиме паузы (для визуализатора)
    if visualizer and getattr(visualizer, 'paused', False):
        import matplotlib.pyplot as plt
        plt.pause(0.1)
        visualizer.fig.canvas.draw_idle()

    # Создаем копию сегментов, чтобы не изменять оригинал
    # Создаем копию сегментов, чтобы не изменять оригинал
    segments_copy = deepcopy(segments)

    # Обновляем сдвиги сегментов на основе предоставленных значений
    for i, segment in enumerate(segments_copy):
        segment.end_shift = shifts[i]
        if i < len(segments_copy) - 1:
            segments_copy[i + 1].start_shift = shifts[i]

        # Пересчитываем угол для каждого сегмента
        segment.calc_angle()

    # Расчет штрафа за превышение угла angle_range
    angle_penalty = 0
    for segment in segments_copy:
        if abs(segment.angle) > angle_range:
            # Резкое увеличение штрафа, если угол превышает angle_range
            angle_penalty += 1000 * (abs(segment.angle) - angle_range) ** 2

    # Расчет штрафа за сумму углов между сегментами
    angle_sum = 0
    for i in range(1, len(segments_copy)):
        # Вычисляем абсолютную разницу между соседними углами
        angle_diff = abs(segments_copy[i].angle - segments_copy[i - 1].angle)
        angle_sum += angle_diff

    # Штраф за сумму углов с регулируемой степенью
    angle_sum_penalty = angle_sum ** angle_sum_power

    # Вычисляем проекцию
    success = well.calc_horizontal_projection(typewell, segments_copy, tvd_to_typewell_shift)
    if not success:
        return float('inf')  # Возвращаем большое значение для недопустимых сдвигов

    # Вычисляем корреляцию Пирсона и MSE между оригинальной и синтетической кривыми
    start_idx = segments_copy[0].start_idx
    end_idx = segments_copy[-1].end_idx

    # Проверяем на NaN значения
    if np.any(np.isnan(well.synt_curve[start_idx: end_idx + 1])):
        return float('inf')

    # Вычисляем MSE
    x = well.value[start_idx: end_idx + 1]
    y = well.synt_curve[start_idx: end_idx + 1]
    mse = mean_squared_error(x, y)

    # Вычисляем корреляцию Пирсона
    const_x = (x == x[0]).all()
    const_y = (y == y[0]).all()

    if not const_x and not const_y:
        pearson = pearson_correlation(x, y)
    else:
        pearson = 0

    # Ограничиваем минимальное значение Пирсона
    pearson = max(pearson, min_pearson_value)

    # Вычисляем пересечения, если требуется
    if num_intervals_self_correlation > 0:
        _, intersections_count = find_intersections(
            well.tvt[self_corr_start_idx: end_idx + 1],
            well.value[self_corr_start_idx: end_idx + 1],
            well,
            max_delta_tvt=0.15 / well.md_range,
            num_intervals=num_intervals_self_correlation,
            start_idx=start_idx
        )
        intersections_component = sc_power ** intersections_count
    else:
        intersections_component = 1.0
        intersections_count = 0

    # Комбинируем метрики
    # Переводим все в метрику, которую нужно минимизировать
    pearson_component = 1 - pearson  # Минимизируем (1 - pearson)

    # Комбинация, взвешенная соответствующими степенями с добавлением штрафов
    metric = ((pearson_component ** pearson_power) *
              (mse ** mse_power) *
              (1.0 / intersections_component)) * (1 + angle_penalty + angle_sum_penalty)

    # Обновляем визуализацию, если она предоставлена
    if visualizer is not None and hasattr(visualizer, 'update_plot'):
        visualizer.update_plot(shifts, metric, mse, pearson)

    return metric