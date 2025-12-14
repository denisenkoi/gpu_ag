import numpy as np
from scipy.optimize import minimize, differential_evolution
from copy import deepcopy
import random
from ag_rewards.ag_func_correlations import calculate_correlation, objective_function_optimizer
from ag_objects.ag_obj_interpretation import create_segments, get_shift_by_idx, trim_segments_to_range


class ObjectiveFunctionWrapper:
    """
    Callable wrapper for objective_function_optimizer.
    Required for multiprocessing (workers=-1) because lambda with closure cannot be pickled.
    """
    def __init__(self, well, typewell, self_corr_start_idx, segments,
                 pearson_power, mse_power, num_intervals_self_correlation,
                 sc_power, angle_range, angle_sum_power, min_pearson_value,
                 tvd_to_typewell_shift):
        self.well = well
        self.typewell = typewell
        self.self_corr_start_idx = self_corr_start_idx
        self.segments = segments
        self.pearson_power = pearson_power
        self.mse_power = mse_power
        self.num_intervals_self_correlation = num_intervals_self_correlation
        self.sc_power = sc_power
        self.angle_range = angle_range
        self.angle_sum_power = angle_sum_power
        self.min_pearson_value = min_pearson_value
        self.tvd_to_typewell_shift = tvd_to_typewell_shift

    def __call__(self, x):
        return objective_function_optimizer(
            x,
            self.well,
            self.typewell,
            self.self_corr_start_idx,
            self.segments,
            self.pearson_power,
            self.mse_power,
            self.num_intervals_self_correlation,
            self.sc_power,
            self.angle_range,
            self.angle_sum_power,
            self.min_pearson_value,
            self.tvd_to_typewell_shift
        )


def optimizer_fit(well,
                  typewell,
                  self_corr_start_idx,
                  segments,
                  angle_range,
                  angle_sum_power,
                  segm_counts_reg,
                  num_iterations,
                  pearson_power,
                  mse_power,
                  num_intervals_self_correlation,
                  sc_power,
                  optimizer_method,
                  min_pearson_value,
                  use_accumulative_bounds,
                  tvd_to_typewell_shift=0.0,
                  multi_threaded=False):
    """
    Оптимизирует сдвиги сегментов

    Args:
        well: Объект скважины
        typewell: Объект опорной скважины
        self_corr_start_idx: Начальный индекс для корреляции
        segments: Список сегментов для оптимизации
        angle_range: Диапазон углов для ограничений
        segm_counts_reg: Список размеров региональных участков
        num_iterations: Число итераций
        pearson_power, mse_power, num_intervals_self_correlation, sc_power: Параметры для расчета метрики
        min_pearson_value: Минимальное допустимое значение для корреляции Пирсона
        use_accumulative_bounds: Флаг использования накопительного расчета границ
        multi_threaded: Не используется (для совместимости)

    Returns:
        Список результатов
    """
    from ag_numerical.ag_optimizer_utils import calculate_optimization_bounds, collect_optimization_stats

    # Рассчитываем начальную корреляцию
    well.calc_horizontal_projection(typewell, segments, tvd_to_typewell_shift)

    # Задаем ограничения для сдвигов на основе angle_range
    bounds = calculate_optimization_bounds(segments, angle_range, use_accumulative_bounds)

    # Начальные сдвиги - текущие конечные сдвиги каждого сегмента
    initial_shifts = [segment.end_shift for segment in segments]

    # Create callable wrapper for objective function (pickleable for multiprocessing)
    obj_function = ObjectiveFunctionWrapper(
        well, typewell, self_corr_start_idx, segments,
        pearson_power, mse_power, num_intervals_self_correlation,
        sc_power, angle_range, angle_sum_power, min_pearson_value,
        tvd_to_typewell_shift
    )

    # Выполняем оптимизацию
    import time
    start_time = time.time()

    if optimizer_method == 'differential_evolution':
        result = differential_evolution(
            obj_function,
            bounds=bounds,
            strategy='rand1bin',  # random base vector for exploration
            mutation=(1.5, 1.99),  # aggressive mutations
            recombination=0.99,  # almost full crossover
            popsize=500,  # large population for global search
            maxiter=1000,  # many iterations
            tol=1e-10,  # never stop early
            workers=-1,  # use all CPU cores (multiprocessing)
            updating='deferred'  # required for parallel workers
        )
    else:
        result = minimize(
            obj_function,
            initial_shifts,
            method=optimizer_method,
            bounds=bounds
        )

    elapsed_time = time.time() - start_time

    # Собираем статистику оптимизации
    optimization_stats = collect_optimization_stats(result, well, typewell, segments, pearson_power, mse_power)
    optimization_stats['elapsed_time_sec'] = round(elapsed_time, 2)
    print(f"Optimization stats: {optimization_stats}")

    # Создаем результат с оптимальными сдвигами
    optimal_segments = deepcopy(segments)
    for i, shift in enumerate(result.x):
        optimal_segments[i].end_shift = shift
        if i < len(optimal_segments) - 1:
            optimal_segments[i + 1].start_shift = shift

    # Обновляем TVT в оригинальном объекте well
    well.calc_horizontal_projection(typewell, optimal_segments, tvd_to_typewell_shift)

    # Рассчитываем итоговую корреляцию для оптимальных сегментов
    well_copy = deepcopy(well)
    well_copy.calc_horizontal_projection(typewell, optimal_segments, tvd_to_typewell_shift)
    corr, _, self_correlation, _, _, pearson, num_points, mse, _, _ = calculate_correlation(
        well_copy,
        self_corr_start_idx,
        optimal_segments[0].start_idx,
        optimal_segments[-1].end_idx,
        float('inf'),
        0,
        0,
        pearson_power,
        mse_power,
        num_intervals_self_correlation,
        sc_power,
        min_pearson_value
    )

    # Генерируем результаты с основным оптимальным решением
    results = [(corr, self_correlation, pearson, mse, num_points, optimal_segments, well_copy)]

    # Сохраняем статистику оптимизации в первом результате
    if len(results) > 0 and hasattr(results[0][6], 'optimization_stats'):
        results[0][6].optimization_stats = optimization_stats

    # Генерируем дополнительные результаты с небольшими возмущениями вокруг оптимального результата
    for _ in range(min(num_iterations - 1, 9)):  # Ограничиваем до 9 дополнительных результатов для эффективности
        noisy_segments = deepcopy(optimal_segments)

        # Добавляем небольшие случайные возмущения
        for i, segment in enumerate(noisy_segments):
            segment_len = segment.end_vs - segment.start_vs
            perturbation_size = segment_len * well.horizontal_well_step * np.tan(np.radians(angle_range * 0.1))
            shift_perturbation = random.uniform(-perturbation_size, perturbation_size)

            segment.end_shift += shift_perturbation
            if i < len(noisy_segments) - 1:
                noisy_segments[i + 1].start_shift = segment.end_shift

        # Рассчитываем корреляцию для возмущенных сегментов
        noisy_well = deepcopy(well)
        success = noisy_well.calc_horizontal_projection(typewell, noisy_segments, tvd_to_typewell_shift)
        if success:
            perturbed_corr, _, perturbed_self_correlation, _, _, perturbed_pearson, perturbed_num_points, perturbed_mse, _, _ = calculate_correlation(
                noisy_well,
                self_corr_start_idx,
                noisy_segments[0].start_idx,
                noisy_segments[-1].end_idx,
                float('inf'),
                0,
                0,
                pearson_power,
                mse_power,
                num_intervals_self_correlation,
                sc_power,
                min_pearson_value
            )

            results.append((perturbed_corr, perturbed_self_correlation, perturbed_pearson, perturbed_mse,
                            perturbed_num_points, noisy_segments, noisy_well))

    # Сортируем результаты по корреляции (по убыванию)
    results.sort(key=lambda x: x[0], reverse=True)

    return results


def get_optimizer_interpretations_list(
        well
        , type_well
        , num_iterations
        , angle_range
        , angle_sum_power
        , segments_count_curr
        , segment_len_curr
        , segm_counts_reg
        , backstep
        , interpretation_start_idx
        , manl_interpr_start_idx
        , big_segment_start_idx
        , cur_start_shift
        , basic_segments
        , current_all_segments
        , pearson_power
        , mse_power
        , num_intervals_self_correlation
        , sc_power
        , optimizer_method
        , min_pearson_value
        , use_accumulative_bounds
):
    """
    Оптимизационная версия get_monte_carlo_interpretations_list
    Имеет тот же интерфейс ввода/вывода для совместимости
    """

    # ДОБАВЛЯЕМ ИНИЦИАЛИЗАЦИЮ TVT - аналогично алгоритму Монте-Карло:
    # Если есть предыдущие сегменты, то обрезаем их до точки начала нового блока
    if current_all_segments:
        # 1. Find MD of new block start
        new_block_start_md = well.measured_depth[big_segment_start_idx]

        # 2. Trim previous segments to new_block_start_md (removes overlap, interpolates shift)
        trimmed_segments = trim_segments_to_range(current_all_segments,
                                                   current_all_segments[0].start_md,
                                                   new_block_start_md,
                                                   well)

        # 3. Use endShift of last trimmed segment as startShift for new block
        if trimmed_segments:
            cur_start_shift = trimmed_segments[-1].end_shift
            current_all_segments = trimmed_segments  # Replace with trimmed!
        else:
            cur_start_shift = get_shift_by_idx(current_all_segments, big_segment_start_idx)

    # Инициализируем TVT для начальной точки
    # Это особенно важно для первой итерации или при изменении размера сегментов
    well.tvt[big_segment_start_idx] = well.true_vertical_depth[big_segment_start_idx] + cur_start_shift

    segments = create_segments(
        well,
        segments_count_curr,
        segment_len_curr,
        big_segment_start_idx,
        cur_start_shift,
        basic_segments)

    # Используем optimizer_fit для оптимизации с передачей параметров min_pearson_value и use_accumulative_bounds
    temp_results = optimizer_fit(
        well,
        type_well,
        self_corr_start_idx=min(interpretation_start_idx, manl_interpr_start_idx),
        segments=segments,
        angle_range=angle_range,
        angle_sum_power=angle_sum_power,
        segm_counts_reg=segm_counts_reg,
        num_iterations=num_iterations,
        pearson_power=pearson_power,
        mse_power=mse_power,
        num_intervals_self_correlation=num_intervals_self_correlation,
        sc_power=sc_power,
        optimizer_method=optimizer_method,
        min_pearson_value=min_pearson_value,
        use_accumulative_bounds=use_accumulative_bounds,
        multi_threaded=False
    )

    extended_results = [(current_all_segments + result[5],) + result for result in temp_results]
    return extended_results