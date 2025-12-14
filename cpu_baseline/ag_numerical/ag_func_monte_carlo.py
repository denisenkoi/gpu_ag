# Add these imports at the top of the file
import math
from copy import deepcopy
import random
from multiprocessing import Pool, Manager
from ag_rewards.ag_func_correlations import calculate_correlation
from ag_numerical.ag_func_interpretation import select_unique_candidates
from ag_objects.ag_obj_well import Well
from ag_objects.ag_obj_typewell import TypeWell
from ag_objects.ag_obj_interpretation import create_segments, get_shift_by_idx, calculate_average_shift_difference, cut_manual_interpretation_part
from tqdm import tqdm
import numpy as np

# Import the optimizer functions
from ag_numerical.ag_func_optimizer import optimizer_fit, get_optimizer_interpretations_list



def monte_carlo_iteration(args):
    (iteration,
     well,
     typewell,
     self_corr_start_idx,
     segments,
     angle_range,
     segm_counts_reg,
     best_corr,
     best_segments,
     mean_self_correlation_intersect,
     num_try_with_intersection,
     pearson_power,
     mse_power,
     num_intervals_self_correlation,
     sc_power,
     lock) = args

    segments_copy = deepcopy(segments)
    segments_copy = update_segments(
        segments_copy,
        well,
        angle_range,
        segm_counts_reg
    )
    corr = 0
    success = well.calc_horizontal_projection(typewell, segments_copy)
    if success:
        corr, \
        best_corr, \
        self_correlation, \
        mean_self_correlation_intersect, \
        num_try_with_intersection, \
        pearson, \
        num_points, \
        mse, \
        intersections_mult, \
        intersections_count = calculate_correlation(
            well,
            self_corr_start_idx,
            start_idx=segments_copy[0].start_idx,
            end_idx=segments_copy[-1].end_idx,
            best_corr=best_corr,
            mean_self_correlation_intersect=mean_self_correlation_intersect,
            num_try_with_intersection=num_try_with_intersection,
            pearson_power=pearson_power,
            mse_power=mse_power,
            num_intervals_self_correlation=num_intervals_self_correlation,
            sc_power=sc_power,
            lock=lock
        )
        return corr, self_correlation, pearson, mse, num_points, segments_copy, deepcopy(well)
    return None


def monte_carlo_fit(well: Well,
                    typewell: TypeWell,
                    self_corr_start_idx,
                    segments,
                    angle_range,
                    segm_counts_reg,
                    num_iterations,
                    pearson_power,
                    mse_power,
                    num_intervals_self_correlation,
                    sc_power,
                    min_pearson_value,
                    multi_threaded=False
                    ):
    with (Manager() as manager):
        well.calc_horizontal_projection(typewell,
                                        segments)

        init_corr, \
        init_best_corr, \
        init_self_correlation, \
        init_mean_self_correlation_intersect, \
        init_num_try_with_intersection, \
        init_pearson, \
        init_num_points, \
        init_mse, \
        init_intersections_mult, \
        init_intersections_count, = \
                calculate_correlation(well,
                                      self_corr_start_idx,
                                      segments[0].start_idx,
                                      segments[-1].end_idx,
                                      float('inf'),
                                      0,
                                      0,
                                      pearson_power,
                                      mse_power,
                                      num_intervals_self_correlation,
                                      sc_power,
                                      min_pearson_value,
                                      lock=None)

        best_corr = manager.Value('d', init_corr)
        mean_self_correlation_intersect = manager.Value('d', init_mean_self_correlation_intersect)
        num_try_with_intersection = manager.Value('d', init_num_try_with_intersection)
        best_segments = manager.list(segments)
        lock = manager.Lock()

        args_list = [
            (
                iteration,
                well,
                typewell,
                self_corr_start_idx,
                segments,
                angle_range,
                segm_counts_reg,
                best_corr,
                best_segments,
                mean_self_correlation_intersect,
                num_try_with_intersection,
                pearson_power,
                mse_power,
                num_intervals_self_correlation,
                sc_power,
                lock
            )
            for iteration in range(num_iterations)
        ]
        results = None
        if multi_threaded:
            # Многопоточный режим
            with Pool() as pool:
                results = pool.map(monte_carlo_iteration, args_list)
        else:
            # Однопоточный режим
            for arg in args_list:
                monte_carlo_iteration(arg)

        # print(num_try_with_intersection.value)
        results = [result for result in results if result is not None]

        return results


def get_monte_carlo_interpretations_list(
          well
        , type_well
        , num_iterations
        , angle_range
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
        , min_pearson_value
):

    segments_begin_error = False
    segments = create_segments(
        well,
        segments_count_curr,
        segment_len_curr,
        big_segment_start_idx,
        cur_start_shift,
        basic_segments)

    # подгоняем по монте карло блок сегментов
    temp_results = monte_carlo_fit(
        well,
        type_well,
        self_corr_start_idx=min(interpretation_start_idx, manl_interpr_start_idx),
        segments=segments,
        angle_range=angle_range,
        segm_counts_reg=segm_counts_reg,
        num_iterations=num_iterations,
        pearson_power=pearson_power,
        mse_power=mse_power,
        num_intervals_self_correlation=num_intervals_self_correlation,
        sc_power=sc_power,
        min_pearson_value=min_pearson_value,
        multi_threaded=True
    )

    extended_results = [(current_all_segments + result[5],) + result for result in temp_results]
    return extended_results


def make_monte_carlo_interpretation(
        well:Well,
        type_well,
        well_manl_interp,
        interpretation_start_idx,
        start_shift,
        pearson_power,
        mse_power,
        num_intervals_sc,
        sc_power,
        angle_range,
        angle_sum_power,
        num_iterations,
        interpretation_end_idx,
        segment_len_list,
        manl_interpr,
        method,
        optimizer_method, # выбор метода, если это оптимизатор
        segments_count,  # количество сегментов для одного прохода оптимизатора
        min_pearson_value,  # минимальное значение корреляции Пирсона
        use_accumulative_bounds  # использовать накопительные bounds
):
    """
    Perform interpretation using either Monte Carlo or optimization method

    Args:
        well: Well object
        type_well: TypeWell object
        well_manl_interp: Manual interpretation for comparison
        interpretation_start_idx: Starting index for interpretation
        start_shift: Initial shift value
        pearson_power, mse_power, num_intervals_self_correlation, sc_power: Parameters for correlation calculation
        num_iterations: Number of iterations
        interpretation_end_idx: Optional end index for interpretation
        segment_len_list: List of segment lengths to try
        manl_interpr: Optional manual interpretation
        method: "monte_carlo" or "optimizer"
        segments_count: Number of segments for one optimizer pass (default: 10)

    Returns:
        Tuple containing the interpretation results
    """

    num_unique_segments = 1
    # список должен быть упорядочен по убыванию

    landing_segments_len = 50
    segm_counts_reg = [2, 4, 6, 10, 15, 20]
    backstep = 0
    typewell_over_range = 0.5
    initial_segments = None

    if manl_interpr is not None and manl_interpr[0].start_idx < interpretation_start_idx:
        manl_interpr_start_idx = manl_interpr[0].start_idx
        print(f'Using manual interpretation for self correlation idx: {interpretation_start_idx} -> {manl_interpr_start_idx}')
        initial_segments = cut_manual_interpretation_part(
            well,
            manl_interpr,
            interpretation_start_idx)
        well.calc_horizontal_projection(type_well, initial_segments)
        start_shift = initial_segments[-1].end_shift
    else:
        manl_interpr_start_idx = interpretation_start_idx

        # проверка, что начало интерпретации находится в диапазоне опорной скважины и корректировка первой точки при необходимости
        if well.true_vertical_depth[interpretation_start_idx] - start_shift > np.max(type_well.true_vertical_depth):
            if abs(well.true_vertical_depth[interpretation_start_idx] - start_shift - np.max(
                    type_well.true_vertical_depth)) * well.md_range > typewell_over_range:
                print('Start_tvt depth over typewell depth range')
            else:
                start_shift = well.true_vertical_depth[interpretation_start_idx] - np.max(type_well.true_vertical_depth)

        if well.true_vertical_depth[interpretation_start_idx] - start_shift < np.min(type_well.true_vertical_depth):
            if abs(well.true_vertical_depth[interpretation_start_idx] - start_shift - np.min(
                    type_well.true_vertical_depth)) * well.md_range > typewell_over_range:
                print('Start_tvt depth under typewell depth range')
            else:
                print('Start_tvt depth under typewell depth range. small ... corrected')
                start_shift = -(well.true_vertical_depth[interpretation_start_idx] - np.min(type_well.true_vertical_depth))

    # изменения для лендинга
    interpretation_end_idx = len(well.measured_depth) - 1 if interpretation_end_idx is None else interpretation_end_idx

    segments_count = math.floor(len(well.measured_depth) / segment_len_list[0]) if segments_count == 0 else segments_count

    print(f'segments count: {segments_count}, well len: {len(well.measured_depth)}, segm_len in meters: {segment_len_list[0] * well.horizontal_well_step}, segm_len in points: {segment_len_list[0]}, big_segm_len: {segments_count * segment_len_list[0]}, interpretation_len: {len(well.measured_depth) - interpretation_start_idx}')
    print(f'Using {method} method for interpretation')

    # интерпретация
    all_segments = big_segments_monte_carlo_interpretation(
        well=well,
        type_well=type_well,
        num_unique_segments=num_unique_segments,
        interpretation_start_idx=interpretation_start_idx,
        interpretation_end_idx=interpretation_end_idx,
        segment_len_list=segment_len_list,
        landing_segments_len=landing_segments_len,
        segments_count=segments_count,
        self_correlation_start_from_idx=manl_interpr_start_idx,
        backstep=backstep,
        num_iterations=num_iterations,
        angle_range=angle_range,
        angle_sum_power=angle_sum_power,
        segm_counts_reg=segm_counts_reg,
        start_shift=start_shift,
        pearson_power=pearson_power,
        mse_power=mse_power,
        num_intervals_self_correlation=num_intervals_sc,
        sc_power=sc_power,
        initial_segments=initial_segments,
        method=method,
        optimizer_method=optimizer_method,
        min_pearson_value=min_pearson_value,
        use_accumulative_bounds=use_accumulative_bounds
    )

    return \
        well, \
        all_segments,\
        interpretation_start_idx,\
        type_well,\
        segm_counts_reg,\
        well_manl_interp,\
        angle_range

def update_segments(segments,
                    well,
                    angle_range,
                    segm_counts_reg):
    reg_mult = 2 / len(segm_counts_reg)
    gl_mult = 0.5
    loc_mult = 0.5

    all_range = len(segm_counts_reg) * reg_mult + gl_mult + loc_mult

    dep_range_mult = math.sin(math.radians(angle_range))
    stochastic_multiplier = random.uniform(-dep_range_mult * loc_mult, dep_range_mult * loc_mult)
    stochastic_factor_gl = random.uniform(-dep_range_mult * gl_mult, dep_range_mult * gl_mult)
    regional_stochastic_mult = random.uniform(-dep_range_mult, dep_range_mult)
    regional_stochastic_mult = regional_stochastic_mult * reg_mult
    for segm_num, segment in enumerate(segments):
        stochastic_factor_loc = random.uniform(-stochastic_multiplier, stochastic_multiplier)
        segment.end_shift += stochastic_factor_gl * (well.vs_thl[segment.end_idx] - well.vs_thl[segments[0].start_idx])
        segment.end_shift += stochastic_factor_loc * (well.vs_thl[segment.end_idx] - well.vs_thl[segment.start_idx])
        if segm_num > 0:
            segment.start_shift = segments[segm_num - 1].end_shift

        for segments_count in segm_counts_reg:
            if segments_count > len(segments):  # если длина регионального участа больше длины segments
                break
            if (segm_num + 1) % segments_count == 0:  # если номер сегмента+1 делится на длину регионального участка
                stochastic_factor_regional = random.uniform(-regional_stochastic_mult, regional_stochastic_mult)
                # цикл по сегментам на региональном участке
                for curr_segment_num in range(segm_num - segments_count + 1, segm_num + 1):
                    # VS конца текущего сегмента - VS начала первого сегмента регионального участка
                    vs_range = well.vs_thl[segments[curr_segment_num].end_idx] \
                               - well.vs_thl[segments[segm_num - segments_count + 1].start_idx]
                    segments[curr_segment_num].end_shift += stochastic_factor_regional * vs_range
                    if curr_segment_num < len(segments) - 1:
                        segments[curr_segment_num + 1].start_shift += stochastic_factor_regional * vs_range

    for segment in segments:
        segment.calc_angle()

    return segments

# Modify big_segments_monte_carlo_interpretation to use the appropriate method
def big_segments_monte_carlo_interpretation(
        well,
        type_well,
        num_unique_segments,
        interpretation_start_idx,
        interpretation_end_idx,
        segment_len_list,
        landing_segments_len,
        segments_count,
        backstep,
        self_correlation_start_from_idx,
        num_iterations,
        angle_range,
        angle_sum_power,
        segm_counts_reg,
        start_shift,
        pearson_power,
        mse_power,
        num_intervals_self_correlation,
        sc_power,
        basic_segments=None,
        initial_segments=None,
        method="monte_carlo",
        optimizer_method="",
        min_pearson_value=-1,
        use_accumulative_bounds=True
        ):
    """
    Main interpretation function that uses either Monte Carlo or optimization method
    """
    manl_interpr_start_idx = interpretation_start_idx if initial_segments is None else initial_segments[0].start_idx
    best_segments_list = None
    # цикл по разным размерам сегментов
    previous_best_segments_list = None
    sorted_results = []

    if (interpretation_start_idx >= interpretation_end_idx):
        print('interpretation_end must be bigger than interpretation_start')

    for segment_len_curr in segment_len_list:
        big_segment_start_idx = interpretation_start_idx
        # интерпретация, цикл по большим сегментам
        interpretation_end_idx = len(well.measured_depth) - segments_count * segment_len_curr if interpretation_end_idx == 0 else interpretation_end_idx

        # если длина сегмента больше, чем длина всей скважины, то пропускаем, переходим к меньшему размеру сегмента
        if segment_len_curr > interpretation_end_idx - interpretation_start_idx:
            continue
        all_segments = [] if initial_segments is None else initial_segments
        step_size = segment_len_curr * (segments_count - backstep)
        angle_range_current = angle_range * segment_len_curr / segment_len_list[0] # вычисляем диапазон углов
        print(f'angle_range_current: {angle_range_current}, step_size: {step_size}, segment_len_curr: {segment_len_curr}, segments_count_curr: {segments_count}')

        i = 0
        pbar_devider = 5
        pbar = tqdm(total=(interpretation_end_idx - interpretation_start_idx) / step_size / pbar_devider + 1)
        num_segments_completed = 0

        pbar.update(big_segment_start_idx - interpretation_start_idx)
        # цикл по большим сегментам
        while big_segment_start_idx <= interpretation_end_idx:
            # Если это самая первая интерпретация, то интерпретируем первый большой сегмент с нуля
            if i == 0 and best_segments_list is None:
                well.tvt[big_segment_start_idx] = well.true_vertical_depth[big_segment_start_idx] + start_shift

                # Use appropriate method based on parameter
                if method == "optimizer":
                    results = get_optimizer_interpretations_list(
                        well
                        , type_well
                        , num_iterations
                        , angle_range_current
                        , angle_sum_power=angle_sum_power
                        , segments_count_curr=segments_count
                        , segment_len_curr=segment_len_curr
                        , segm_counts_reg=segm_counts_reg
                        , backstep=backstep
                        , interpretation_start_idx=interpretation_start_idx
                        , manl_interpr_start_idx=manl_interpr_start_idx
                        , big_segment_start_idx=big_segment_start_idx
                        , cur_start_shift=start_shift  # Важно: передаем начальный сдвиг
                        , basic_segments=basic_segments
                        , current_all_segments=all_segments
                        , pearson_power=pearson_power
                        , mse_power=mse_power
                        , num_intervals_self_correlation=num_intervals_self_correlation
                        , sc_power=sc_power
                        , optimizer_method=optimizer_method
                        , min_pearson_value=min_pearson_value
                        , use_accumulative_bounds=use_accumulative_bounds
                    )
                else:
                    results = get_monte_carlo_interpretations_list(
                        well
                        , type_well
                        , num_iterations
                        , angle_range_current
                        , segments_count
                        , segment_len_curr
                        , segm_counts_reg
                        , backstep
                        , interpretation_start_idx
                        , manl_interpr_start_idx
                        , big_segment_start_idx
                        , start_shift
                        , basic_segments
                        , all_segments
                        , pearson_power
                        , mse_power
                        , num_intervals_self_correlation
                        , sc_power
                        , min_pearson_value
                    )

                if len(results) == 0:
                    print('empty results on first fit!')
            # если вторая+ итерация или первая итерация второго+, то пристраиваем новую интерпретацию ко всем
            # альтернативным вариантам интерпретаций
            #
            else:
                results = []
                # перебираем лучшие прошлые решения и достраиваем их новыми
                for j, (current_all_segments, _, _, _, _, _, best_segments, best_well) in enumerate(best_segments_list):
                    if big_segment_start_idx + segment_len_curr >= len(well.measured_depth):
                        big_segment_start_idx = len(well.measured_depth) - segment_len_curr - 1

                    # тут хитро берем шифт интерполированно из segments.
                    cur_start_shift = get_shift_by_idx(current_all_segments, big_segment_start_idx)

                    # если это первый сегмент последующего размера сегментов, то обнуляем сегменты и
                    # расчитывем начальный tvt нормализуя
                    if i == 0:
                        # тут для итераций с уменьшением длины сегментов похоже ошибка!!!!!!!!
                        well.tvt[big_segment_start_idx] = well.true_vertical_depth[big_segment_start_idx] + start_shift
                        current_all_segments = all_segments

                    if all_segments != [] and current_all_segments[-1].end_idx != big_segment_start_idx:
                        print('current_all_segments[-1].end_idx != big_segment_start_idx')

                    if previous_best_segments_list and j >= len(previous_best_segments_list):
                        print('govno1')

                    # Use appropriate method based on parameter
                    if method == "optimizer":
                        extended_results = get_optimizer_interpretations_list(
                            well
                            , type_well
                            , num_iterations
                            , angle_range_current
                            , angle_sum_power=angle_sum_power
                            , segments_count_curr=segments_count
                            , segment_len_curr=segment_len_curr
                            , segm_counts_reg=segm_counts_reg
                            , backstep=backstep
                            , interpretation_start_idx=interpretation_start_idx
                            , manl_interpr_start_idx=manl_interpr_start_idx
                            , big_segment_start_idx=big_segment_start_idx
                            , cur_start_shift=cur_start_shift  # Важно: передаем текущий сдвиг
                            , basic_segments=previous_best_segments_list[j] if previous_best_segments_list else basic_segments
                            , current_all_segments=current_all_segments
                            , pearson_power=pearson_power
                            , mse_power=mse_power
                            , num_intervals_self_correlation=num_intervals_self_correlation
                            , sc_power=sc_power
                            , optimizer_method=optimizer_method
                            , min_pearson_value=min_pearson_value
                            , use_accumulative_bounds=use_accumulative_bounds
                        )
                    else:
                        extended_results = get_monte_carlo_interpretations_list(
                            well
                            , type_well
                            , num_iterations
                            , angle_range_current
                            , segments_count
                            , segment_len_curr
                            , segm_counts_reg
                            , backstep
                            , interpretation_start_idx
                            , manl_interpr_start_idx
                            , big_segment_start_idx
                            , cur_start_shift
                            , previous_best_segments_list[j] if previous_best_segments_list else basic_segments
                            , current_all_segments
                            , pearson_power
                            , mse_power
                            , num_intervals_self_correlation
                            , sc_power
                            , method
                            , min_pearson_value
                        )
                    
                    if len(extended_results) == 0:
                        print('empty results!')
                    results.extend(extended_results)

            if len(results) == 0:
                print('empty results on next fit!')
                
            # Упорядочиваем результаты по corr по убыванию
            sorted_results = sorted((r for r in results if r is not None), key=lambda x: x[1], reverse=True)

            make_second_fit = False
            best_segments_list = select_unique_candidates(sorted_results,
                                                          num_unique_segments,
                                                          0 if make_second_fit else backstep)

            # тут ещё раз монтекарлим, только более точная подгонка
            fitted_results_list = []

            if make_second_fit:
                for best_segments in best_segments_list:
                    angle_range_mult = 0.5
                    segment_len = best_segments[0][0].end_vs - best_segments[0][0].start_vs
                    mean_shift_delta = calculate_average_shift_difference(best_segments[0])
                    angle_range_precise = math.degrees(math.atan(mean_shift_delta / segment_len)) * angle_range_mult
                    
                    # Use appropriate method for second fit
                    if method == "optimizer":
                        temp_results = optimizer_fit(
                            best_segments[7],
                            type_well,
                            interpretation_start_idx,
                            best_segments[0],
                            angle_range_precise,
                            segm_counts_reg,
                            int(num_iterations / 3),
                            pearson_power,
                            mse_power,
                            num_intervals_self_correlation,
                            sc_power
                        )
                    else:
                        temp_results = monte_carlo_fit(
                            best_segments[7],
                            type_well,
                            interpretation_start_idx,
                            best_segments[0],
                            angle_range_precise,
                            segm_counts_reg,
                            int(num_iterations / 3),
                            pearson_power,
                            mse_power,
                            num_intervals_self_correlation,
                            sc_power
                        )

                    extended_results = [(result[5],) + result for result in temp_results]
                    fitted_results_list.extend(extended_results)
                    # Упорядочиваем уточнённые результаты по corr по убыванию
                    sorted_results = sorted((r for r in fitted_results_list if r is not None), key=lambda x: x[1], reverse=True)
                    if len(sorted_results) == 0:
                        print('empty results on second fit!')

                best_segments_list = select_unique_candidates(sorted_results,
                                                              num_unique_segments,
                                                              backstep)

            big_segment_start_idx += step_size
            i += 1

            num_segments_completed += 1
            if num_segments_completed % pbar_devider == 0:
                pbar.update(1)

        best_segments_list = select_unique_candidates(sorted_results,
                                                      num_unique_segments)

        # тут нужно зафиксировать окончательные интерпретации из best_segments_list
        # для использования их на следующем прогоне с меньшим размером сегмента
        previous_best_segments_list = []
        for best_segments in best_segments_list:
            previous_best_segments_list.append(deepcopy(best_segments[0]))

    return best_segments_list[0][0]
