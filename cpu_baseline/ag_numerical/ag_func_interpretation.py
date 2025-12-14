import math
from tqdm import tqdm
from ag_objects.ag_obj_well import Well
from ag_objects.ag_obj_interpretation import create_segments, calculate_segments_difference,\
    calculate_uniqueness_threshold, check_segments
from ag_rewards.ag_func_correlations import calculate_correlation
import numpy as np


def select_unique_candidates(candidates,
                             num_unique_segments,
                             backstep=0):
    if not candidates:
        return []

    # Сначала добавляем лучшего кандидата
    unique_candidates = [(
        candidates[0][0] if backstep == 0 else candidates[0][0][:-backstep],
        candidates[0][1],
        candidates[0][2],
        candidates[0][3],
        candidates[0][4],
        candidates[0][5],
        candidates[0][6],
        candidates[0][7]
    )]

    # Вычисляем порог уникальности
    uniqueness_threshold = calculate_uniqueness_threshold(candidates, candidates[0][6])

    # Проходимся по оставшимся кандидатам и выбираем уникальные
    for all_segments, current_corr, self_correlation, pearson, mse, num_points, current_segments, current_well in candidates[1:]:  # Пропускаем первый элемент, так как он уже добавлен
        is_unique = all(
            calculate_segments_difference(current_segments, existing_segments) >= uniqueness_threshold
            for _, _, _, _, _, _, existing_segments, _ in unique_candidates
        )

        if is_unique:
            unique_candidates.append(
                (all_segments if backstep == 0 else all_segments[:-backstep],
                 current_corr,
                 self_correlation,
                 pearson,
                 mse,
                 num_points,
                 current_segments,
                 current_well))

        if len(unique_candidates) >= num_unique_segments:
            break

    for all_segments in unique_candidates:
        check_segments(all_segments[0])

    return unique_candidates


def fit_segment(
        segments,
        segment_idx,
        well:Well,
        type_well,
        interpretation_start_idx,
        angle_range,
        num_steps = 50):

    segment_len = segments[segment_idx].end_vs - segments[segment_idx].start_vs
    vertical_range = (segment_len * well.horizontal_well_step * math.tan(math.radians(angle_range)))

    segments[segment_idx].end_shift -= vertical_range # устанавливаем правый шифт сегмента, инкапсуляция рыдает
    if len(segments) > segment_idx + 1:
        segments[segment_idx + 1].start_shift -= vertical_range # устанавливаем левый шифт сегмента справа от текущего

    step = 2 * vertical_range / num_steps

    best_corr = 0
    mean_self_correlation_intersect = 1
    num_try_with_intersection = 0
    while segments[segment_idx].end_shift <= vertical_range: # цикл по разным шифтам

        well.calc_horizontal_projection(type_well,
                                        segments,
                                        [segment_idx])

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
            interpretation_start_idx,
            # interpretation_start_idx считает пирсона и L2 по сегменту и прошлым сегментам, второе только по сегменту
            segments[segment_idx].start_idx,
            segments[segment_idx].end_idx,
            mean_self_correlation_intersect,
            num_try_with_intersection,
            best_corr
        )
        segments[segment_idx].end_shift += step
        if len(segments) > segment_idx + 1:
            segments[segment_idx + 1].start_shift += step

        if corr > best_corr:
            best_corr = corr
            best_shift = segments[segment_idx].end_shift

    segments[segment_idx].end_shift = best_shift
    if len(segments) > segment_idx + 1:
        segments[segment_idx + 1].start_shift = best_shift
        segments[segment_idx + 1].start_shift = best_shift



def make_sequential_interpretation(
        well:Well,
        type_well,
        well_data_manual_interpretation,
        interpretation_start_idx,
        start_shift,
        num_iterations):

    num_unique_segments = 3
    # список должен быть упорядочен по убыванию
    segment_len_list = [300]
    segments_count = 2
    angle_range = 10
    segm_counts_reg = [2, 4, 6, 10, 15, 20]
    backstep = 0
    typewell_over_range = 0.5
    last_segment_len = 50
    num_steps = 100

    interpretation_end_idx = len(well.measured_depth) - 1

    # проверка, что начало интерпретации находится в диапазоне опорной скважины и корректировка первой точки при необходимости
    if well.true_vertical_depth[interpretation_start_idx] - start_shift > np.max(type_well.true_vertical_depth):
        if abs(well.true_vertical_depth[interpretation_start_idx] - start_shift - np.max(type_well.true_vertical_depth)) * well.md_range > typewell_over_range:
            print('Start_tvt depth over typewell depth range')
        else:
            start_shift = well.true_vertical_depth[interpretation_start_idx] - np.max(type_well.true_vertical_depth)

    if well.true_vertical_depth[interpretation_start_idx] - start_shift < np.min(type_well.true_vertical_depth):
        if abs(well.true_vertical_depth[interpretation_start_idx] - start_shift - np.min(type_well.true_vertical_depth)) * well.md_range > typewell_over_range:
            print('Start_tvt depth under typewell depth range')
        else:
            print('Start_tvt depth under typewell depth range. small ... corrected')
            start_shift = -(well.true_vertical_depth[interpretation_start_idx] - np.min(type_well.true_vertical_depth))

    segments_count = math.floor(len(well.measured_depth) / segment_len_list[0]) if segments_count == 0 else segments_count

    print(f'segments count: {segments_count}, well len: {len(well.measured_depth)}, segm_len in meters: {segment_len_list[0] * well.horizontal_well_step}, segm_len in points: {segment_len_list[0]}, big_segm_len: {segments_count * segment_len_list[0]}, interpretation_len: {len(well.measured_depth) - interpretation_start_idx}')

    segment_len = segment_len_list[0]

    segments_count = math.floor((len(well.measured_depth) - interpretation_start_idx - last_segment_len) / segment_len)
    segments = create_segments(well=well,
                               segment_len=segment_len,
                               segments_count=segments_count,
                               start_idx=interpretation_start_idx,
                               start_shift=start_shift)

    for segment_idx, segment in enumerate(tqdm(segments)):
        fit_segment(segments,
                    segment_idx,
                    well,
                    type_well,
                    interpretation_start_idx,
                    angle_range,
                    num_steps)

    all_segments = segments


    return \
        well, \
        all_segments,\
        interpretation_start_idx,\
        type_well,\
        segm_counts_reg,\
        well_data_manual_interpretation,\
        angle_range