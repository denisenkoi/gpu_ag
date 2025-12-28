import numpy as np
from ag_rewards.ag_func_correlations import linear_interpolation
from copy import deepcopy
import math
import logging
from ag_objects.ag_obj_well import Well

logger = logging.getLogger(__name__)


class Segment:
    def __init__(self, data_source, well=None, start_idx=None, start_shift=None, end_idx=None, end_shift=None):
        """
        Универсальный конструктор для Segment

        Args:
            data_source: dict (JSON segment) или Well object (legacy)
            well: Well object (required for JSON initialization)
            start_idx, start_shift, end_idx, end_shift: legacy parameters
        """
        if isinstance(data_source, dict):  # JSON segment
            if well is None:
                raise ValueError("Well object required for JSON segment initialization")
            self.init_from_json(data_source, well)
        else:  # Legacy: first param is well object
            well = data_source  # В старом API первый параметр был Well
            self.init_from_well(well, start_idx, start_shift, end_idx, end_shift)

    def init_from_json(self, json_segment, well):
        """Инициализация из JSON сегмента"""
        # Ожидаемая структура JSON сегмента:
        # {"startMd": 1000, "startShift": 0.5, "endShift": 0.7, "endMd": 1100}

        if 'startMd' not in json_segment:
            raise ValueError("JSON segment must contain startMd")
        if 'endMd' not in json_segment:
            raise ValueError("JSON segment must contain endMd")

        start_md = json_segment['startMd']
        end_md = json_segment['endMd']
        start_shift = json_segment.get('startShift', 0.0)
        end_shift = json_segment.get('endShift', 0.0)

        # CRITICAL CHECK: Segment must be within well MD range
        # If segment starts after well ends - skip this segment
        assert start_md <= well.max_md, \
            f"CRITICAL: Segment startMd={start_md:.2f}m is beyond well range (max_md={well.max_md:.2f}m). " \
            f"Segment is 'right' of the well - cannot create!"

        # If segment ends before well starts - skip this segment
        assert end_md >= well.min_md, \
            f"CRITICAL: Segment endMd={end_md:.2f}m is before well range (min_md={well.min_md:.2f}m). " \
            f"Segment is 'left' of the well - cannot create!"

        # Находим индексы через well.md2idx()
        start_idx = well.md2idx(start_md)
        end_idx = well.md2idx(end_md)

        # Вызываем стандартную инициализацию
        self.init_from_well(well, start_idx, start_shift, end_idx, end_shift)

    def init_from_well(self, well, start_idx, start_shift, end_idx, end_shift):
        """Стандартная инициализация из Well объекта и параметров"""
        cond = (start_shift is not None
                and end_shift is not None
                and (type(start_idx) == int or type(start_idx) == np.int64))
        assert cond, f"Invalid args: start_idx={start_idx} ({type(start_idx)}), start_shift={start_shift}, end_shift={end_shift}"

        self.start_idx = start_idx
        self.start_shift = start_shift
        self.end_shift = end_shift
        self.end_idx = end_idx
        self.start_vs = well.vs_thl[start_idx]
        self.end_vs = well.vs_thl[end_idx]
        self.start_md = well.measured_depth[start_idx]
        self.end_md = well.measured_depth[end_idx]
        self.angle = math.degrees(math.atan((self.end_shift - self.start_shift) / (self.end_vs - self.start_vs)))
        self.calc_angle()

    def calc_angle(self):
        self.angle = math.degrees(math.atan((self.end_shift - self.start_shift) / (self.end_vs - self.start_vs)))

    def interpolate_shift(self, md, well: Well):
        """Функция линейной интерполяции"""
        vs = well.md2vs(md, well)
        # Линейная интерполяция
        return self.start_shift + (self.end_shift - self.start_shift) * (
                (vs - self.start_vs) / (self.end_vs - self.start_vs))

    def to_json(self):
        """Конвертация сегмента обратно в JSON"""
        return {
            'startMd': self.start_md,
            'endMd': self.end_md,
            'startShift': self.start_shift,
            'endShift': self.end_shift
        }


def create_segments_from_json(json_segments, well:Well, current_end_md=None, last_md=None):
    """
    Создание списка сегментов из JSON массива

    Args:
        json_segments: массив JSON сегментов
        well: Well object
        current_end_md: текущая MD для обрезки из эмулятора (deprecated, use last_md)
        last_md: MD для endMd последнего сегмента (обычно lateralWellLastMD)

    Returns:
        List[Segment]: список созданных сегментов
    """
    # Use last_md if provided, otherwise fall back to current_end_md for backwards compatibility
    end_md_for_last = last_md if last_md is not None else current_end_md

    segments = []
    for i, json_segment in enumerate(json_segments):
        # Определяем следующий сегмент для расчета endMd
        next_segment = json_segments[i + 1] if i + 1 < len(json_segments) else None
        # Создаем расширенный json_segment с endMd
        extended_json_segment = json_segment.copy()

        # Вычисляем endMd
        if next_segment and 'startMd' in next_segment:
            extended_json_segment['endMd'] = next_segment['startMd']
        elif end_md_for_last is not None:
            # Последний сегмент - используем переданную last_md
            extended_json_segment['endMd'] = end_md_for_last
        else:
            # Нет last_md - используем конец траектории скважины
            extended_json_segment['endMd'] = well.measured_depth[-1]

        start_md = extended_json_segment.get('startMd')
        end_md = extended_json_segment.get('endMd')

        # Skip segments completely outside well bounds
        if start_md is not None and start_md > well.max_md:
            logger.debug(f"Skipping segment {i}: startMd={start_md:.2f}m > well.max_md={well.max_md:.2f}m")
            continue
        if end_md is not None and end_md < well.min_md:
            logger.debug(f"Skipping segment {i}: endMd={end_md:.2f}m < well.min_md={well.min_md:.2f}m")
            continue

        # Clamp segment to well bounds
        if start_md is not None and start_md < well.min_md:
            extended_json_segment['startMd'] = well.min_md
        if end_md is not None and end_md > well.max_md:
            extended_json_segment['endMd'] = well.max_md

        # Создаем сегмент
        segment = Segment(extended_json_segment, well=well)

        if segment.start_idx >= segment.end_idx:
            print(f"ERROR in MD: {well.measured_depth[segment.start_idx]}")
            print(f"error in {i} SEGMENT.  segment.start_idx({segment.start_idx}) MUST be < then segment.end_idx({segment.end_idx})")
            raise ValueError('segment.start_idx == segment.end_idx!!!!!!!!!!!!!!')
        segments.append(segment)

    return segments


def segments_to_json(segments):
    """
    Конвертация списка сегментов в JSON массив

    Args:
        segments: List[Segment]

    Returns:
        list: JSON массив сегментов
    """
    return [segment.to_json() for segment in segments]


def create_segments(well,
                    segments_count,
                    segment_len,
                    start_idx,
                    start_shift,
                    basic_segments=None):
    if len(well.measured_depth) <= start_idx + segment_len:
        print('len(well.md) <= start_idx + segment_len')
    assert (len(well.measured_depth) > start_idx + segment_len)
    # Создание нового списка сегментов
    new_segments = []
    # Вычисление новой длины сегмента
    total_length = segments_count * segment_len
    new_segment_len = total_length // segments_count

    for i in range(segments_count):
        # Вычисление индексов и сдвигов для каждого нового сегмента
        segment_start_idx = start_idx + i * new_segment_len
        segment_end_idx = segment_start_idx + new_segment_len
        if segment_start_idx >= len(well.measured_depth) - 1:
            break
        if segment_end_idx >= len(well.measured_depth):
            segment_end_idx = len(well.measured_depth) - 1
        if basic_segments:
            # если есть интерпретированный сегмент, то получаем из него стартовый и конечные шифты
            segment_start_shift = start_shift if i == 0 else get_shift_by_idx(basic_segments, segment_start_idx)
            segment_end_shift = get_shift_by_idx(basic_segments, segment_end_idx)
        else:
            # Если сегменты не предоставлены, используется начальный сдвиг
            segment_start_shift = start_shift
            segment_end_shift = start_shift

        if segment_end_idx >= len(well.measured_depth) or segment_start_idx >= len(well.measured_depth):
            print('idx of segment over well_len')

        # Создание и добавление нового сегмента
        segment = Segment(well, start_idx=segment_start_idx, start_shift=segment_start_shift,
                          end_idx=segment_end_idx, end_shift=segment_end_shift)
        new_segments.append(segment)

    if new_segments[0].start_shift != start_shift:
        print('govno0')
    return new_segments


def create_empty_interpretation(well,
                                segment_len,
                                segments_count,
                                interpretation_start_idx,
                                last_segment_len,
                                start_shift):
    segments_count = math.floor((len(well.measured_depth) - interpretation_start_idx - last_segment_len) / segment_len)

    last_segment_start_point = len(well.measured_depth) - last_segment_len

    all_segments = create_segments(well=well,
                                   segments_count=segments_count,
                                   segment_len=segment_len,
                                   start_idx=interpretation_start_idx,
                                   start_shift=start_shift)


def calculate_segments_difference(segments1, segments2):
    # Рассчитываем среднеквадратичное отклонение между end_shift двух наборов сегментов
    differences = [(s1.end_shift - s2.end_shift) ** 2 for s1, s2 in zip(segments1, segments2)]
    return np.sqrt(np.mean(differences))


def calculate_uniqueness_threshold(candidates, best_candidate_segments, num_elements=10):
    # Рассчитываем среднеквадратичное отклонение для заданного числа элементов от лучшего кандидата
    rms_differences = [
        calculate_segments_difference(best_candidate_segments, segments)
        for _, _, _, _, _, _, segments, _ in candidates[1:num_elements + 1]
        # Пропускаем лучший элемент, сравниваем со следующими
    ]
    return np.mean(rms_differences) if rms_differences else 0.1


def calculate_average_shift_difference(segments):
    total_difference = 0
    count = 0

    for segment in segments:
        total_difference += abs(segment.end_shift - segment.start_shift)
        count += 1

    if count > 0:
        return total_difference / count
    else:
        return 0


def get_shift_by_idx(segments, idx):
    for segment in segments:
        if segment.start_idx <= idx <= segment.end_idx:
            fraction = (idx - segment.start_idx) / (segment.end_idx - segment.start_idx)

            # Линейная интерполяция между начальным и конечным шифтами
            interpolated_shift = segment.start_shift + fraction * (segment.end_shift - segment.start_shift)
            return interpolated_shift

    return None


def check_segments(segments):
    for segm_idx, segm in enumerate(segments):
        if segm_idx < len(segments) - 1 and abs(segm.end_shift - segments[segm_idx + 1].start_shift) > 1E-10:
            print('Razlom!')


def denormalize_segments(segments, well: Well):
    denorm_segments = deepcopy(segments)
    for segment in denorm_segments:
        segment.start_vs = segment.start_vs * well.md_range + well.min_vs
        segment.end_vs = segment.end_vs * well.md_range + well.min_vs

        segment.start_md = segment.start_md * well.md_range + well.min_md
        segment.end_md = segment.end_md * well.md_range + well.min_md

        segment.start_shift *= well.md_range
        segment.end_shift *= well.md_range
    return denorm_segments


def normalize_segments(segments, range_md):
    denorm_segments = deepcopy(segments)
    for segment in denorm_segments:
        segment.start_vs /= range_md
        segment.end_vs /= range_md

        segment.start_shift /= range_md
        segment.end_shift /= range_md
    return denorm_segments


# Функция для поиска сегмента
def find_segment_with_md(md, segments):
    for segment in segments:
        if segment.start_md <= md <= segment.end_md:
            return segment
    return None


def cut_manual_interpretation_part(
        well: Well,
        manl_interp,
        cut_idx):
    cuted_interpretation = []
    for cur_segm in manl_interp:
        if cur_segm.end_idx < cut_idx:
            cuted_interpretation.append(cur_segm)
        else:
            last_segn_end_shift = linear_interpolation(
                well.vs_thl[cur_segm.start_idx],
                cur_segm.start_shift,
                well.vs_thl[cur_segm.end_idx],
                cur_segm.end_shift,
                well.vs_thl[cut_idx])

            new_segment = Segment(well, start_idx=cur_segm.start_idx, start_shift=cur_segm.start_shift,
                                  end_idx=cut_idx, end_shift=last_segn_end_shift)

            cuted_interpretation.append(new_segment)
            break
    return cuted_interpretation


def get_shift_by_md(segments, md):
    """Get interpolated shift by measured depth"""
    for segment in segments:
        if segment.start_md <= md <= segment.end_md:
            if segment.start_md == segment.end_md:
                return segment.start_shift

            ratio = (md - segment.start_md) / (segment.end_md - segment.start_md)
            return segment.start_shift + ratio * (segment.end_shift - segment.start_shift)
    return None


def trim_segments_to_range(segments, start_md, end_md, well):
    """Trim segments to MD range with shift interpolation at boundaries"""
    if not segments or start_md >= end_md:
        return []

    trimmed_segments = []

    for segment in segments:
        # Skip segments completely outside range
        if segment.end_md <= start_md or segment.start_md >= end_md:
            continue

        # Create trimmed segment copy
        trimmed_segment = deepcopy(segment)

        # Trim start boundary
        if segment.start_md < start_md < segment.end_md:
            # Interpolate shift at start_md
            interpolated_start_shift = segment.interpolate_shift(start_md, well)

            # Update segment boundaries
            trimmed_segment.start_md = start_md
            trimmed_segment.start_shift = interpolated_start_shift
            # Update start_idx and start_vs accordingly
            trimmed_segment.start_idx = well.md2idx(start_md)
            trimmed_segment.start_vs = well.md2vs(start_md, well)

        # Trim end boundary
        if segment.start_md < end_md < segment.end_md:
            # Interpolate shift at end_md
            interpolated_end_shift = segment.interpolate_shift(end_md, well)

            # Update segment boundaries
            trimmed_segment.end_md = end_md
            trimmed_segment.end_shift = interpolated_end_shift
            # Update end_idx and end_vs accordingly
            trimmed_segment.end_idx = well.md2idx(end_md)
            trimmed_segment.end_vs = well.md2vs(end_md, well)

        # Recalculate angle after trimming
        trimmed_segment.calc_angle()

        trimmed_segments.append(trimmed_segment)

    return trimmed_segments


def trim_segments_to_tvt_range(segments, min_tvt, max_tvt, well):
    """
    Trim segments to TVT range with shift interpolation at boundaries

    Args:
        segments: List of segments to trim
        min_tvt: Minimum TVT value (inclusive)
        max_tvt: Maximum TVT value (inclusive)
        well: Well object with TVT values calculated

    Returns:
        List of trimmed segments within TVT range
    """
    if not segments or min_tvt >= max_tvt:
        return []

    trimmed_segments = []

    for segment in segments:
        # Get TVT values for segment boundaries
        start_tvt = well.tvt[segment.start_idx]
        end_tvt = well.tvt[segment.end_idx]

        # Check for NaN values
        if np.isnan(start_tvt) or np.isnan(end_tvt):
            continue

        # Skip segments completely outside range
        if end_tvt < min_tvt or start_tvt > max_tvt:
            continue

        # Create trimmed segment copy
        trimmed_segment = deepcopy(segment)

        # Trim start boundary
        if start_tvt < min_tvt:
            # Find index where TVT >= min_tvt
            for idx in range(segment.start_idx, segment.end_idx + 1):
                if not np.isnan(well.tvt[idx]) and well.tvt[idx] >= min_tvt:
                    # Interpolate shift at this point
                    interpolated_shift = segment.interpolate_shift(well.measured_depth[idx], well)

                    # Update segment boundaries
                    trimmed_segment.start_idx = idx
                    trimmed_segment.start_shift = interpolated_shift
                    trimmed_segment.start_md = well.measured_depth[idx]
                    trimmed_segment.start_vs = well.vs_thl[idx]
                    break
            else:
                # No valid point found within range
                continue

        # Trim end boundary
        if end_tvt > max_tvt:
            # Find last index where TVT <= max_tvt
            for idx in range(segment.end_idx, segment.start_idx - 1, -1):
                if not np.isnan(well.tvt[idx]) and well.tvt[idx] <= max_tvt:
                    # Interpolate shift at this point
                    interpolated_shift = segment.interpolate_shift(well.measured_depth[idx], well)

                    # Update segment boundaries
                    trimmed_segment.end_idx = idx
                    trimmed_segment.end_shift = interpolated_shift
                    trimmed_segment.end_md = well.measured_depth[idx]
                    trimmed_segment.end_vs = well.vs_thl[idx]
                    break
            else:
                # No valid point found within range
                continue

        # Check that trimmed segment is still valid
        if trimmed_segment.start_idx >= trimmed_segment.end_idx:
            continue

        # Recalculate angle after trimming
        trimmed_segment.calc_angle()

        trimmed_segments.append(trimmed_segment)

    return trimmed_segments