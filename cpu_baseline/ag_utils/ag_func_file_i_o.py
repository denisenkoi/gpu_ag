import pandas as pd
from ag_objects.ag_obj_well import Well
from ag_objects.ag_obj_typewell import TypeWell
from ag_objects.ag_obj_interpretation import Segment
from copy import deepcopy
import json

def save_well_data_to_csv(well_data, filename):
    df_to_save = pd.DataFrame({
        'MD': well_data.measured_depth,
        'VS': well_data.vs_thl,
        'Depth': well_data.true_vertical_depth,
        'Curve': well_data.value,
        'TVT': well_data.tvt,
        'Synt_Curve': well_data.synt_curve
    })
    df_to_save.to_csv(filename, index=False)


def save_segments_to_csv(segments, filename):
    segments_data = []

    for segment in segments:
        segment_dict = {
            'Start_Idx': segment.start_idx,
            'Start_Shift': segment.start_shift,
            'End_Idx': segment.end_idx,
            'End_Shift': segment.end_shift,
            'Start_VS': segment.start_vs,
            'End_VS': segment.end_vs
        }
        segments_data.append(segment_dict)

    df_segments = pd.DataFrame(segments_data)
    df_segments.to_csv(filename, index=False)


def load_manual_interpretation():
    interpretation = pd.read_csv('interpretation.csv', delimiter='\t')
    return interpretation[['MD', 'TVT']]


def load_data():
    df_well = pd.read_csv('hz_well_geometry_logs.csv')
    df_typewell_data = pd.read_csv('VertWellLog.csv')

    # print(df_well.columns)
    # print(df_typewell_data.columns)

    max_value = max(df_well['Curve'].max(), df_typewell_data['Curve'].max())
    min_value = min(df_well['Curve'].min(), df_typewell_data['Curve'].min())

    well = Well(df_well)

    type_well = TypeWell(df_typewell_data,
                         max_value)

    well.normalize(max_value,
                   type_well.min_depth)

    type_well.normalize(max_value,
                        well.min_depth,
                        well.md_range)

    manual_interpretation = load_manual_interpretation()

    well_data_manual_interpretation = deepcopy(well)

    return well, type_well, well_data_manual_interpretation

def load_interpretation_json(
        well:Well,
        well_name,
        dir='InputInterpretationData',
        normalize=True,
        interpretation_for_check = None):

    with open(f'{dir}/{well_name}') as file:
        interpretation_data = json.load(file)['shifts']

    normaliz_multiplier = well.md_range if normalize else 1
    add_value = well.min_md if normalize else 0

    segments = []
    for i, interpretation_point in enumerate(interpretation_data):
        if i < len(interpretation_data) - 1:
            segment_end_idx = well.md2idx((interpretation_data[i + 1]['start_md'] - add_value) / normaliz_multiplier)
            end_shift = -interpretation_data[i + 1]['shift']  / normaliz_multiplier

            segment = Segment(
                well,
                well.md2idx((interpretation_point['start_md'] - add_value) / normaliz_multiplier),
                -interpretation_point['shift'] / normaliz_multiplier,
                segment_end_idx,
                end_shift)

            if segment.end_idx > segment.start_idx:
                segments.append(segment)
        else:
            if len(segments) == 0:
                print('Segments is empty')
            if segments[-1].end_idx < len(well.measured_depth) - 1:
                print('last MD of importing interpretation is before well END!')
                segment_end_idx = len(well.measured_depth) - 1
                end_shift = -interpretation_data[i]['shift'] / normaliz_multiplier
                segments.append(Segment(
                    well,
                    well.md2idx((interpretation_point['start_md'] - add_value) / normaliz_multiplier),
                    -interpretation_point['shift'] / normaliz_multiplier,
                    segment_end_idx,
                    end_shift))

    return segments


def save_segments_to_json(segments,
                          well,
                          file_path):
    # Создание массива shifts
    shifts = []
    for segment in segments:
        shift_dict = {
            'start_md': well.measured_depth[segment.start_idx],  # MD начала сегмента
            'shift': -segment.start_shift  # shift начала сегмента
        }
        shifts.append(shift_dict)

    shift_dict = {
        'start_md': well.measured_depth[segments[-1].end_idx],  # MD конца последнего сегмента
        'shift': -segments[-1].end_shift  # shift конца последнего сегмента
    }
    shifts.append(shift_dict)

    # Сохранение данных в JSON файл
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump({'shifts': shifts}, file, indent=4, ensure_ascii=False)

def load_json_data(well_name='Fasken 9-16-21 Unit 1 #171.json',
                   dir='AG_DATA_CURRENT_AG_nrom_gr',
                   normalize = True):
    well_data_dir = 'InputVerticalTrackData'

    with open(f'{dir}/{well_data_dir}/{well_name}') as file:
        wells_data = json.load(file)

    df_well = pd.DataFrame({
        'MD': [well_point['md'] for well_point in wells_data['log']],
        'VS': [well_point['md'] for well_point in wells_data['log']],  # Допускаем, что 'VS' = 'MD'
        'Depth': [well_point['tvd'] for well_point in wells_data['log']],
        'Curve': [well_point['value'] for well_point in wells_data['log']]
    })

    df_typewell_data = pd.DataFrame({
        'Depth': [well_point['tvd'] for well_point in wells_data['typeLog']],
        'Curve': [well_point['value'] for well_point in wells_data['typeLog']]
    })

    # print(df_well.columns)
    # print(df_typewell_data.columns)

    max_value = max(df_well['Curve'].max(), df_typewell_data['Curve'].max())
    min_value = min(df_well['Curve'].min(), df_typewell_data['Curve'].min())

    well = Well(df_well)

    type_well = TypeWell(df_typewell_data)

    if abs(wells_data['startShift']) > 0.1:
        print('Non zero start shift!')

    start_shift = (wells_data['startShift'])

    startMd = wells_data['startMd']

    if normalize:
        well.normalize(max_value,
                       type_well.min_depth)

        type_well.normalize(max_value,
                            well.min_depth,
                            well.md_range)


        start_shift = start_shift / well.md_range

        startMd = (startMd - well.min_md) / well.md_range

    manual_interpretation = load_interpretation_json(well,
                                                     well_name,
                                                     f'{dir}/InputInterpretationData',
                                                     normalize)

    well_with_manual_interpretation = deepcopy(well)
    well_with_manual_interpretation.calc_horizontal_projection(type_well, manual_interpretation)

    interpretation_start_idx = well.md2idx(startMd)

    return (well,
            type_well,
            well_with_manual_interpretation,
            manual_interpretation,
            interpretation_start_idx,
            start_shift)