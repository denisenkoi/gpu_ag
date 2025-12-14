import os


def get_initial_params():
    pearson_power_list = [2]

    mse_power_list = [0.001, 1]
    mse_power_list = [0.05]

    segment_len_lists = [[300]]

    nums_intervals_self_correlation = [5, 10, 15, 20, 25]
    nums_intervals_self_correlation = [20]  # best values

    sc_power_list = [1.02, 1.06, 1.1, 1.2, 1.3, 1.5, 2]
    sc_power_list = [1.15]  # best values
    # sc_power_list = [1.2]

    wells_list = [
        'XBC Giddings Estate 2171H.json',
    ]
    wells_list = ['XBC Giddings Estate 2171H.json']
    # wells_list = []

    num_iterations = 2000

    # Максимальный угол отклонения от горизонтали
    angle_range = 10

    # Степень штрафа за сумму углов между сегментами
    angle_sum_power_list = [0.2]

    # Количество сегментов для одного прохода оптимизатора
    segments_count = 5

    # Минимальное значение корреляции Пирсона для алгоритмов оптимизации
    min_pearson_value = -1

    # Использовать накопительный расчет границ для оптимизации
    use_accumulative_bounds = True

    dir = '../AG_DATA_CURRENT_AG_nrom_gr'
    well_data_dir = 'InputVerticalTrackData'
    # Создание пути к директории
    path = f'{dir}/{well_data_dir}'

    file_names = os.listdir(path)
    well_names_all = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

    return (
        pearson_power_list,
        mse_power_list,
        segment_len_lists,
        nums_intervals_self_correlation,
        sc_power_list,
        num_iterations,
        wells_list,
        well_names_all,
        segments_count,
        min_pearson_value,
        use_accumulative_bounds,
        angle_range,  # добавляем новые параметры
        angle_sum_power_list
    )


def get_dir_name_for_grid(wells_calc_params) -> object:
    dir_name = f'auto_interpretations_{wells_calc_params["segments_len_list"][0]}  pears_pow {str(wells_calc_params["pearson_power"]).replace(".", "_")}'
    dir_name += f'  mse_pow {str(wells_calc_params["mse_power"]).replace(".", "_")}'
    dir_name += f'  num_SCI {str(wells_calc_params["num_intervals_sc"]).replace(".", "_")}'
    dir_name += f'  SCI_pow {str(wells_calc_params["sc_power"]).replace(".", "_")}'

    # Добавляем segments_count в имя директории, если оно было изменено
    if "segments_count" in wells_calc_params:
        dir_name += f'  segments_count {str(wells_calc_params["segments_count"]).replace(".", "_")}'

    dir_name += '/'

    return dir_name


def generate_combinations():
    (pearson_power_list,
     mse_power_list,
     segment_len_lists,
     nums_intervals_self_correlation,
     sc_power_list,
     num_iterations,
     wells_list,
     well_names_all,
     segments_count,
     min_pearson_value,
     use_accumulative_bounds,
     angle_range,
     angle_sum_power_list) = get_initial_params()

    grid_list = []

    start_from_well = 'Rio Rojo 27-30 Unit 1 #123.json'
    start_from_well = None

    if start_from_well is not None:
        action = False
    else:
        action = True

    print(f' Wells count: {len(wells_list)}')

    params_count = (len(mse_power_list)
                    * len(pearson_power_list)
                    * len(nums_intervals_self_correlation)
                    * len(sc_power_list))
    print(f'Params count: {params_count}')

    for mse_power in mse_power_list:
        for pearson_power in pearson_power_list:
            for num_intervals_sc in nums_intervals_self_correlation:
                for sc_power in sc_power_list:
                    for angle_sum_power in angle_sum_power_list:  # Добавляем перебор по новому параметру
                        for well_name in well_names_all:
                            for segments_len_list in segment_len_lists:
                                if well_name == start_from_well:
                                    action = True

                                if wells_list != [] and well_name not in wells_list:
                                    continue

                                if action == False:
                                    continue

                                grid_list.append({
                                    'mse_power': mse_power,
                                    'pearson_power': pearson_power,
                                    'num_intervals_sc': num_intervals_sc,
                                    'sc_power': sc_power,
                                    'well_name': well_name,
                                    'segments_len_list': segments_len_list,
                                    'num_iterations': num_iterations,
                                    'segments_count': segments_count,
                                    'min_pearson_value': min_pearson_value,
                                    'use_accumulative_bounds': use_accumulative_bounds,
                                    'angle_range': angle_range,  # Новые параметры
                                    'angle_sum_power': angle_sum_power
                                })
    return grid_list