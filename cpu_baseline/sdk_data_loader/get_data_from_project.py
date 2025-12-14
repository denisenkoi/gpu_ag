#!/usr/bin/env python3
import os
import json
import time
import math
import concurrent.futures
import numpy as np
from rogii_solo.client import SoloClient
from rogii_solo.calculations.enums import EMeasureUnits
from rogii_solo.calculations.trajectory import calculate_trajectory, interpolate_trajectory_point
from typing import Dict, Any, List, Optional
import pandas as pd
from tqdm import tqdm


def feet_to_meters(value):
    """Конвертирует значение из футов в метры"""
    if value is None or pd.isna(value):
        return None
    return float(value) * 0.3048


def process_well(well_data, project, output_dir):
    """
    Обработка отдельной скважины и сохранение результатов в JSON файл.

    Args:
        well_data (dict): Данные скважины
        project: Объект проекта Solo Cloud
        output_dir (str): Директория для сохранения файлов

    Returns:
        tuple: (успех/ошибка, имя скважины, сообщение)
    """
    wells = project.wells
    well_obj = wells.find_by_id(well_data['uuid'])
    if well_obj is None:
        return (False, well_data['name'], f"Не удалось получить объект скважины с UUID: {well_data['uuid']}")

    well_name = well_data['name']
    print(f"Начинаем обработку скважины: {well_name}")

    # Определяем, нужна ли конвертация из футов в метры
    is_foot_project = project.measure_unit == "FOOT"
    is_foot_project = False
    measure_units = EMeasureUnits.FOOT if is_foot_project else EMeasureUnits.METER

    # Формирование результирующего JSON
    result_data = {}

    # Получение лога "GR"
    gr_log = None
    for log in well_obj.logs.to_dict():
        if log['name'] == "GR":
            gr_log = well_obj.logs.find_by_id(log['uuid'])
            break

    # Если лог GR найден, получаем его точки
    if gr_log:
        log_points = gr_log.points.to_df()
        print(f"Найден лог GR, количество точек: {len(log_points)}")

        # Получение данных траектории скважины
        trajectory = well_obj.trajectory
        trajectory_df = trajectory.to_df()
        print(f"Получена траектория, количество точек: {len(trajectory_df)}")

        # Подготавливаем данные инклинометрии для расчета
        raw_trajectory = []
        for _, row in trajectory_df.iterrows():
            station = {
                'md': float(row['md']),
                'incl': math.radians(float(row['incl'])),
                'azim': math.radians(float(row['azim']))
            }
            raw_trajectory.append(station)

        # Данные о скважине для расчетов
        well_meta = {
            'convergence': 0.0,  # Настройте, если известно
            'kb': well_obj.kb,
            'xsrf': well_obj.xsrf,
            'ysrf': well_obj.ysrf,
            'tie_in_tvd': well_obj.tie_in_tvd,
            'tie_in_ns': well_obj.tie_in_ns,
            'tie_in_ew': well_obj.tie_in_ew,
            'azimuth': math.radians(well_obj.azimuth) if well_obj.azimuth is not None else 0.0
        }

        print(f"Расчет полной траектории...")
        # Рассчитываем полную траекторию
        calculated_trajectory = calculate_trajectory(
            raw_trajectory=raw_trajectory,
            well=well_meta,
            measure_units=measure_units
        )

        if not calculated_trajectory:
            print(f"Ошибка при расчете траектории для скважины {well_name}")
            return (False, well_name, "Ошибка при расчете траектории")

        print(f"Траектория рассчитана, количество точек: {len(calculated_trajectory)}")

        # Формирование массива точек лога
        log_data = []
        print(f"Обработка точек лога...")

        # Преобразуем точки траектории в более удобный формат словарей
        trajectory_points = []
        for point in calculated_trajectory:
            trajectory_points.append({
                'md': point['md'],
                'incl': point['incl'],
                'azim': point['azim'],
                'tvd': point['tvd'],
                'ns': point['ns'],
                'ew': point['ew'],
                'x': point['x'],
                'y': point['y'],
                'vs': point['vs'],
                'dls': point['dls'],
                'dog_leg': point['dog_leg']
            })

        # Для каждой точки лога находим соответствующие параметры траектории
        for _, point in log_points.iterrows():
            md = point['md']

            # Находим две ближайшие точки траектории для интерполяции
            left_point_idx = None
            right_point_idx = None

            for idx, traj_point in enumerate(trajectory_points):
                if traj_point['md'] <= md:
                    left_point_idx = idx
                if traj_point['md'] >= md and right_point_idx is None:
                    right_point_idx = idx

            # Пропускаем точки за пределами диапазона траектории
            if left_point_idx is None or right_point_idx is None:
                continue  # Пропускаем экстраполяцию

            # Если обе точки совпадают, берем значения этой точки
            if left_point_idx == right_point_idx:
                traj_point = trajectory_points[left_point_idx]
                tvd = traj_point['tvd']
                azimutRad = traj_point['azim']
                inclinationRad = traj_point['incl']
                dogLegRad = traj_point['dog_leg']
                northSouth = traj_point['ns']
                eastWest = traj_point['ew']
            else:
                # Используем встроенную функцию интерполяции
                left_point = trajectory_points[left_point_idx]
                right_point = trajectory_points[right_point_idx]

                # Интерполируем точку траектории
                interpolated_point = interpolate_trajectory_point(
                    left_point=left_point,
                    right_point=right_point,
                    md=md,
                    well=well_meta,
                    measure_units=measure_units
                )

                tvd = interpolated_point['tvd']
                azimutRad = interpolated_point['azim']
                inclinationRad = interpolated_point['incl']
                dogLegRad = interpolated_point['dog_leg']
                northSouth = interpolated_point['ns']
                eastWest = interpolated_point['ew']

            # Конвертируем значения с учетом единиц измерения
            md_val = float(md) if not pd.isna(md) else None
            tvd_val = float(tvd) if not pd.isna(tvd) else None
            value_val = float(point['value']) if not pd.isna(point['value']) else None

            # Если проект в футах, конвертируем расстояния в метры для JSON
            if is_foot_project:
                md_val = feet_to_meters(md_val)
                tvd_val = feet_to_meters(tvd_val)
                northSouth = feet_to_meters(northSouth)
                eastWest = feet_to_meters(eastWest)

            # Преобразуем все расчетные значения в числа или None для JSON-совместимости
            azimutRad_val = float(azimutRad) if azimutRad is not None and not pd.isna(azimutRad) else None
            inclinationRad_val = float(inclinationRad) if inclinationRad is not None and not pd.isna(
                inclinationRad) else None
            dogLegRad_val = float(dogLegRad) if dogLegRad is not None and not pd.isna(dogLegRad) else None
            northSouth_val = float(northSouth) if northSouth is not None and not pd.isna(northSouth) else None
            eastWest_val = float(eastWest) if eastWest is not None and not pd.isna(eastWest) else None

            # Формируем точку лога со всеми необходимыми параметрами
            log_point = {
                "md": md_val,
                "tvd": tvd_val,
                "value": value_val,
                "measuredDepth": md_val,  # Дублируем md для совместимости
                "azimutRad": azimutRad_val,
                "inclinationRad": inclinationRad_val,
                "dogLegRad": dogLegRad_val,
                "northSouth": northSouth_val,
                "eastWest": eastWest_val
            }

            log_data.append(log_point)

        result_data["log"] = log_data
    else:
        result_data["log"] = []

    # Получение маркеров (tops)
    tops_data = []
    # Проверка наличия топсетов в скважине
    if well_obj.topsets.to_dict():
        # Берем первый топсет (или starred_topset, если он есть)
        topset = well_obj.starred_topset if well_obj.starred_topset else well_obj.topsets.find_by_id(
            well_obj.topsets.to_dict()[0]['uuid'])

        if topset:
            tops = topset.tops.to_dict()
            # Получаем данные траектории (если еще не получены)
            if 'calculated_trajectory' not in locals():
                # Подготавливаем данные инклинометрии для расчета
                raw_trajectory = []
                for _, row in trajectory_df.iterrows():
                    station = {
                        'md': float(row['md']),
                        'incl': math.radians(float(row['incl'])),
                        'azim': math.radians(float(row['azim']))
                    }
                    raw_trajectory.append(station)

                # Данные о скважине для расчетов
                well_meta = {
                    'convergence': 0.0,  # Настройте, если известно
                    'kb': well_obj.kb,
                    'xsrf': well_obj.xsrf,
                    'ysrf': well_obj.ysrf,
                    'tie_in_tvd': well_obj.tie_in_tvd,
                    'tie_in_ns': well_obj.tie_in_ns,
                    'tie_in_ew': well_obj.tie_in_ew,
                    'azimuth': math.radians(well_obj.azimuth) if well_obj.azimuth is not None else 0.0
                }

                # Рассчитываем полную траекторию
                calculated_trajectory = calculate_trajectory(
                    raw_trajectory=raw_trajectory,
                    well=well_meta,
                    measure_units=measure_units
                )

                # Преобразуем точки траектории в более удобный формат словарей
                trajectory_points = []
                for point in calculated_trajectory:
                    trajectory_points.append({
                        'md': point['md'],
                        'incl': point['incl'],
                        'azim': point['azim'],
                        'tvd': point['tvd'],
                        'ns': point['ns'],
                        'ew': point['ew'],
                        'x': point['x'],
                        'y': point['y'],
                        'vs': point['vs'],
                        'dls': point['dls'],
                        'dog_leg': point['dog_leg']
                    })

            for top in tops:
                # Получаем MD маркера
                md = top['md']

                # Используем интерполяцию для определения TVD
                # Находим две ближайшие точки траектории для интерполяции
                left_point_idx = None
                right_point_idx = None

                for idx, traj_point in enumerate(trajectory_points):
                    if traj_point['md'] <= md:
                        left_point_idx = idx
                    if traj_point['md'] >= md and right_point_idx is None:
                        right_point_idx = idx

                # Если MD меньше минимального или больше максимального в траектории
                if left_point_idx is None or right_point_idx is None:
                    continue  # Пропускаем экстраполяцию

                # Если обе точки совпадают, берем значения этой точки
                if left_point_idx == right_point_idx:
                    tvd = trajectory_points[left_point_idx]['tvd']
                else:
                    # Используем встроенную функцию интерполяции
                    left_point = trajectory_points[left_point_idx]
                    right_point = trajectory_points[right_point_idx]

                    # Интерполируем точку траектории
                    interpolated_point = interpolate_trajectory_point(
                        left_point=left_point,
                        right_point=right_point,
                        md=md,
                        well=well_meta,
                        measure_units=measure_units
                    )

                    tvd = interpolated_point['tvd']

                # Преобразуем значения с учетом единиц измерения
                tvd_val = float(tvd) if not pd.isna(tvd) else None

                # Если проект в футах, конвертируем в метры
                if is_foot_project:
                    tvd_val = feet_to_meters(tvd_val)

                top_data = {
                    "tvd": tvd_val,
                    "uuid": top['uuid'],
                    "name": top['name']  # Добавляем имя маркера
                }
                tops_data.append(top_data)

    result_data["tops"] = tops_data

    # Добавление метаданных
    result_data["shiftTVDToTypewell"] = 0.0

    # Добавляем имя скважины в явном виде
    result_data["wellName"] = well_name
    result_data["wellUuid"] = well_data['uuid']

    # Добавляем координаты устья скважины
    x_val = float(well_meta.get('xsrf', 0)) if not pd.isna(well_meta.get('xsrf', 0)) else 0.0
    y_val = float(well_meta.get('ysrf', 0)) if not pd.isna(well_meta.get('ysrf', 0)) else 0.0
    z_val = float(well_meta.get('kb', 0)) if not pd.isna(well_meta.get('kb', 0)) else 0.0

    # Если проект в футах, конвертируем координаты в метры
    if is_foot_project:
        x_val = feet_to_meters(x_val)
        y_val = feet_to_meters(y_val)
        z_val = feet_to_meters(z_val)

    result_data["wellhead"] = {
        "x": x_val,
        "y": y_val,
        "z": z_val
    }

    # Определение startMd - берем из первой точки траектории, если она есть
    if not trajectory_df.empty:
        start_md = trajectory_df['md'].iloc[0]
        start_md_val = float(start_md) if not pd.isna(start_md) else 0.0

        # Если проект в футах, конвертируем в метры
        if is_foot_project:
            start_md_val = feet_to_meters(start_md_val)

        result_data["startMd"] = start_md_val
    else:
        result_data["startMd"] = 0.0

    result_data["startShift"] = 0.0

    # Сохранение JSON файла с пользовательским обработчиком для NaN и других проблемных значений
    file_path = os.path.join(output_dir, f"{well_name}.json")
    with open(file_path, 'w', encoding='utf-8') as f:
        # Используем собственный JSONEncoder для обработки проблемных значений
        class CustomJSONEncoder(json.JSONEncoder):
            def default(self, obj):
                if pd.isna(obj) or obj is float('nan') or obj is float('inf') or obj is float('-inf'):
                    return None
                return super().default(obj)

        json.dump(result_data, f, indent=4, cls=CustomJSONEncoder)

    return (True, well_name, f"Данные сохранены в файл {file_path}")


def export_wells_data(project_name, client_id, client_secret, output_dir="output", max_workers=5):
    """
    Обход всех скважин проекта и формирование JSON файлов с данными, используя
    многопоточную обработку.

    Args:
        project_name (str): Имя проекта
        client_id (str): Client ID для Solo API
        client_secret (str): Client Secret для Solo API
        output_dir (str): Директория для сохранения файлов
        max_workers (int): Максимальное количество параллельных потоков
    """
    # Создаем директорию для выходных файлов, если она не существует
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Подключение к Solo API
    start_time = time.time()
    solo_client = SoloClient(client_id=client_id, client_secret=client_secret)

    # Получение проекта по имени
    solo_client.set_project_by_name(project_name)
    project = solo_client.project
    print(f"Успешно подключились к проекту: {project.name}")

    # Получение списка всех скважин проекта
    wells_data = project.wells.to_dict()
    print(f"Найдено {len(wells_data)} скважин в проекте")

    print(wells_data)

    # Запуск параллельной обработки скважин
    print(f"Начинаем параллельную обработку в {max_workers} потоков")
    successful = 0
    failed = 0

    # Создаем список для отслеживания уже обработанных скважин
    processed_well_names = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Создаем список задач для каждой скважины
        futures = [executor.submit(process_well, well, project, output_dir) for well in wells_data]

        # Показываем прогресс выполнения
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Обработка скважин"):
            success, well_name, message = future.result()

            # Проверяем, не был ли этот well_name уже обработан
            # Это может произойти из-за дубликатов или проблем с параллельной обработкой
            if well_name in processed_well_names:
                print(f"⚠️ Скважина {well_name} уже была обработана ранее")
                continue

            processed_well_names.append(well_name)

            if success:
                successful += 1
                print(f"✓ Скважина {well_name}: {message}")
            else:
                failed += 1
                print(f"✗ Скважина {well_name}: {message}")

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"\nОбработка завершена за {elapsed_time:.2f} секунд.")
    print(f"Успешно обработано: {successful} скважин")
    print(f"Ошибки при обработке: {failed} скважин")
    print(f"Данные сохранены в директории '{output_dir}'")

def main():
    """Основная функция для запуска скрипта"""

    client = SoloClient(client_id='00001-51639-soX52-solopythonsdk', client_secret='gWY90e2Wv3KoO510icxq')

    # Получение данных для подключения
    client_id = '00001-51639-soX52-solopythonsdk'
    client_secret = 'gWY90e2Wv3KoO510icxq'
    project_name = "RND_398_Midland_Basin_Data_for_3D_GR"
    output_dir = "../../InputVerticalTrackData"
    max_workers = 5  # Фиксированное значение 5 потоков

    # Запуск экспорта
    export_wells_data(project_name, client_id, client_secret, output_dir, max_workers)


if __name__ == "__main__":
    main()