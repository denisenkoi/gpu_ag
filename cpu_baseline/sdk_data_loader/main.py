import json
import os
from rogii_solo import SoloClient  # Предполагаем, что SDK установлен
from rogii_solo.calculations.trajectory import calculate_trajectory

# Ваш ключ доступа к API
ACCESS_KEY = "your_access_key_here"

# Инициализация клиента Solo API
client = SoloClient(client_id='00001-51639-soX52-solopythonsdk', client_secret='gWY90e2Wv3KoO510icxq')

# Папки для сохранения файлов
INTERPRETATION_DATA_DIR = "InputInterpretationData"
VERTICAL_TRACK_DATA_DIR = "InputVerticalTrackData"

# Создание папок, если они не существуют
os.makedirs(INTERPRETATION_DATA_DIR, exist_ok=True)
os.makedirs(VERTICAL_TRACK_DATA_DIR, exist_ok=True)

def get_projects():
    """
    Получение списка доступных проектов.
    """
    try:
        projects = client.projects
        return projects
    except Exception as e:
        print(f"Ошибка при получении проектов: {e}")
        return []

def get_wells(project_id):
    """
    Получение списка скважин для заданного проекта.
    """
    try:
        wells = client.wells.list(project_id=project_id)
        return wells
    except Exception as e:
        print(f"Ошибка при получении скважин для проекта {project_id}: {e}")
        return []

def get_well_data(well):
    """
    Получение данных о сдвигах и начальных глубинах для скважины.
    """
    trajectory_points = well.trajectory
    return trajectory_points

def get_wellhead_coordinates(well_id):
    """
    Получение координат устья скважины.
    """
    try:
        well = client.wells.get(well_id=well_id)
        return well.wellhead_coordinates  # Предполагаем, что это поле доступно
    except Exception as e:
        print(f"Ошибка при получении координат устья для скважины {well_id}: {e}")
        return None

def get_well_log_data(well_id):
    """
    Получение данных лога (md, tvd, value) для скважины.
    """
    try:
        log_data = client.wells.get_log(well_id=well_id)
        log = [{"md": point.md, "tvd": point.tvd, "value": point.value} for point in log_data.points]
        return log
    except Exception as e:
        print(f"Ошибка при получении данных лога для скважины {well_id}: {e}")
        return []

def create_interpretation_file(well_name, shifts, coordinates):
    """
    Создание JSON-файла с данными shifts и координатами устья в папке InputInterpretationData.
    """
    output = {
        "shifts": shifts,
        "wellhead_coordinates": coordinates
    }
    filename = os.path.join(INTERPRETATION_DATA_DIR, f"{well_name}.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)
    print(f"Создан файл: {filename}")

def create_vertical_track_file(well_name, log_data):
    """
    Создание JSON-файла с данными лога (md, tvd, value) в папке InputVerticalTrackData.
    """
    output = {
        "log": log_data
    }
    filename = os.path.join(VERTICAL_TRACK_DATA_DIR, f"{well_name}.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)
    print(f"Создан файл: {filename}")

def main():
    # Получаем список проектов
    projects = get_projects()

    for project in projects:
        print(project.virtual, project.name)

    for project in projects:
        # if project.name == "RND_398_Midland_Basin_Data_for_3D_GR":
        if project.name == "VP_for_AutoTops_Testing":
            #project_id = project.
            # Получаем список скважин для проекта
            wells = project.wells
            for well in wells:
                well_id = well.api
                well_name = well.name  # Предполагаем, что есть поле name
                print(project.name, well_name)
                well_name = well.name
                # Получаем данные shifts
                shifts = get_well_data(well)
                # Получаем координаты устья
                coordinates = get_wellhead_coordinates(well_id)
                # Получаем данные лога
                log_data = get_well_log_data(well_id)
                if shifts and coordinates and log_data:
                    # Создаем файл для shifts и координат в InputInterpretationData
                    create_interpretation_file(well_name, shifts, coordinates)
                    # Создаем файл для данных лога в InputVerticalTrackData
                    create_vertical_track_file(well_name, log_data)
            print(len(wells))

if __name__ == "__main__":
    main()