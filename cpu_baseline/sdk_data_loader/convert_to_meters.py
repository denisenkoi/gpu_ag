#!/usr/bin/env python3
import os
import json
import shutil
from pathlib import Path
from tqdm import tqdm

# Константа для конвертации футов в метры
FEET_TO_METERS = 0.3048


def feet_to_meters(value):
    """Конвертирует значение из футов в метры"""
    if value is None:
        return None
    return float(value) * FEET_TO_METERS


def convert_json_file(input_file, output_file):
    """
    Конвертирует значения из футов в метры в JSON-файле и сохраняет результат.

    Args:
        input_file (str): Путь к исходному JSON-файлу
        output_file (str): Путь для сохранения конвертированного JSON-файла
    """
    # Загружаем JSON из файла
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Конвертируем log данные
    if "log" in data:
        for point in data["log"]:
            # Конвертируем расстояния
            if "md" in point:
                point["md"] = feet_to_meters(point["md"])
            if "tvd" in point:
                point["tvd"] = feet_to_meters(point["tvd"])
            if "measuredDepth" in point:
                point["measuredDepth"] = feet_to_meters(point["measuredDepth"])
            if "northSouth" in point:
                point["northSouth"] = feet_to_meters(point["northSouth"])
            if "eastWest" in point:
                point["eastWest"] = feet_to_meters(point["eastWest"])

            # Не конвертируем value, это не расстояние, а значение гамма-каротажа или другого лога
            # Не конвертируем угловые значения (azimutRad, inclinationRad, dogLegRad)

    # Конвертируем tops данные
    if "tops" in data:
        for top in data["tops"]:
            if "tvd" in top:
                top["tvd"] = feet_to_meters(top["tvd"])

    # Конвертируем startMd
    if "startMd" in data:
        data["startMd"] = feet_to_meters(data["startMd"])

    # Конвертируем startShift (если это расстояние)
    if "startShift" in data:
        data["startShift"] = feet_to_meters(data["startShift"])

    # Конвертируем shiftTVDToTypewell (если это расстояние)
    if "shiftTVDToTypewell" in data:
        data["shiftTVDToTypewell"] = feet_to_meters(data["shiftTVDToTypewell"])

    # НЕ конвертируем координаты wellhead - они уже в правильных единицах

    # Сохраняем конвертированный JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)


def main():
    # Директория с JSON-файлами
    input_dir = "../../InputVerticalTrackData"

    # Директория для сохранения конвертированных файлов
    output_dir = os.path.join(input_dir, "meters")

    # Создаем директорию для сохранения, если она не существует
    os.makedirs(output_dir, exist_ok=True)

    # Получаем список всех JSON-файлов в директории
    json_files = list(Path(input_dir).glob("*.json"))
    print(f"Найдено {len(json_files)} JSON-файлов")

    # Обрабатываем каждый файл
    for json_file in tqdm(json_files, desc="Конвертация файлов"):
        # Формируем путь к выходному файлу
        output_file = os.path.join(output_dir, os.path.basename(json_file))

        # Конвертируем файл
        convert_json_file(json_file, output_file)

    print(f"\nКонвертация завершена. Результаты сохранены в {output_dir}")


if __name__ == "__main__":
    main()