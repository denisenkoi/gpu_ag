"""
Wells State Manager - управление состоянием обработки скважин
Работает с файлом processed_wells.json для отслеживания прогресса
"""

import json
import time
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


def get_timestamps() -> tuple[float, str]:
    """Получить timestamp и читаемое время"""
    now = time.time()
    readable = datetime.fromtimestamp(now).strftime('%Y-%m-%d %H:%M:%S')
    return now, readable


def load_processed_wells_state(results_dir: str) -> Dict[str, Any]:
    """Загрузить состояние из processed_wells.json"""
    state_file = Path(results_dir) / 'processed_wells.json'
    
    if not state_file.exists():
        logger.info("State file not found, creating new state")
        return {"wells": []}
    
    try:
        with open(state_file, 'r', encoding='utf-8') as f:
            state = json.load(f)
        logger.info(f"Loaded state with {len(state.get('wells', []))} wells")
        return state
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.warning(f"Failed to load state file: {e}, creating new state")
        return {"wells": []}


def save_processed_wells_state(state: Dict[str, Any], results_dir: str) -> None:
    """Сохранить состояние в processed_wells.json"""
    state_file = Path(results_dir) / 'processed_wells.json'
    
    try:
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        logger.debug(f"Saved state with {len(state.get('wells', []))} wells")
    except Exception as e:
        logger.error(f"Failed to save state file: {e}")


def scan_results_directory(results_dir: str) -> List[str]:
    """Сканировать директорию и найти все *.json файлы (кроме служебных)"""
    results_path = Path(results_dir)
    if not results_path.exists():
        logger.warning(f"Results directory not found: {results_dir}")
        return []
    
    # Файлы, которые нужно игнорировать
    ignore_files = {
        'comparison_report.json',
        'processed_wells.json'
    }
    
    json_files = []
    for file_path in results_path.glob('*.json'):
        if file_path.name not in ignore_files:
            json_files.append(file_path.name)
    
    # Также игнорируем папку intermediate
    intermediate_dir = results_path / 'intermediate'
    if intermediate_dir.exists():
        logger.debug("Ignoring intermediate directory")
    
    logger.info(f"Found {len(json_files)} JSON files in {results_dir}")
    return sorted(json_files)  # Сортируем для предсказуемого порядка


def extract_well_name_from_filename(filename: str) -> str:
    """Извлечь имя скважины из имени файла"""
    # Формат: Well1000~EGFDL_20250827_143059.json
    # Извлекаем все до первого '_'
    if '_' in filename:
        return filename.split('_')[0]
    else:
        # Если нет underscore, берем имя без расширения
        return filename.replace('.json', '')


def sync_wells_state(current_files: List[str], existing_state: Dict[str, Any]) -> Dict[str, Any]:
    """Синхронизировать состояние с файлами в директории"""
    if "wells" not in existing_state:
        existing_state["wells"] = []
    
    existing_wells = {well['filename']: well for well in existing_state['wells']}
    updated_wells = []
    
    # Добавляем новые файлы как pending
    for filename in current_files:
        if filename in existing_wells:
            # Файл уже есть в состоянии - оставляем как есть
            updated_wells.append(existing_wells[filename])
        else:
            # Новый файл - добавляем как pending
            well_name = extract_well_name_from_filename(filename)
            new_well = {
                "filename": filename,
                "well_name": well_name,
                "status": "pending",
                "started_at_timestamp": None,
                "started_at_readable": None,
                "completed_at_timestamp": None,
                "completed_at_readable": None,
                "error_message": None,
                "error_traceback": None
            }
            updated_wells.append(new_well)
            logger.info(f"Added new well to state: {filename}")
    
    # Удаляем из состояния файлы, которых больше нет в директории
    removed_count = len(existing_state['wells']) - len(updated_wells)
    if removed_count > 0:
        logger.info(f"Removed {removed_count} missing files from state")
    
    existing_state['wells'] = updated_wells
    return existing_state


def check_timeouts(state: Dict[str, Any], timeout_minutes: int = 20) -> None:
    """Проверить зависшие in_progress задачи и перевести их в pending"""
    if "wells" not in state:
        return
    
    current_time = time.time()
    timeout_seconds = timeout_minutes * 60
    timeout_count = 0
    
    for well in state['wells']:
        if (well['status'] == 'in_progress' and 
            well['started_at_timestamp'] is not None):
            
            elapsed = current_time - well['started_at_timestamp']
            if elapsed > timeout_seconds:
                logger.warning(f"Well {well['filename']} timed out after {elapsed/60:.1f} minutes, resetting to pending")
                well['status'] = 'pending'
                well['started_at_timestamp'] = None
                well['started_at_readable'] = None
                timeout_count += 1
    
    if timeout_count > 0:
        logger.info(f"Reset {timeout_count} timed out wells to pending")


def get_next_pending_well(state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Найти первую скважину со статусом pending"""
    if "wells" not in state:
        return None
    
    for well in state['wells']:
        if well['status'] == 'pending':
            return well
    
    return None


def update_well_status(state: Dict[str, Any], filename: str, status: str, 
                      error_msg: Optional[str] = None, 
                      error_tb: Optional[str] = None) -> None:
    """Обновить статус скважины с временными метками"""
    if "wells" not in state:
        return
    
    timestamp, readable = get_timestamps()
    
    for well in state['wells']:
        if well['filename'] == filename:
            well['status'] = status
            
            if status == 'in_progress':
                well['started_at_timestamp'] = timestamp
                well['started_at_readable'] = readable
                well['completed_at_timestamp'] = None
                well['completed_at_readable'] = None
                well['error_message'] = None
                well['error_traceback'] = None
                
            elif status == 'completed':
                well['completed_at_timestamp'] = timestamp
                well['completed_at_readable'] = readable
                well['error_message'] = None
                well['error_traceback'] = None
                
            elif status == 'error':
                well['completed_at_timestamp'] = timestamp
                well['completed_at_readable'] = readable
                well['error_message'] = error_msg
                well['error_traceback'] = error_tb
                
            logger.info(f"Updated well {filename} to status: {status}")
            return
    
    logger.warning(f"Well {filename} not found in state for status update")


def get_state_summary(state: Dict[str, Any]) -> Dict[str, int]:
    """Получить сводку по статусам скважин"""
    if "wells" not in state:
        return {}
    
    summary = {}
    for well in state['wells']:
        status = well['status']
        summary[status] = summary.get(status, 0) + 1
    
    return summary


def print_state_summary(results_dir: str) -> None:
    """Вывести сводку состояния в лог"""
    state = load_processed_wells_state(results_dir)
    summary = get_state_summary(state)
    
    if summary:
        summary_str = ", ".join([f"{status}: {count}" for status, count in summary.items()])
        logger.info(f"Wells summary: {summary_str}")
    else:
        logger.info("No wells in state")