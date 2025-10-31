#!/usr/bin/env python3
"""
Unified Pipeline для обработки архитектурных планов
Последовательно запускает:
1. Hatching detection (hatching_mask.py)
2. Export objects to JSON (export_objects.py)
3. Align openings (visualize_polygons_align.py)
4. Visualize polygons with junction analysis (visualize_polygons_w.py)
"""

import os
import shutil
import sys
import time
import traceback
import subprocess
import argparse
from pathlib import Path

# Добавляем текущую директорию в путь для импортов
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def parse_arguments():
    """Парсит аргументы командной строки"""
    parser = argparse.ArgumentParser(description='Unified Pipeline для обработки архитектурных планов')
    parser.add_argument('input_file', help='Путь к входному JPG файлу')
    return parser.parse_args()

def get_base_filename(input_file):
    """Получает базовое имя файла без расширения"""
    base_name = Path(input_file).stem
    return base_name

def print_separator(title="", width=80):
    """Выводит разделитель с заголовком"""
    print("=" * width)
    if title:
        print(f"{title:^{width}}")
        print("=" * width)

def check_file_exists(file_path, description=""):
    """Проверяет существование файла"""
    if os.path.exists(file_path):
        print(f"  ✓ Файл найден: {file_path}")
        return True
    else:
        print(f"  ✗ Файл не найден: {file_path}")
        if description:
            print(f"    Описание: {description}")
        return False

def check_required_files(input_file):
    """Проверяет наличие всех необходимых файлов"""
    print_separator("ПРОВЕРКА НЕОБХОДИМЫХ ФАЙЛОВ")
    
    # Проверяем входной файл по полному пути
    if not os.path.exists(input_file):
        print(f"  ✗ Файл не найден: {input_file}")
        print(f"    Описание: Входное изображение плана")
        return False
    else:
        print(f"  ✓ Файл найден: {input_file}")
        print(f"    Описание: Входное изображение плана")
    
    # Проверяем скрипты в текущей директории
    required_files = [
        ("hatching_mask.py", "Скрипт обнаружения штриховки"),
        ("export_objects.py", "Скрипт экспорта объектов в JSON"),
        ("visualize_polygons_align.py", "Скрипт выравнивания проемов"),
        ("visualize_polygons_w.py", "Скрипт визуализации полигонов"),
        ("cubicasa_vectorize.py", "Модуль векторизации"),
        ("improved_junction_type_analyzer.py", "Модуль анализа junctions"),
        ("floortrans/__init__.py", "Модуль FloorTrans"),
    ]
    
    # Проверяем модель
    model_files = [
        ("model_best_val_loss_var.pkl", "Модель нейронной сети"),
    ]
    
    all_files_exist = True
    
    print("\nОсновные файлы:")
    for file_path, description in required_files:
        if not check_file_exists(file_path, description):
            all_files_exist = False
    
    print("\nФайлы модели:")
    for file_path, description in model_files:
        if not check_file_exists(file_path, description):
            print(f"  ⚠ Предупреждение: {file_path} не найден")
            print(f"    Описание: {description}")
            print(f"    Примечание: Экспорт объектов может не работать без модели")
    
    return all_files_exist

def run_script(script_name, description="", input_file=None):
    """Запускает Python скрипт и обрабатывает ошибки"""
    print_separator(f"ЗАПУСК: {description if description else script_name}")
    
    try:
        start_time = time.time()
        print(f"Запуск скрипта: {script_name}")
        print(f"Время начала: {time.strftime('%H:%M:%S')}")
        
        # Подготавливаем команду
        cmd = [sys.executable, script_name]
        if input_file:
            cmd.append(input_file)
        
        # Добавляем переменную окружения для кодировки
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        # Запускаем скрипт как subprocess для лучшей изоляции
        result = subprocess.run(
            cmd,
            cwd=current_dir,
            capture_output=True,
            text=True,
            timeout=300,  # 5 минут таймаут
            env=env
        )
  
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Логируем вывод
        if result.stdout:
            print("\nВывод:")
            print(result.stdout)
        if result.stderr:
            print("\nСообщения об ошибках:")
            print(result.stderr)
        
        if result.returncode == 0:
            print(f"\n✓ Успешно (время: {elapsed_time:.2f} с)")
            return True
        else:
            print(f"  ✗ Скрипт {script_name} завершился с ошибкой (код: {result.returncode})")
            if result.stderr:
                print("\nОшибки:")
                print(result.stderr)
            if result.stdout:
                print("\nВывод скрипта:")
                print(result.stdout)
            return False
            
    except subprocess.TimeoutExpired:
        print(f"  ✗ Скрипт {script_name} превысил время ожидания (5 минут)")
        return False
    except Exception as e:
        print(f"  ✗ Непредвиденная ошибка при запуске {script_name}: {e}")
        traceback.print_exc()
        return False

def check_output_files(base_name):
    """Проверяет наличие выходных файлов"""
    print_separator("ПРОВЕРКА ВЫХОДНЫХ ФАЙЛОВ")
    
    output_files = [
        (f"{base_name}_hatching_mask.png", "Маска штриховки"),
        (f"{base_name}_objects.json", "JSON файл с объектами"),
        (f"{base_name}_objects_wall_coordinates.json", "JSON файл с координатами стен"),
        (f"{base_name}_objects_wall_polygons.svg", "SVG файл визуализации"),
    ]
    
    print("Проверка выходных файлов:")
    for file_path, description in output_files:
        check_file_exists(file_path, description)

def run_hatching_detection(input_file):
    """Этап 1: Обнаружение штриховки"""
    return run_script("hatching_mask.py", "Hatching Detection", input_file)

def run_export_objects(input_file):
    """Этап 2: Экспорт объектов в JSON"""
    return run_script("export_objects.py", "Export Objects to JSON", input_file)

def run_align_openings(json_file):
    """Этап 3: Выравнивание проемов на основе JSON"""
    return run_script("visualize_polygons_align.py", "Align Openings", json_file)

def run_visualization(json_file):
    """Этап 4: Визуализация полигонов с анализом junctions"""
    return run_script("visualize_polygons_w.py", "Visualize Polygons with Junction Analysis", json_file)

def main():
    """Основная функция pipeline"""
    # Парсим аргументы командной строки
    args = parse_arguments()
    input_file = args.input_file
    base_name = get_base_filename(input_file)
    
    print_separator("UNIFIED PIPELINE ДЛЯ ОБРАБОТКИ АРХИТЕКТУРНЫХ ПЛАНОВ")
    
    total_start_time = time.time()
    print(f"Время начала: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Входной файл: {input_file}")
    print(f"Базовое имя: {base_name}")
    
    # Этап 0: Проверка необходимых файлов
    if not check_required_files(input_file):
        print("\n✗ Ошибка: отсутствуют необходимые файлы")
        print("Пожалуйста, убедитесь, что все необходимые файлы присутствуют в директории")
        return False
    
    # Этап 1: Обнаружение штриховки
    print_separator("ЭТАП 1/4: ОБНАРУЖЕНИЕ ШТРИХОВКИ")
    if not run_hatching_detection(input_file):
        print("\n✗ Ошибка на этапе обнаружения штриховки")
        return False
    
    # Этап 2: Экспорт объектов в JSON
    print_separator("ЭТАП 2/4: ЭКСПОРТ ОБЪЕКТОВ В JSON")
    if not run_export_objects(input_file):
        print("\n✗ Ошибка на этапе экспорта объектов")
        return False
    
    # Этап 3: Выравнивание проемов
    print_separator("ЭТАП 3/4: ВЫРАВНИВАНИЕ ПРОЕМОВ")
    input_json = f"{base_name}_objects.json"
    if not run_align_openings(input_json):
        print("\n✗ Ошибка на этапе выравнивания проемов")
        return False
    
    # Этап 4: Визуализация полигонов
    print_separator("ЭТАП 4/4: ВИЗУАЛИЗАЦИЯ ПОЛИГОНОВ")
    # Используем файл с координатами стен, созданный на этапе 3
    wall_coordinates_json = f"{base_name}_wall_coordinates.json"
    if not run_visualization(wall_coordinates_json):
        print("\n✗ Ошибка на этапе визуализации")
        return False
    
    # Завершение
    total_end_time = time.time()
    total_elapsed_time = total_end_time - total_start_time
    
    print_separator("PIPELINE ЗАВЕРШЕН УСПЕШНО")
    print(f"Общее время выполнения: {total_elapsed_time:.2f} секунд")
    print(f"Время завершения: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Проверка выходных файлов
    check_output_files(base_name)
    
    print("\n✓ Все этапы pipeline выполнены успешно!")
    print("\nСозданные файлы:")
    print(f"  - {base_name}_hatching_mask.png (маска штриховки)")
    print(f"  - {base_name}_objects.json (объекты плана, выровненные)")
    print(f"  - {base_name}_objects_wall_coordinates.json (координаты стен)")
    print(f"  - {base_name}_objects_wall_polygons.svg (визуализация)")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
