#!/bin/bash
# Shell скрипт для запуска unified pipeline обработки архитектурных планов

# Проверяем наличие параметра с именем входного файла
if [ $# -eq 0 ]; then
    echo "Ошибка: не указано имя входного JPG файла"
    echo "Использование: $0 путь/к/файлу.jpg"
    echo "Пример: $0 floor/plan_floor1.jpg"
    exit 1
fi

INPUT_FILE="$1"

# Проверяем существование входного файла
if [ ! -f "$INPUT_FILE" ]; then
    echo "Ошибка: входной файл не найден: $INPUT_FILE"
    exit 1
fi

# Проверяем, что это JPG файл
if [[ ! "$INPUT_FILE" =~ \.jpg$ && ! "$INPUT_FILE" =~ \.jpeg$ ]]; then
    echo "Ошибка: входной файл должен быть в формате JPG: $INPUT_FILE"
    exit 1
fi

# Определяем директорию со скриптами (относительно текущей директории)
SCRIPT_DIR="floor"

# Проверяем наличие Python
if ! command -v python3 &> /dev/null; then
    echo "Ошибка: Python3 не найден. Пожалуйста, установите Python3."
    exit 1
fi

# Запускаем pipeline
echo "Запуск unified pipeline..."
echo "Входной файл: $INPUT_FILE"

# Переходим в директорию со скриптами для выполнения
cd "$SCRIPT_DIR"

# Получаем относительный путь к входному файлу от директории со скриптами
RELATIVE_INPUT_FILE="../$INPUT_FILE"

# Запускаем Python скрипт с относительным путем к входному файлу
python3 run_pipeline.py "$RELATIVE_INPUT_FILE"

# Проверяем результат
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Pipeline выполнен успешно!"
    echo "=========================================="
    echo ""
    echo "Созданные файлы:"
    # Получаем базовое имя файла без расширения для поиска выходных файлов
    BASE_NAME=$(basename "$INPUT_FILE" .jpg)
    BASE_NAME=$(basename "$BASE_NAME" .jpeg)
    
    # Ищем выходные файлы в текущей директории (floor)
    ls -la "enhanced_hatching_strict_mask.png" "${BASE_NAME}_objects.json" "wall_coordinates.json" "wall_polygons.svg" 2>/dev/null || echo "Некоторые выходные файлы могут отсутствовать"
else
    echo ""
    echo "=========================================="
    echo "Ошибка выполнения pipeline!"
    echo "=========================================="
    exit 1
fi