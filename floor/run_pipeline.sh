#!/bin/bash
# Shell скрипт для запуска unified pipeline обработки архитектурных планов

# Переходим в директорию со скриптами
cd "$(dirname "$0")"

# Проверяем наличие Python
if ! command -v python3 &> /dev/null; then
    echo "Ошибка: Python3 не найден. Пожалуйста, установите Python3."
    exit 1
fi

# Проверяем наличие pip
if ! command -v pip3 &> /dev/null; then
    echo "Ошибка: pip3 не найден. Пожалуйста, установите pip3."
    exit 1
fi

# Проверяем наличие файла requirements.txt
if [ -f "requirements.txt" ]; then
    echo "Проверка зависимостей Python..."
    pip3 install -r requirements.txt
fi

# Запускаем pipeline
echo "Запуск unified pipeline..."
python3 run_pipeline.py

# Проверяем результат
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Pipeline выполнен успешно!"
    echo "=========================================="
    echo ""
    echo "Созданные файлы:"
    ls -la enhanced_hatching_strict_mask.png plan_floor1_objects.json wall_coordinates.json wall_polygons.svg 2>/dev/null || echo "Некоторые выходные файлы могут отсутствовать"
else
    echo ""
    echo "=========================================="
    echo "Ошибка выполнения pipeline!"
    echo "=========================================="
    exit 1
fi