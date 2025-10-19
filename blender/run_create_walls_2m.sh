#!/bin/bash

# Автоматический запуск скрипта создания 3D стен высотой 2 метра в Blender
# Для macOS

echo "=========================================="
echo "АВТОМАТИЧЕСКИЙ ЗАПУСК СОЗДАНИЯ 3D СТЕН (2М)"
echo "=========================================="

# Путь к Blender (измените если нужно)
BLENDER_PATH="/Applications/Blender.app/Contents/MacOS/Blender"

# Путь к скрипту
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/create_walls_2m.py"

# Проверяем существование Blender
if [ ! -f "$BLENDER_PATH" ]; then
    echo "ОШИБКА: Blender не найден по пути: $BLENDER_PATH"
    echo "Пожалуйста, установите Blender или измените путь в скрипте"
    exit 1
fi

# Проверяем существование скрипта
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "ОШИБКА: Скрипт не найден по пути: $SCRIPT_PATH"
    exit 1
fi

# Проверяем существование файла с данными
DATA_PATH="$SCRIPT_DIR/wall_coordinates.json"
if [ ! -f "$DATA_PATH" ]; then
    echo "ОШИБКА: Файл с данными не найден: $DATA_PATH"
    exit 1
fi

echo "Запуск Blender с автоматическим созданием стен высотой 2 метра..."
echo "Путь к Blender: $BLENDER_PATH"
echo "Путь к скрипту: $SCRIPT_PATH"
echo ""

# Запускаем Blender в фоновом режиме с выполнением скрипта
"$BLENDER_PATH" --background --python "$SCRIPT_PATH"

echo ""
echo "=========================================="
echo "ЗАВЕРШЕНО!"
echo "Проверьте результат в файле: $SCRIPT_DIR/wall_coordinates_3d.obj"
echo "=========================================="