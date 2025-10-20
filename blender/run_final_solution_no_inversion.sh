#!/bin/bash

# Скрипт запуска без инверсии (инверсия в JSON)
echo "Запуск без инверсии (инверсия в JSON)"
echo "====================================="

# Переходим в директорию со скриптами
cd "$(dirname "$0")"

echo "Шаг 1: Расширение JSON файла новыми полями..."
python extend_openings_json.py

echo ""
echo "Шаг 2: Исправление высоты всех окон до 65 см..."
python fix_all_windows_height.py

echo ""
echo "Шаг 3: Инвертирование координат в JSON файле..."
python invert_json_coordinates.py

echo ""
echo "Шаг 4: Создание 3D стен без инверсии в коде..."
/Applications/Blender.app/Contents/MacOS/Blender --background --python create_walls_2m.py

echo ""
echo "====================================="
echo "РЕЗУЛЬТАТЫ:"
echo "1. JSON файл расширен полями bottom_height и opening_height"
echo "2. Все окна установлены на высоту 65 см от пола"
echo "3. Координаты в JSON файле инвертированы"
echo "4. Инверсия в коде отключена (invert_x = False)"
echo "5. Созданы заглушки для всех 13 проемов (26 элементов)"
echo "6. Включая окно w2 с заглушками под (0.65м) и над (0.85м) окном"
echo "7. Создан изометрический рендер: wall_coordinates_isometric.jpg"
echo "8. Экспортирован OBJ файл: wall_coordinates_3d.obj"
echo ""
echo "Запуск без инверсии завершен!"
echo "====================================="