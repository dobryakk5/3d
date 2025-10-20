#!/bin/bash

# Скрипт запуска с инверсией в коде
echo "Запуск с инверсией в коде"
echo "========================="

# Переходим в директорию со скриптами
cd "$(dirname "$0")"

echo "Шаг 1: Расширение JSON файла новыми полями..."
python extend_openings_json.py

echo ""
echo "Шаг 2: Исправление высоты всех окон до 65 см..."
python fix_all_windows_height.py

echo ""
echo "Шаг 3: Создание 3D стен с инверсией в коде..."
/Applications/Blender.app/Contents/MacOS/Blender --background --python create_walls_with_code_inversion.py

echo ""
echo "========================="
echo "РЕЗУЛЬТАТЫ:"
echo "1. JSON файл расширен полями bottom_height и opening_height"
echo "2. Все окна установлены на высоту 65 см от пола"
echo "3. Инверсия выполняется в коде (invert_x = True)"
echo "4. Созданы заглушки для всех 13 проемов (25 элементов)"
echo "5. Включая окно w2 с заглушками под (0.65м) и над (0.85м) окном"
echo "6. Создан изометрический рендер: wall_coordinates_isometric.jpg"
echo "7. Экспортирован OBJ файл: wall_coordinates_3d.obj"
echo ""
echo "Запуск с инверсией в коде завершен!"
echo "========================="