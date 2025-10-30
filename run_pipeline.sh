#!/bin/bash
# Shell скрипт для запуска unified pipeline обработки архитектурных планов

# Проверяем наличие параметра с именем входного файла
if [ $# -eq 0 ]; then
    echo "Ошибка: не указано имя входного JPG файла"
    echo "Использование: $0 путь/к/файлу.jpg [часть]"
    echo "Пример: $0 floor/plan_floor1.jpg"
    echo "Части:"
    echo "  1 - только первая часть (обработка изображения)"
    echo "  2 - только вторая часть (Blender)"
    echo "  без параметра - обе части"
    exit 1
fi

INPUT_FILE="$1"
PART="${2:-both}"  # По умолчанию запускаем обе части

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
BLENDER_DIR="blender"

# Проверяем наличие Python
if ! command -v python3 &> /dev/null; then
    echo "Ошибка: Python3 не найден. Пожалуйста, установите Python3."
    exit 1
fi

# Проверяем наличие Blender
if ! [ -f "/Applications/Blender.app/Contents/MacOS/Blender" ]; then
    echo "Ошибка: Blender не найден в /Applications/Blender.app/Contents/MacOS/Blender"
    exit 1
fi

# Запускаем первую часть pipeline
if [ "$PART" = "1" ] || [ "$PART" = "both" ]; then
    echo "=========================================="
    echo "ЗАПУСК ПЕРВОЙ ЧАСТИ PIPELINE"
    echo "=========================================="
    echo "Входной файл: $INPUT_FILE"
    
    # Переходим в директорию со скриптами для выполнения
    cd "$SCRIPT_DIR"
    
    # Получаем относительный путь к входному файлу от директории со скриптами
    RELATIVE_INPUT_FILE="../$INPUT_FILE"
    
    # Запускаем Python скрипт с относительным путем к входному файлу
    python3 floor_pipeline.py "$RELATIVE_INPUT_FILE"
    
    # Проверяем результат выполнения первой части
    if [ $? -ne 0 ]; then
        echo "Ошибка: первая часть pipeline завершилась с ошибкой"
        exit 1
    fi
    
    # Возвращаемся в исходную директорию
    cd ..
    
    echo "Первая часть pipeline завершена успешно"
fi

# Запускаем вторую часть pipeline
if [ "$PART" = "2" ] || [ "$PART" = "both" ]; then
    echo "=========================================="
    echo "ЗАПУСК ВТОРОЙ ЧАСТИ PIPELINE"
    echo "=========================================="
    
    # Вычисляем префикс на основе имени входного файла (без расширения)
    BASE_NAME=$(basename "$INPUT_FILE")
    BASE_NAME="${BASE_NAME%.*}"
    JSON_PREFIX="${BASE_NAME}_objects"
    
    # Проверяем наличие префиксного файла координат стен
    if [ ! -f "$SCRIPT_DIR/${JSON_PREFIX}_wall_coordinates.json" ]; then
        echo "Ошибка: файл $SCRIPT_DIR/${JSON_PREFIX}_wall_coordinates.json не найден"
        echo "Пожалуйста, сначала запустите первую часть pipeline для того же входного файла"
        exit 1
    fi
    
    # Копируем префиксный файл координат стен в директорию blender
    echo "Копирование ${JSON_PREFIX}_wall_coordinates.json в директорию blender..."
    cp "$SCRIPT_DIR/${JSON_PREFIX}_wall_coordinates.json" "$BLENDER_DIR/"
    
    if [ $? -ne 0 ]; then
        echo "Ошибка: не удалось скопировать ${JSON_PREFIX}_wall_coordinates.json в директорию blender"
        exit 1
    fi
    
    echo "Файл успешно скопирован"
    
    # Переходим в директорию blender
    cd "$BLENDER_DIR"
    
    # Запускаем invert_json_coord.py
    echo "Запуск invert_json_coord.py..."
    python3 invert_json_coord.py "${JSON_PREFIX}_wall_coordinates.json"
    
    if [ $? -ne 0 ]; then
        echo "Ошибка: invert_json_coord.py завершился с ошибкой"
        exit 1
    fi
    
    echo "invert_json_coord.py выполнен успешно"
    
    # Запускаем Blender в фоновом режиме
    echo "Запуск Blender в фоновом режиме для создания 3D стен..."
    echo "Это может занять несколько минут, пожалуйста подождите..."
    /Applications/Blender.app/Contents/MacOS/Blender --background --python create_walls_2m.py -- --json "${JSON_PREFIX}_wall_coordinates_inverted.json"
    blender_exit_code=$?
    
    if [ $blender_exit_code -eq 0 ]; then
        echo "Blender фоновый режим успешно завершил работу"
    else
        echo "Предупреждение: Blender завершил работу с кодом $blender_exit_code"
    fi
    
    echo "Проверка результатов работы Blender..."
    if [ -f "${JSON_PREFIX}_wall_coordinates_inverted_3d.obj" ]; then
        echo "✓ Найден файл ${JSON_PREFIX}_wall_coordinates_inverted_3d.obj"
    else
        echo "✗ Файл ${JSON_PREFIX}_wall_coordinates_inverted_3d.obj не найден"
    fi
    
    if [ -f "${JSON_PREFIX}_wall_coordinates_inverted_isometric.jpg" ]; then
        echo "✓ Найден файл ${JSON_PREFIX}_wall_coordinates_inverted_isometric.jpg"
    else
        echo "✗ Файл ${JSON_PREFIX}_wall_coordinates_inverted_isometric.jpg не найден"
    fi
    
    echo "Blender в фоновом режиме выполнен"
    
    # Запускаем Blender в обычном режиме
    echo "Запуск Blender в обычном режиме для финальной визуализации..."
    /Applications/Blender.app/Contents/MacOS/Blender --python create_outline_with_openings.py -- --json "${JSON_PREFIX}_wall_coordinates_inverted.json" --heights-obj "${JSON_PREFIX}_wall_coordinates_inverted_3d.obj" --out "${JSON_PREFIX}_outline_with_openings.obj"
    blender_normal_exit_code=$?
    
    if [ $blender_normal_exit_code -eq 0 ]; then
        echo "Blender в обычном режиме успешно завершил работу"
    else
        echo "Предупреждение: Blender в обычном режиме завершился с кодом $blender_normal_exit_code"
        # Не выходим с ошибкой, так как это может быть нормальным поведением
    fi
    
    echo "Проверка результатов работы Blender в обычном режиме..."
    if [ -f "${JSON_PREFIX}_outline_with_openings.obj" ]; then
        echo "✓ Найден файл ${JSON_PREFIX}_outline_with_openings.obj"
    else
        echo "✗ Файл ${JSON_PREFIX}_outline_with_openings.obj не найден"
    fi
    
    if [ -f "${JSON_PREFIX}_mesh_normals.json" ]; then
        echo "✓ Найден файл ${JSON_PREFIX}_mesh_normals.json"
    else
        echo "✗ Файл ${JSON_PREFIX}_mesh_normals.json не найден"
    fi
    
    echo "Blender в обычном режиме выполнен"
    
    # Возвращаемся в исходную директорию
    cd ..
    
    echo "Вторая часть pipeline завершена"
fi

echo "=========================================="
echo "PIPELINE ЗАВЕРШЕН УСПЕШНО"
echo "=========================================="

# Проверяем результат
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Pipeline выполнен успешно!"
    echo "=========================================="
    echo ""
    echo "Созданные файлы (в директории floor):"
    BASE_NAME=$(basename "$INPUT_FILE")
    BASE_NAME="${BASE_NAME%.*}"
    JSON_PREFIX="${BASE_NAME}_objects"
    ls -la "floor/${BASE_NAME}_hatching_mask.png" "floor/${BASE_NAME}_objects.json" "floor/${JSON_PREFIX}_wall_coordinates.json" "floor/${JSON_PREFIX}_wall_polygons.svg" 2>/dev/null || echo "Некоторые выходные файлы могут отсутствовать"
else
    echo ""
    echo "=========================================="
    echo "Ошибка выполнения pipeline!"
    echo "=========================================="
    exit 1
fi
