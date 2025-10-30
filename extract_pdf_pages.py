#!/usr/bin/env python3
"""
Скрипт для извлечения страниц из PDF файла "Проект дом короб.pdf"
Извлекает вторую и третью страницы (пропуская первую с картинкой)
и сохраняет их как файлы 1.jpg и 2.jpg
"""

import fitz  # PyMuPDF
import os
import sys

def extract_pdf_pages(pdf_path="Проект дом короб.pdf", output_dir="."):
    """
    Извлекает страницы из PDF файла
    
    Args:
        pdf_path (str): Путь к PDF файлу
        output_dir (str): Директория для сохранения изображений
    """
    try:
        # Проверяем существование файла
        if not os.path.exists(pdf_path):
            print(f"Ошибка: Файл {pdf_path} не найден")
            return False
        
        # Открываем PDF документ
        print(f"Открываем PDF файл: {pdf_path}")
        doc = fitz.open(pdf_path)
        
        # Проверяем количество страниц
        page_count = doc.page_count
        print(f"Всего страниц в PDF: {page_count}")
        
        if page_count < 3:
            print(f"Ошибка: В PDF файле всего {page_count} страниц, нужно минимум 3")
            doc.close()
            return False
        
        # Настройки качества изображений
        dpi = 300  # Высокое качество для чертежей
        zoom = dpi / 72  # Коэффициент масштабирования
        
        # Извлекаем вторую страницу (индекс 1) и сохраняем как 1.jpg
        print("Извлекаем вторую страницу...")
        page = doc.load_page(1)  # Индекс 1 = вторая страница
        
        # Создаем матрицу преобразования для высокого качества
        mat = fitz.Matrix(zoom, zoom)
        
        # Рендерим страницу в изображение
        pix = page.get_pixmap(matrix=mat)
        
        # Сохраняем как JPEG
        output_path_1 = os.path.join(output_dir, "1.jpg")
        pix.save(output_path_1, "jpeg")
        print(f"Вторая страница сохранена как: {output_path_1}")
        
        # Извлекаем третью страницу (индекс 2) и сохраняем как 2.jpg
        print("Извлекаем третью страницу...")
        page = doc.load_page(2)  # Индекс 2 = третья страница
        
        # Рендерим страницу в изображение
        pix = page.get_pixmap(matrix=mat)
        
        # Сохраняем как JPEG
        output_path_2 = os.path.join(output_dir, "2.jpg")
        pix.save(output_path_2, "jpeg")
        print(f"Третья страница сохранена как: {output_path_2}")
        
        # Закрываем документ
        doc.close()
        
        print("\nГотово! Страницы успешно извлечены:")
        print(f"  - {output_path_1}")
        print(f"  - {output_path_2}")
        
        return True
        
    except Exception as e:
        print(f"Ошибка при обработке PDF: {e}")
        return False

def main():
    """Основная функция"""
    print("=== Извлечение страниц из PDF ===")
    
    # Определяем путь к PDF файлу
    pdf_path = "Проект дом короб.pdf"
    
    # Извлекаем страницы
    success = extract_pdf_pages(pdf_path)
    
    if success:
        print("\n✓ Операция завершена успешно!")
        
        # Проверяем созданные файлы
        if os.path.exists("1.jpg"):
            size_1 = os.path.getsize("1.jpg")
            print(f"  - Файл 1.jpg создан (размер: {size_1} байт)")
        
        if os.path.exists("2.jpg"):
            size_2 = os.path.getsize("2.jpg")
            print(f"  - Файл 2.jpg создан (размер: {size_2} байт)")
    else:
        print("\n✗ Операция завершилась с ошибкой")
        sys.exit(1)

if __name__ == "__main__":
    main()