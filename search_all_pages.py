#!/usr/bin/env python3
"""
Скрипт для поиска "ОК-" и "ДВ-" на всех страницах PDF
"""

import fitz  # PyMuPDF
import json

def search_all_pages(pdf_path="Проект дом короб.pdf"):
    """Ищет "ОК-" и "ДВ-" на всех страницах PDF"""
    
    try:
        doc = fitz.open(pdf_path)
        print(f"Открыт PDF: {pdf_path}")
        print(f"Всего страниц: {doc.page_count}")
        
        all_results = {}
        
        for page_idx in range(doc.page_count):
            page_num = page_idx + 1
            page_key = f"страница {page_num}"
            
            print(f"\n=== Анализ страницы {page_num} ===")
            page = doc.load_page(page_idx)
            
            # Ищем "ОК-" и "ДВ-"
            ok_instances = page.search_for("ОК-")
            dv_instances = page.search_for("ДВ-")
            
            # Также пробуем более широкий поиск
            ok_wide = page.search_for("ОК")
            dv_wide = page.search_for("ДВ")
            
            print(f"Поиск 'ОК-': {len(ok_instances)} вхождений")
            print(f"Поиск 'ДВ-': {len(dv_instances)} вхождений")
            print(f"Поиск 'ОК': {len(ok_wide)} вхождений")
            print(f"Поиск 'ДВ': {len(dv_wide)} вхождений")
            
            # Собираем результаты
            page_results = {
                "ОК": [],
                "ДВ": []
            }
            
            # Добавляем точные совпадения
            for rect in ok_instances:
                text = page.get_text("text", clip=rect).strip()
                if text.startswith("ОК-"):
                    page_results["ОК"].append({
                        "текст": text,
                        "координаты": [rect.x0, rect.y0, rect.x1, rect.y1]
                    })
            
            for rect in dv_instances:
                text = page.get_text("text", clip=rect).strip()
                if text.startswith("ДВ-"):
                    page_results["ДВ"].append({
                        "текст": text,
                        "координаты": [rect.x0, rect.y0, rect.x1, rect.y1]
                    })
            
            # Если есть широкие совпадения, показываем их
            if ok_wide and not ok_instances:
                print("Найдены 'ОК' (без дефиса):")
                for rect in ok_wide:
                    text = page.get_text("text", clip=rect).strip()
                    print(f"  '{text}' - координаты: {rect}")
            
            if dv_wide and not dv_instances:
                print("Найдены 'ДВ' (без дефиса):")
                for rect in dv_wide:
                    text = page.get_text("text", clip=rect).strip()
                    print(f"  '{text}' - координаты: {rect}")
            
            # Добавляем в общие результаты только если есть что-то
            if page_results["ОК"] or page_results["ДВ"]:
                all_results[page_key] = page_results
        
        doc.close()
        
        # Сохраняем результаты
        if all_results:
            with open("found_objects.json", 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            print(f"\nРезультаты сохранены в found_objects.json")
        else:
            print("\nОбъекты 'ОК-' и 'ДВ-' не найдены ни на одной странице")
        
        return all_results
        
    except Exception as e:
        print(f"Ошибка: {e}")
        return None

def main():
    """Основная функция"""
    print("=== Поиск 'ОК-' и 'ДВ-' на всех страницах PDF ===")
    
    results = search_all_pages()
    
    if results:
        print(f"\nНайдено объектов на страницах:")
        total_ok = 0
        total_dv = 0
        
        for page_key, page_data in results.items():
            ok_count = len(page_data.get("ОК", []))
            dv_count = len(page_data.get("ДВ", []))
            total_ok += ok_count
            total_dv += dv_count
            
            if ok_count > 0 or dv_count > 0:
                print(f"  {page_key}: ОК-{ok_count}, ДВ-{dv_count}")
        
        print(f"\nИтого: ОК-{total_ok}, ДВ-{total_dv}")

if __name__ == "__main__":
    main()