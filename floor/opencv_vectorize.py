# diagnostic.py
# Проверим ЧТО именно не получилось

from PIL import Image
import numpy as np

def diagnose_floor_plan(image_path):
    """
    Диагностика floor plan перед обработкой
    """
    img = Image.open(image_path)
    arr = np.array(img)
    
    print("🔍 Диагностика floor plan:")
    print(f"   Размер: {img.size}")
    print(f"   Режим: {img.mode}")
    print(f"   Формат: {img.format}")
    
    # Проверка 1: Цветной или ч/б?
    if len(arr.shape) == 3:
        print(f"   Каналы: {arr.shape[2]} (RGB)")
    else:
        print(f"   Каналы: 1 (Grayscale)")
    
    # Проверка 2: Контрастность
    if len(arr.shape) == 3:
        gray = np.mean(arr, axis=2)
    else:
        gray = arr
    
    contrast = gray.std()
    print(f"   Контрастность: {contrast:.1f}")
    
    if contrast < 30:
        print("   ⚠️  НИЗКАЯ контрастность - плохо для CV")
    
    # Проверка 3: Разрешение
    if img.size[0] < 512 or img.size[1] < 512:
        print("   ⚠️  Низкое разрешение - upscale перед обработкой")
    
    # Проверка 4: Текст на чертеже
    text_density = detect_text_density(arr)
    print(f"   Плотность текста: {text_density:.1f}%")
    
    if text_density > 20:
        print("   ⚠️  Много текста - нужен OCR + очистка")
    
    # Проверка 5: Сложность
    edge_density = detect_edge_density(arr)
    print(f"   Плотность деталей: {edge_density:.1f}%")
    
    if edge_density > 40:
        print("   ⚠️  Очень детальный чертеж - сложно для базовых моделей")

# Запустите это на вашем floor plan
diagnose_floor_plan('floor_plan.jpg')