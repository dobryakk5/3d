# 🎓 Руководство по дообучению модели CubiCasa5K

## Стратегия дообучения для российских CAD чертежей

### 📋 Общий подход: Transfer Learning (перенос обучения)

Модель уже умеет:
- ✅ Находить планировку
- ✅ Понимать что такое комнаты
- ✅ Различать стены и пространство

Нужно научить:
- 🎯 Распознавать штриховку как стены
- 🎯 Видеть дуги как двери
- 🎯 Находить окна по текстовым меткам

---

## 🎯 План дообучения (от простого к сложному)

### Этап 1: Стены (САМОЕ ПРОСТОЕ - начнём с этого) ⭐

**Почему начинаем со стен:**
- ✅ Самая простая задача
- ✅ Больше всего данных на плане
- ✅ Четкие границы
- ✅ Модель уже понимает концепцию стен

**Что нужно:**

1. **Собрать данные: 50-100 планов**
   ```
   plan_floor1.jpg    → wall_mask_1.png
   plan_floor2.jpg    → wall_mask_2.png
   plan_floor3.jpg    → wall_mask_3.png
   ...
   ```

2. **Разметить стены** (создать маски):
   - Белый цвет (255) = стена
   - Черный цвет (0) = не стена
   - Формат: PNG, размер как оригинал

3. **Дообучить только слой стен**:
   - Заморозить остальные слои модели
   - Обучить только выход для класса "Wall"
   - 10-20 эпох

**Ожидаемый результат:** 85-90% точность на стенах

---

### Этап 2: Двери (СРЕДНЯЯ СЛОЖНОСТЬ)

**Почему двери вторые:**
- ✅ Четкие символы (дуги)
- ✅ Ограниченное количество на плане
- ⚠️ Нужно распознать дугу + текст

**Подход 1: Дообучение на дугах (проще)**

1. **Собрать 30-50 примеров дверей**
   ```
   door_001.jpg → door_mask_001.png
   door_002.jpg → door_mask_002.png
   ```

2. **Разметка:**
   - Вырезать области с дверями (128x128px)
   - Отметить где дверь (белый), где нет (черный)
   - Включить дугу в разметку

3. **Аугментация:**
   - Повороты (0°, 90°, 180°, 270°)
   - Масштабирование (0.8x - 1.2x)
   - Получаем 200-300 примеров из 50

**Подход 2: Hybrid с OCR (точнее)**

1. **Детекция дуг через Computer Vision**
   - Используем HoughCircles (уже есть в коде)

2. **Находим текст ДВ-***
   - Tesseract OCR для кириллицы
   - Ищем паттерн "ДВ-\d+"

3. **Связываем дугу с текстом**
   - Ближайшая дуга к тексту = дверь

**Ожидаемый результат:** 70-80% точность

---

### Этап 3: Окна (САМОЕ СЛОЖНОЕ)

**Почему окна последние:**
- ⚠️ Только текстовые метки (ОК-*, ДН-*)
- ⚠️ Нет четкого графического символа
- ⚠️ Нужен OCR + контекст

**Подход: OCR + контекстная модель**

1. **Находим текст OCR**
   ```python
   import pytesseract

   # Конфиг для кириллицы
   config = '--oem 3 --psm 6 -l rus'
   text = pytesseract.image_to_data(image, config=config, output_type=Output.DICT)

   # Фильтруем только ОК-*, ДН-*
   for i, word in enumerate(text['text']):
       if 'ОК' in word or 'ДН' in word:
           x, y, w, h = text['left'][i], text['top'][i], text['width'][i], text['height'][i]
           # Это окно!
   ```

2. **Смотрим контекст:**
   - Окно обычно на внешней стене
   - Проверяем стену рядом с текстом
   - Если стена внешняя → окно

3. **Создаём маску окон**

**Ожидаемый результат:** 60-70% точность

---

## 💻 Практическая реализация (Этап 1: Стены)

### Шаг 1: Создание инструмента разметки

Я создам простой инструмент для разметки стен:

```python
# annotate_walls.py
import cv2
import numpy as np
from pathlib import Path

class WallAnnotator:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        self.mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        self.drawing = False
        self.brush_size = 10

    def draw(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                cv2.circle(self.mask, (x, y), self.brush_size, 255, -1)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False

    def annotate(self):
        cv2.namedWindow('Annotate Walls')
        cv2.setMouseCallback('Annotate Walls', self.draw)

        while True:
            # Overlay mask on image
            overlay = self.image.copy()
            overlay[self.mask > 0] = [0, 255, 0]  # Green for walls
            display = cv2.addWeighted(self.image, 0.7, overlay, 0.3, 0)

            cv2.imshow('Annotate Walls', display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):  # Save
                return self.mask
            elif key == ord('q'):  # Quit
                return None
            elif key == ord('+'):  # Increase brush
                self.brush_size += 2
            elif key == ord('-'):  # Decrease brush
                self.brush_size = max(2, self.brush_size - 2)

# Использование
annotator = WallAnnotator('plan_floor1.jpg')
mask = annotator.annotate()
if mask is not None:
    cv2.imwrite('wall_mask_1.png', mask)
```

### Шаг 2: Автоматическая разметка (Semi-supervised)

Используем существующую модель для создания первичной разметки:

```python
# auto_annotate.py
import torch
import numpy as np
from PIL import Image

def auto_annotate_walls(image_path, model):
    """Создаёт первичную разметку, которую потом можно подправить вручную"""

    # Загрузить изображение
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)

    # Получить предсказание
    with torch.no_grad():
        pred = model(img_tensor)

    # Извлечь стены (класс 2)
    rooms_logits = pred[0, 21:33]
    rooms_pred = torch.softmax(rooms_logits, 0)
    wall_prob = rooms_pred[2].cpu().numpy()

    # Порог
    wall_mask = (wall_prob > 0.5).astype(np.uint8) * 255

    return wall_mask

# Создаём разметку автоматически, потом уточняем вручную
mask = auto_annotate_walls('plan_floor1.jpg', model)
Image.fromarray(mask).save('wall_mask_1_auto.png')
```

### Шаг 3: Подготовка датасета

```python
# prepare_dataset.py
import os
from pathlib import Path
import json

def prepare_training_data(data_dir='training_data'):
    """
    Структура:
    training_data/
        images/
            plan_001.jpg
            plan_002.jpg
            ...
        masks/
            walls/
                plan_001.png
                plan_002.png
            doors/
                plan_001.png
                plan_002.png
    """

    dataset = {
        'images': [],
        'masks': []
    }

    images_dir = Path(data_dir) / 'images'
    masks_dir = Path(data_dir) / 'masks' / 'walls'

    for img_path in sorted(images_dir.glob('*.jpg')):
        mask_path = masks_dir / f"{img_path.stem}.png"

        if mask_path.exists():
            dataset['images'].append(str(img_path))
            dataset['masks'].append(str(mask_path))

    # Сохранить список
    with open(f'{data_dir}/dataset.json', 'w') as f:
        json.dump(dataset, f, indent=2)

    print(f"Prepared {len(dataset['images'])} training samples")

    return dataset
```

### Шаг 4: Дообучение модели

```python
# fine_tune_walls.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import json

class WallDataset(Dataset):
    def __init__(self, json_path, transform=None):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.data['images'])

    def __getitem__(self, idx):
        # Load image
        img = Image.open(self.data['images'][idx]).convert('RGB')
        img = np.array(img, dtype=np.float32) / 255.0
        img = (img - 0.5) * 2  # Normalize
        img = torch.from_numpy(img).permute(2, 0, 1)

        # Load mask
        mask = Image.open(self.data['masks'][idx]).convert('L')
        mask = np.array(mask, dtype=np.float32) / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0)

        return img, mask

def fine_tune_wall_detection(model, dataset_path, epochs=20, lr=1e-4):
    """
    Дообучение только для класса стен
    """

    # Заморозить все веса кроме последнего слоя
    for param in model.parameters():
        param.requires_grad = False

    # Разморозить только выход для стен (каналы 21-23)
    # Это room segmentation слои
    model.conv4_.weight.requires_grad = True
    model.conv4_.bias.requires_grad = True
    model.upsample.weight.requires_grad = True
    model.upsample.bias.requires_grad = True

    # Датасет
    dataset = WallDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Оптимизатор - только для разморожженных весов
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )

    # Loss function
    criterion = nn.BCEWithLogitsLoss()

    model.cuda()
    model.train()

    for epoch in range(epochs):
        total_loss = 0

        for images, masks in dataloader:
            images = images.cuda()
            masks = masks.cuda()

            # Forward
            outputs = model(images)

            # Извлечь стены (класс 2 в room segmentation)
            wall_logits = outputs[:, 23:24, :, :]  # Канал стен

            # Loss
            loss = criterion(wall_logits, masks)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        # Сохранить каждые 5 эпох
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'loss': avg_loss
            }, f'finetuned_walls_epoch_{epoch+1}.pkl')

    print("Fine-tuning complete!")
    return model

# Использование
from floortrans.models import get_model

# Загрузить базовую модель
model = get_model('hg_furukawa_original', 51)
model.conv4_ = torch.nn.Conv2d(256, 44, bias=True, kernel_size=1)
model.upsample = torch.nn.ConvTranspose2d(44, 44, kernel_size=4, stride=4)

checkpoint = torch.load('model_best_val_loss_var.pkl')
model.load_state_dict(checkpoint['model_state'])

# Дообучить
model = fine_tune_wall_detection(
    model,
    'training_data/dataset.json',
    epochs=20,
    lr=1e-4
)

# Сохранить финальную модель
torch.save({
    'model_state': model.state_dict(),
}, 'model_finetuned_walls.pkl')
```

---

## 📊 Минимальные требования к данным

### Для стен:
- **Минимум:** 30 планов
- **Оптимально:** 100 планов
- **Время разметки:** ~5-10 минут на план
- **Итого:** 5-16 часов работы

### Для дверей:
- **Минимум:** 20 планов (100+ дверей)
- **Оптимально:** 50 планов (250+ дверей)
- **Время:** ~10 минут на план
- **Итого:** 3-8 часов

### Для окон:
- **Минимум:** 20 планов (100+ окон)
- **OCR подход:** не требует разметки
- **Время:** настройка OCR ~2 часа

---

## 🚀 Быстрый старт (минимальный набор)

Если времени мало, вот минимум для первого эксперимента:

1. **10 планов** похожих на ваш
2. **Разметить только стены** (самое простое)
3. **Дообучить 10 эпох** (~1 час на GPU)
4. **Протестировать**

Ожидаемый результат: +20-30% точности на стенах

---

## 🛠️ Альтернатива: Использовать SAM (Segment Anything)

Новый подход от Meta - можно разметить интерактивно:

```python
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# Загрузить SAM
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")

# Автоматическая сегментация
mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(image)

# Выбрать маски стен вручную (кликнув на них)
# Намного быстрее чем рисовать!
```

---

## ❓ Что лучше для вас?

### Вариант A: Минимальные усилия (OCR-based)
✅ Время: 1 день
✅ Данные: 0 разметки
✅ Результат: 50-60% точность
- Только OCR для дверей (ДВ-*) и окон (ОК-*)
- Дуги через HoughCircles

### Вариант B: Средние усилия (Hybrid)
✅ Время: 1 неделя
✅ Данные: 30 планов
✅ Результат: 70-80% точность
- Дообучить стены
- OCR + CV для дверей/окон

### Вариант C: Максимум качества (Full fine-tuning)
✅ Время: 2-3 недели
✅ Данные: 100 планов
✅ Результат: 85-90% точность
- Дообучить все классы
- Большой датасет

---

## 📝 Следующие шаги

Что хотите делать?

1. **Начать с разметки стен** - я создам инструмент?
2. **Попробовать OCR подход** для дверей/окон без разметки?
3. **Использовать SAM** для быстрой разметки?
4. **Hybrid подход** - комбинация всего?

Напишите какой вариант вам ближе, и я создам готовый код!
