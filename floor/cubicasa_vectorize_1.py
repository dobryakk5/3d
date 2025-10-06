# cubicasa_vectorize.py
# Автоматическая векторизация floor plan → SVG

import torch
import cv2
import numpy as np
from PIL import Image
import svgwrite
from pathlib import Path

# Импорт модели CubiCasa
from floortrans.models import get_model
from floortrans.post_prosessing import split_prediction, get_polygons

class FloorPlanVectorizer:
    """
    Автоматическая векторизация floor plans с помощью ML
    """
    
    def __init__(self, model_path='model_best_val_loss_var.pkl'):
        print("🔄 Загрузка модели CubiCasa...")

        # Загрузить модель
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

        # Загрузить checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)

        # Создать модель с правильным количеством классов
        n_classes = 44  # Количество классов в checkpoint
        self.model = get_model('hg_furukawa_original', 51)
        self.model.conv4_ = torch.nn.Conv2d(256, n_classes, bias=True, kernel_size=1)
        self.model.upsample = torch.nn.ConvTranspose2d(n_classes, n_classes, kernel_size=4, stride=4)

        # Загрузить веса
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()
        self.model.to(self.device)

        print(f"✅ Модель загружена на {self.device}")
    
    def vectorize(self, image_path, output_svg='output.svg', output_dxf='output.dxf'):
        """
        Векторизует floor plan
        
        Returns:
            dict: {
                'walls': [...],      # Список стен (линии)
                'doors': [...],      # Список дверей
                'windows': [...],    # Список окон
                'rooms': [...]       # Полигоны комнат
            }
        """
        print(f"📄 Обработка: {image_path}")
        
        # 1. Загрузить и подготовить изображение
        image = Image.open(image_path).convert('RGB')
        img_size = 512  # Размер для модели
        image_resized = image.resize((img_size, img_size))
        
        # Конвертировать в тензор
        img_array = np.array(image_resized).transpose((2, 0, 1))
        img_tensor = torch.FloatTensor(img_array).unsqueeze(0) / 255.0
        img_tensor = img_tensor.to(self.device)
        
        # 2. Инференс модели
        print("🤖 Запуск ML-модели...")
        with torch.no_grad():
            predictions = self.model(img_tensor)

        # 3. Постобработка (извлечение структуры)
        print("🔍 Извлечение стен, дверей, окон...")
        split = [21, 12, 11]
        predictions = predictions.cpu()  # Переместить на CPU перед обработкой
        heatmaps, rooms, icons = split_prediction(predictions, (img_size, img_size), split)

        all_opening_types = [1, 2]  # Window, Door
        polygons, types, room_polygons, room_types = get_polygons(
            (heatmaps, rooms, icons), 0.4, all_opening_types)

        result = {
            'polygons': polygons,
            'types': types,
            'room_polygons': room_polygons,
            'room_types': room_types,
            'heatmaps': heatmaps,
            'rooms': rooms,
            'icons': icons
        }


        # 4. Конвертировать в удобный формат
        vectors = self._process_results(result, image.size)
        
        # 5. Экспорт в SVG
        self._export_svg(vectors, output_svg, image.size)
        
        # 6. Экспорт в DXF (опционально)
        self._export_dxf(vectors, output_dxf)
        
        print(f"✅ Готово!")
        print(f"   SVG: {output_svg}")
        print(f"   DXF: {output_dxf}")
        
        return vectors
    
    def _process_results(self, result, original_size):
        """
        Конвертирует результаты модели в векторы
        """
        vectors = {
            'walls': [],
            'doors': [],
            'windows': [],
            'rooms': []
        }

        # Масштаб для возврата к оригинальному размеру
        scale_x = original_size[0] / 512
        scale_y = original_size[1] / 512

        polygons = result['polygons']
        types = result['types']
        room_polygons = result['room_polygons']
        room_types = result['room_types']

        # Обработать стены, двери, окна
        for i, pol in enumerate(polygons):
            pol_type = types[i]
            scaled_pol = [(x * scale_x, y * scale_y) for y, x in pol]

            if pol_type['type'] == 'wall':
                vectors['walls'].append(scaled_pol)
            elif pol_type['type'] == 'icon':
                icon_class = pol_type['class']
                if icon_class == 1:  # Window
                    vectors['windows'].append(scaled_pol)
                elif icon_class == 2:  # Door
                    vectors['doors'].append(scaled_pol)

        # Обработать комнаты
        for i, room_pol in enumerate(room_polygons):
            if hasattr(room_pol, 'exterior'):
                # Shapely Polygon
                coords = list(room_pol.exterior.coords)
                scaled_coords = [(x * scale_x, y * scale_y) for x, y in coords]
                vectors['rooms'].append(scaled_coords)

        return vectors
    
    def _extract_lines(self, mask):
        """
        Извлекает линии из binary mask с помощью HoughLinesP
        """
        # Скелетонизация для получения тонких линий
        skeleton = cv2.ximgproc.thinning(mask.astype(np.uint8) * 255)
        
        # Hough Line Transform
        lines = cv2.HoughLinesP(
            skeleton,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=20,
            maxLineGap=5
        )
        
        if lines is None:
            return []
        
        # Конвертировать в формат [(x1,y1), (x2,y2)]
        return [
            ((line[0][0], line[0][1]), (line[0][2], line[0][3]))
            for line in lines
        ]
    
    def _extract_rectangles(self, mask):
        """
        Извлекает прямоугольники (двери, окна)
        """
        contours, _ = cv2.findContours(
            mask.astype(np.uint8) * 255,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        rectangles = []
        for cnt in contours:
            # Аппроксимировать контур прямоугольником
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            rectangles.append(box)
        
        return rectangles
    
    def _extract_contours(self, mask):
        """
        Извлекает контуры комнат
        """
        contours, _ = cv2.findContours(
            mask.astype(np.uint8) * 255,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        return [cnt.squeeze() for cnt in contours if len(cnt) > 2]
    
    def _scale_line(self, line, scale_x, scale_y):
        """Масштабирует линию"""
        (x1, y1), (x2, y2) = line
        return (
            (x1 * scale_x, y1 * scale_y),
            (x2 * scale_x, y2 * scale_y)
        )
    
    def _scale_rect(self, rect, scale_x, scale_y):
        """Масштабирует прямоугольник"""
        return [(x * scale_x, y * scale_y) for (x, y) in rect]
    
    def _scale_contour(self, contour, scale_x, scale_y):
        """Масштабирует контур"""
        return [(x * scale_x, y * scale_y) for (x, y) in contour]
    
    def _export_svg(self, vectors, output_path, size):
        """
        Экспортирует векторы в SVG
        """
        dwg = svgwrite.Drawing(output_path, size=size)

        # Добавить стены (черные полигоны)
        wall_group = dwg.g(id='walls', fill='black', fill_opacity=0.8)
        for wall_pol in vectors['walls']:
            if len(wall_pol) > 0:
                dwg.add(dwg.polygon(points=wall_pol))
        dwg.add(wall_group)
        
        # Добавить двери (синие прямоугольники)
        door_group = dwg.g(id='doors', fill='blue', fill_opacity=0.3)
        for rect in vectors['doors']:
            points = [(x, y) for (x, y) in rect]
            door_group.add(dwg.polygon(points=points))
        dwg.add(door_group)
        
        # Добавить окна (голубые прямоугольники)
        window_group = dwg.g(id='windows', fill='cyan', fill_opacity=0.3)
        for rect in vectors['windows']:
            points = [(x, y) for (x, y) in rect]
            window_group.add(dwg.polygon(points=points))
        dwg.add(window_group)
        
        # Добавить комнаты (полупрозрачные полигоны)
        room_group = dwg.g(id='rooms', fill='lightgray', fill_opacity=0.2, stroke='gray')
        for contour in vectors['rooms']:
            points = [(x, y) for (x, y) in contour]
            room_group.add(dwg.polygon(points=points))
        dwg.add(room_group)
        
        dwg.save()
    
    def _export_dxf(self, vectors, output_path):
        """
        Экспортирует векторы в DXF
        """
        try:
            import ezdxf
        except ImportError:
            print("⚠️  ezdxf не установлен, пропускаю DXF экспорт")
            return
        
        # Создать новый DXF документ
        doc = ezdxf.new('R2010')
        msp = doc.modelspace()
        
        # Добавить стены на слой "WALLS"
        doc.layers.new(name='WALLS', dxfattribs={'color': 7})
        for (x1, y1), (x2, y2) in vectors['walls']:
            msp.add_line((x1, y1), (x2, y2), dxfattribs={'layer': 'WALLS'})
        
        # Добавить двери на слой "DOORS"
        doc.layers.new(name='DOORS', dxfattribs={'color': 5})
        for rect in vectors['doors']:
            points = [(x, y, 0) for (x, y) in rect]
            msp.add_lwpolyline(points, dxfattribs={'layer': 'DOORS'})
        
        # Добавить окна на слой "WINDOWS"
        doc.layers.new(name='WINDOWS', dxfattribs={'color': 4})
        for rect in vectors['windows']:
            points = [(x, y, 0) for (x, y) in rect]
            msp.add_lwpolyline(points, dxfattribs={'layer': 'WINDOWS'})
        
        # Сохранить
        doc.saveas(output_path)


# ============================================
# ИСПОЛЬЗОВАНИЕ
# ============================================

if __name__ == "__main__":
    # Создать векторайзер
    vectorizer = FloorPlanVectorizer(model_path='model_best_val_loss_var.pkl')
    
    # Векторизовать floor plan
    result = vectorizer.vectorize(
        image_path='floor_plan.jpg',
        output_svg='floor_plan.svg',
        output_dxf='floor_plan.dxf'
    )
    
    # Статистика
    print(f"\n📊 Статистика:")
    print(f"   Стен: {len(result['walls'])}")
    print(f"   Дверей: {len(result['doors'])}")
    print(f"   Окон: {len(result['windows'])}")
    print(f"   Комнат: {len(result['rooms'])}")