import cv2
import os
import numpy as np

# Путь к шаблонам символов (должны быть изображения одного размера)
TEMPLATE_DIR = r'templates'

# Загрузка всех шаблонов в словарь
def load_templates():
    templates = {}
    for filename in os.listdir(TEMPLATE_DIR):
        if filename.endswith('.png'):
            label = os.path.splitext(filename)[0]  # имя файла = символ
            img = cv2.imread(os.path.join(TEMPLATE_DIR, filename), 0)
            templates[label] = img
    return templates

# Сравнение символа с шаблонами
def match_char(char_img, templates):
    best_score = float('inf')
    best_char = ''
    for label, template in templates.items():
        # Масштабируем шаблон под размер символа
        resized_template = cv2.resize(template, (char_img.shape[1], char_img.shape[0]))
        res = cv2.matchTemplate(char_img, resized_template, cv2.TM_SQDIFF)
        min_val, _, _, _ = cv2.minMaxLoc(res)
        if min_val < best_score:
            best_score = min_val
            best_char = label
    return best_char

# Предобработка изображения
def preprocess_image(path):
    image = cv2.imread(path)

    # Переводим в оттенки серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Повышаем контраст — CLAHE (адаптивный гистограммный эквалайзер)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Немного блюрим, чтобы сгладить шум
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

    # Порог с Otsu + инверсия (чёрный фон, белый текст)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # (опционально) Морфология для усиления контуров
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Отладка: покажи результат
    cv2.imshow("Threshold", thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return thresh



# Разделение символов (наивно — для разреженного текста)
def extract_characters(thresh_img):
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append((x, y, w, h, thresh_img[y:y+h, x:x+w]))

    # Сортировка по Y (сначала строки сверху вниз)
    boxes = sorted(boxes, key=lambda b: b[1])

    lines = []
    current_line = []
    line_threshold = 15  # насколько символы должны быть близко по Y, чтобы быть в одной строке

    for box in boxes:
        x, y, w, h, char_img = box
        if not current_line:
            current_line.append(box)
        else:
            prev_y = current_line[-1][1]
            if abs(y - prev_y) < line_threshold:
                current_line.append(box)
            else:
                lines.append(current_line)
                current_line = [box]

    if current_line:
        lines.append(current_line)

    # В каждой строке сортируем символы по X
    sorted_char_lines = []
    for line in lines:
        sorted_line = sorted(line, key=lambda b: b[0])  # сортировка по x
        sorted_char_lines.append([(x, char_img) for x, _, _, _, char_img in sorted_line])

    return sorted_char_lines  # ← Теперь каждый символ представлен как (x, char_img)



# Основной процесс
def recognize_text(image_path):
    templates = load_templates()
    thresh = preprocess_image(image_path)
    lines = extract_characters(thresh)

    recognized = ''
    for line in lines:
        line_result = ''
        # Получаем список координат X и ширин символов
        char_positions = [(x, img, img.shape[1]) for x, img in line]
        # Сортируем по x, на всякий случай
        char_positions.sort(key=lambda tup: tup[0])
        # Средняя ширина символа
        avg_char_width = np.mean([w for _, _, w in char_positions])

        prev_right = None
        for x, char_img, w in char_positions:
            if prev_right is not None:
                gap = x - prev_right
                if gap > avg_char_width * 1.5:  # Порог для определения пробела
                    line_result += ' '
            line_result += match_char(char_img, templates)
            prev_right = x + w
        recognized += line_result + '\n'

    return recognized.strip()


def visualize_character_boxes(image_path, thresh_img, lines):
    img_color = cv2.imread(image_path)
    if img_color is None:
        print("Ошибка загрузки изображения!")
        return

    for line in lines:
        prev_right = None
        avg_width = np.mean([char_img.shape[1] for _, char_img in line])
        for x, char_img in line:
            h, w = char_img.shape
            top_left = (x, 0)  # пока без точного Y, можно доработать
            bottom_right = (x + w, h)
            cv2.rectangle(img_color, top_left, bottom_right, (0, 255, 0), 2)

            if prev_right is not None:
                gap = x - prev_right
                if gap > avg_width * 1.5:
                    cv2.line(img_color, (prev_right, h//2), (x, h//2), (0, 0, 255), 2)
                    cv2.putText(img_color, f'{gap:.0f}', (prev_right + 2, h//2 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            prev_right = x + w

    # Показываем изображение
    cv2.imshow("Character Boxes", img_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Запуск

image_path = 'input_image.jpg'
thresh = preprocess_image(image_path)
lines = extract_characters(thresh)
visualize_character_boxes(image_path, thresh, lines)

result = recognize_text('input_image.jpg')
print("Распознанный текст:", result)