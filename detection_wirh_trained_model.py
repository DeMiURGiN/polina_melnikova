import easyocr
import os
from PIL import Image

reader = easyocr.Reader(['en'], recog_network='custom_example')
input_folder = 'image_vertical'  # Замените на актуальный путь
output_file_easy = 'results_easy_trained.txt'  # Файл для сохранения результатов

# Проверяем, что папка существует
if not os.path.exists(input_folder):
    print(f"Папка {input_folder} не найдена!")
    exit()

# Собираем список изображений в папке
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
images = [f for f in os.listdir(input_folder) if f.lower().endswith(image_extensions)]

if not images:
    print("В папке нет подходящих изображений!")
    exit()

# Открываем файл для записи результатов
with open(output_file_easy, 'w', encoding='utf-8') as f:
    for image_name in images:
        image_path = os.path.join(input_folder, image_name)
        try:
            # Открываем изображение
            img = Image.open(image_path)

            # Распознаём текст
            text = reader.readtext(img, detail=0, allowlist='АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ-0123456789*. ')

            # Записываем результаты в файл
            f.write(f"=== {image_name} ===\n")
            f.write("\n".join(text) + "\n\n")

            print(f"Обработано: {image_name}")
        except Exception as e:
            print(f"Ошибка при обработке {image_name}: {e}")


print(f"Готово! Результаты сохранены в {output_file_easy}")
