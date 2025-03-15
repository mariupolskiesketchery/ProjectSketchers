from flask import Flask, request, render_template, send_from_directory
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ExifTags
import io
import os
import time
import base64
import pytesseract
import requests

app = Flask(__name__)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Загрузка модели YOLO
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

YANDEX_SPELLER_API_KEY = "y0__xCpoPeCCBjD8xggp62hwxIkS9MzCdf6eJQRVRelG3BCwKaG9w"
YANDEX_SPELLER_URL = "https://speller.yandex.net/services/spellservice.json/checkText"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return "Ошибка: изображение не загружено"

    file = request.files['image']
    if file.filename == '':
        return "Ошибка: файл не выбран"

    img = Image.open(file)
    img_np = np.array(img)

    # Обнаружение объектов с использованием YOLO
    results = model(img_np)
    results.render()
    result_img = Image.fromarray(results.ims[0])

    # Сбор информации о найденных объектах
    detected_objects = []
    for det in results.xyxy[0]:
        class_id = int(det[5])
        confidence = float(det[4])
        object_name = model.names[class_id]
        detected_objects.append(f"{object_name} ({confidence:.2f})")

    # Если объектов не найдено
    if not detected_objects:
        detected_objects_message = "Объекты не найдены."
    else:
        detected_objects_message = f"Обнаруженные объекты: {', '.join(detected_objects)}"

    # Извлечение текста
    extracted_text = pytesseract.image_to_string(result_img, lang='rus')

    # Исправление текста с использованием Яндекс.Спеллера
    corrected_text = correct_text_with_yandex_speller(extracted_text)

    # Если текст был изменен
    if extracted_text.strip() != corrected_text.strip():
        text_change_message = f"Текст был изменен: '{extracted_text.strip()}' -> '{corrected_text.strip()}'"
    else:
        text_change_message = "Текст не был распознан или изменений не было."

    # Добавление исправленного текста к изображению
    result_img = overlay_text_on_image(result_img, corrected_text)

    # Сохранение изображения с метаданными
    output_metadata_folder = 'resultimage'
    os.makedirs(output_metadata_folder, exist_ok=True)

    timestamp = int(time.time())
    output_path = os.path.join(output_metadata_folder, f'corrected_image_{timestamp}.png')

    metadata = f"Detected Objects: {', '.join(detected_objects)}\nCorrected Text: {corrected_text}"
    add_metadata_to_image(result_img, metadata, output_path)

    # Подготовка изображений для отображения
    original_image_base64 = image_to_base64(img)
    corrected_image_base64 = image_to_base64(result_img)

    # Отправка сообщений в шаблон
    return render_template('result.html',
                           original_image=original_image_base64,
                           corrected_image=corrected_image_base64,
                           detected_objects_message=detected_objects_message,
                           text_change_message=text_change_message)

@app.route('/search', methods=['POST'])
def search_metadata():
    search_word = request.form.get('search_word', '').strip()
    if not search_word:
        return "Ошибка: введите слово для поиска"

    folder_path = 'resultimage'
    matching_images = search_images_by_metadata(folder_path, search_word)

    return render_template('search_results.html', search_word=search_word, images=matching_images)

# Маршрут для скачивания изображений из папки resultimage
@app.route('/resultimage/<filename>')
def download_file(filename):
    return send_from_directory('resultimage', filename)

def correct_text_with_yandex_speller(text):
    words = text.split()
    corrected_words = []

    for word in words:
        response = requests.post(YANDEX_SPELLER_URL, data={'text': word}, headers={'Authorization': f'Api-Key {YANDEX_SPELLER_API_KEY}'}).json()
        if response and 's' in response[0] and response[0]['s']:
            corrected_word = response[0]['s'][0]
        else:
            corrected_word = word
        corrected_words.append(corrected_word)

    return ' '.join(corrected_words)

def overlay_text_on_image(image, text):
    draw = ImageDraw.Draw(image)

    font_path = "C:\\Windows\\Fonts\\arial.ttf"
    font_size = 20
    font = ImageFont.truetype(font_path, font_size)

    draw.text((20, 20), text, font=font, fill=(255, 255, 255))
    return image

def add_metadata_to_image(image, metadata, output_path):
    exif = image.getexif()
    exif[0x9286] = metadata.encode('utf-8')
    image.save(output_path, exif=exif)

def image_to_base64(image):
    img_io = io.BytesIO()
    image.save(img_io, 'PNG')
    img_io.seek(0)
    return base64.b64encode(img_io.read()).decode('utf-8')

def search_images_by_metadata(folder_path, search_word):
    matching_images = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('png', 'jpg', 'jpeg')):
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path)
            exif_data = image.getexif()

            if 0x9286 in exif_data:
                metadata = exif_data[0x9286].decode('utf-8').lower()
                if search_word.lower() in metadata:
                    matching_images.append(filename)

    return matching_images

if __name__ == '__main__':
    app.run(debug=True)