from ultralytics import YOLO
import os
import cv2, torch
from flask import Flask, flash, render_template, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from flask import send_from_directory
from PIL import Image, ExifTags

app = Flask(__name__)
# Add a secret key for flash messages
app.secret_key = "your_secret_key_here"

model = YOLO('yolov8m.pt')

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['DETECTED'] = 'detected'
app.config['METADATA'] = 'metadata'

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DETECTED'], exist_ok=True)
os.makedirs(app.config['METADATA'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def process_image(image_path):
    """Process an image and return detection results, save metadata and detected image"""
    # Get base filename
    base_filename = os.path.basename(image_path)
    
    # Загружаем картинку
    im = cv2.imread(image_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  # Переводим в RGB
    im = cv2.resize(im, (640, 640))  # Изменяем размер картинки под модель
    
    im2 = torch.tensor(im).permute(2, 0, 1)  # Переводим в тензор и меняем размерность
    im2 = im2.float() / im2.max().float()  # Нормализуем
    im2 = im2.unsqueeze(0)  # Добавляем размерность
    results = model.predict(im2)  # Делаем предсказание
    
    # Создаем список для хранения метаданных
    metadata = []
    for i, cl, xyxy in zip(range(100), results[0].boxes.cls, results[0].boxes.xyxy):
        c = int(cl.item())  # Номер класса объекта
        name = results[0].names[c]  # Название класса объекта
        xy = xyxy.tolist()  # Координаты объекта
        metadata.append(f"Объект {i}: {name} в координатах {xy}")
    
    # Сохраняем метаданные в изображение
    image = Image.fromarray(im)
    exif = image.getexif()
    if exif is None:
        exif = {}
    exif[0x9286] = "\n".join(metadata).encode('utf-8')  # Добавляем метаданные в EXIF
    
    # Сохраняем изображение с метаданными
    metadata_output_path = os.path.join(app.config['METADATA'], base_filename)
    image.save(metadata_output_path, exif=exif)
    
    # Сохраняем изображение с обнаруженными объектами
    detected_output_path = os.path.join(app.config['DETECTED'], base_filename)
    cv2.imwrite(detected_output_path, cv2.cvtColor(results[0].plot(), cv2.COLOR_RGB2BGR))
    
    return results[0].verbose(), metadata

def get_exif_data(image_path):
    """Получает EXIF-данные из изображения."""
    image = Image.open(image_path)
    exif_data = image.getexif()
    if exif_data is None:
        return {}
    return exif_data

def search_images_by_metadata(folder_path, search_word):
    """Ищет изображения в папке по указанным метаданным."""
    matching_images = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(tuple(ALLOWED_EXTENSIONS)):
            image_path = os.path.join(folder_path, filename)
            exif_data = get_exif_data(image_path)
            # Проверяем, есть ли метаданные в EXIF
            if 0x9286 in exif_data:
                metadata = exif_data[0x9286].decode('utf-8').lower()
                if search_word.lower() in metadata:
                    matching_images.append(filename)
    return matching_images

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            p = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            file.save(p)
            return redirect(url_for('detect', name=filename))
    return render_template('upload.html')

@app.route('/detect/<name>')
def detect(name):
    p = os.path.join(app.config['UPLOAD_FOLDER'], name)
    text, metadata = process_image(p)
    return render_template('detect.html', name=name, text=text, metadata=metadata)

@app.route('/uploads/<name>')
def download_file(name):
    return send_from_directory(app.config["UPLOAD_FOLDER"], name)

@app.route('/detected/<name>')
def download_detected_file(name):
    return send_from_directory(app.config["DETECTED"], name)

@app.route('/metadata/<name>')
def download_metadata_file(name):
    return send_from_directory(app.config["METADATA"], name)

@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        search_word = request.form.get('search_word', '')
        if search_word:
            matching_images = search_images_by_metadata(app.config['METADATA'], search_word)
            return render_template('search_results.html', images=matching_images, search_word=search_word)
    return render_template('search.html')

@app.route('/api/search', methods=['POST'])
def api_search():
    search_word = request.json.get('search_word', '')
    if search_word:
        matching_images = search_images_by_metadata(app.config['METADATA'], search_word)
        return jsonify({'images': matching_images})
    return jsonify({'error': 'No search word provided'}), 400

@app.route('/batch_process', methods=['GET', 'POST'])
def batch_process():
    if request.method == 'POST':
        # Process all images in the uploads folder
        processed_files = []
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            if filename.lower().endswith(tuple(ALLOWED_EXTENSIONS)):
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                process_image(image_path)
                processed_files.append(filename)
        return render_template('batch_results.html', files=processed_files)
    return render_template('batch_process.html')

if __name__ == '__main__':
    app.run(debug=True)
