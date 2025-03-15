import cv2
import numpy as np

# Загрузите модель YOLO
yolo_net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

# Загружаем имена классов
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Загружаем изображение
image = cv2.imread('image.jpg')

# Получаем размер изображения
height, width, channels = image.shape

# Подготовка изображения для YOLO
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

yolo_net.setInput(blob)

# Получаем выходные данные из сети
layer_names = yolo_net.getLayerNames()
output_layers = [layer_names[i - 1] for i in yolo_net.getUnconnectedOutLayers()]

# Прогоняем изображение через модель
outs = yolo_net.forward(output_layers)

# Обработка результатов
class_ids = []
confidences = []
boxes = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > 0.5:  # Можно регулировать порог уверенности
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Применение non-maxima suppression для удаления избыточных коробок
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Отображаем результаты
for i in range(len(boxes)):
    if i in indices:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = str(round(confidences[i], 2))

        # Рисуем прямоугольники и подписываем
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, f'{label} {confidence}', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Показываем итоговое изображение
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()