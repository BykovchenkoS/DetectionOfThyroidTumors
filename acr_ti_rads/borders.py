import cv2
import numpy as np
from matplotlib import pyplot as plt
import os


def extract_boundary_pixels(mask):
    """
    Эта функция используется для извлечения пикселей, которые лежат на границе объекта (узла) в маске

     Эрозия (cv2.erode) :
        Эрозия — это морфологическая операция, которая "съедает" края объекта.
        Мы используем ядро размером (3, 3) (матрица из единиц) и выполняем эрозию один раз (iterations=1).
        Результатом эрозии является уменьшенная версия объекта, где его края сдвинуты внутрь.

    Граница объекта определяется как разница между исходной маской и эродированной маской
    Пиксели, которые были удалены при эрозии, остаются в boundary. Эти пиксели и являются границей объекта.
     """
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(mask, kernel, iterations=1)
    boundary = mask - eroded

    return boundary


def analyze_boundary_intensity(image, boundary):
    """
    Эта функция анализирует интенсивность пикселей на границе объекта

    Мы используем маску границы (boundary > 0) для выборки пикселей из исходного изображения.
    В результате получаем массив значений яркости всех пикселей, которые лежат на границе.

    Средняя интенсивность вычисляется как среднее значение всех пикселей на границе.

    Стандартное отклонение показывает, насколько значения яркости на границе варьируются вокруг среднего значения.
        Низкое значение указывает на однородную границу.
        Высокое значение говорит о большом разбросе яркости (неровная или шумная граница).
    """

    boundary_pixels = image[boundary > 0]

    mean_intensity = np.mean(boundary_pixels)  # отражает общую яркость границы
    std_intensity = np.std(boundary_pixels)

    return mean_intensity, std_intensity


def analyze_boundary_texture(image, boundary):
    """
    Эта функция анализирует текстуру границы объекта, используя градиент изображения

    Градиент изображения вычисляется с помощью оператора Собеля.
    gradient_x и gradient_y содержат изменения яркости вдоль соответствующих осей.

    Магнитуда градиента вычисляется как евклидова норма.
    Это значение показывает, насколько резко меняется яркость в каждой точке изображения.

    Средний градиент (mean_gradient) :
        Характеризует текстуру границы:
            Низкий градиент указывает на плавную границу.
            Высокий градиент говорит о резких изменениях яркости (неровная или бугристая граница).
    """
    # Вычисляем градиент изображения
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)

    # Извлекаем градиент на границе
    boundary_gradient = gradient_magnitude[boundary > 0]

    # Вычисляем средний градиент
    mean_gradient = np.mean(boundary_gradient)

    return mean_gradient


def analyze_contour_shape(mask):
    """
    Эта функция анализирует форму контура объекта, вычисляя компактность.

    Компактность — это числовая характеристика формы объекта, которая отражает, насколько форма объекта близка к
    идеальной окружности (или сфере в трехмерном случае). Чем ближе значение компактности к 1, тем более "круглой"
    или "компактной" является форма объекта.

    Используем функцию cv2.findContours, чтобы найти все контуры в маске.
    Из всех контуров выбирается самый большой (по площади).

    Компактность вычисляется как отношение квадрата периметра к площади, нормализованное множителем 4π:
        Чем ближе значение compactness к 1, тем более круглый или компактный контур.
        Высокое значение compactness указывает на сложную форму контура (например, бугристую).
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    compactness = (perimeter ** 2) / (4 * np.pi * area)
    return compactness


def determine_border_type_by_pixels(image, mask):
    boundary = extract_boundary_pixels(mask)

    mean_intensity, std_intensity = analyze_boundary_intensity(image, boundary)
    mean_gradient = analyze_boundary_texture(image, boundary)

    compactness = analyze_contour_shape(mask)

    print(f"std_intensity: {std_intensity}, mean_gradient: {mean_gradient}, compactness: {compactness}")

    if std_intensity < 30 and mean_gradient < 40 and compactness < 1.6:
        return "Неявная"
    elif std_intensity < 35 and mean_gradient < 80 and compactness < 1.8:
        return "Явная"
    elif std_intensity < 45 and mean_gradient < 100 and compactness < 2.0:
        return "Волнообразная (бугристая)"
    else:
        return "Выпячивание из железы"


def process_image_by_number(image_number):
    cropped_images_dir = '../dataset_coco_neuro_3/images_neuro_3'
    cropped_masks_dir = '../dataset_coco_neuro_3/masks'
    full_images_dir = '../dataset_coco_neuro_1/images_neuro_1'
    full_masks_dir = '../dataset_coco_neuro_1/masks'

    cropped_image_path = os.path.join(cropped_images_dir, f"{image_number}.jpg")
    cropped_mask_files = [f for f in os.listdir(cropped_masks_dir) if f.startswith(f"{image_number}_Node")]

    if not cropped_mask_files:
        print(f"Маска узла для изображения {image_number} не найдена.")
        return

    cropped_mask_path = os.path.join(cropped_masks_dir, cropped_mask_files[0])

    full_image_path = os.path.join(full_images_dir, f"{image_number}.jpg")
    full_mask_thyroid_files = [f for f in os.listdir(full_masks_dir) if f.startswith(f"{image_number}_Thyroid")]
    full_mask_carotis_files = [f for f in os.listdir(full_masks_dir) if f.startswith(f"{image_number}_Carotis")]

    if not full_mask_thyroid_files:
        print(f"Маска thyroid для изображения {image_number} не найдена.")
        return

    full_mask_thyroid_path = os.path.join(full_masks_dir, full_mask_thyroid_files[0])

    full_mask_carotis_path = None
    if full_mask_carotis_files:
        full_mask_carotis_path = os.path.join(full_masks_dir, full_mask_carotis_files[0])

    if not os.path.exists(cropped_image_path):
        print(f"Обрезанное изображение {cropped_image_path} не найдено.")
        return
    if not os.path.exists(cropped_mask_path):
        print(f"Маска узла {cropped_mask_path} не найдена.")
        return
    if not os.path.exists(full_image_path):
        print(f"Полное изображение {full_image_path} не найдено.")
        return
    if not os.path.exists(full_mask_thyroid_path):
        print(f"Маска thyroid {full_mask_thyroid_path} не найдена.")
        return

    cropped_image = cv2.imread(cropped_image_path, cv2.IMREAD_GRAYSCALE)
    cropped_mask = cv2.imread(cropped_mask_path, cv2.IMREAD_GRAYSCALE)

    full_image = cv2.imread(full_image_path, cv2.IMREAD_GRAYSCALE)
    full_mask_thyroid = cv2.imread(full_mask_thyroid_path, cv2.IMREAD_GRAYSCALE)
    full_mask_carotis = None
    if full_mask_carotis_path and os.path.exists(full_mask_carotis_path):
        full_mask_carotis = cv2.imread(full_mask_carotis_path, cv2.IMREAD_GRAYSCALE)

    border_type = determine_border_type_by_pixels(cropped_image, cropped_mask)

    node_in_thyroid = np.any(cv2.bitwise_and(cropped_mask, full_mask_thyroid))
    node_near_carotis = False
    if full_mask_carotis is not None:
        node_near_carotis = np.any(cv2.bitwise_and(cropped_mask, full_mask_carotis))

    print(f"Изображение {image_number}:")
    print(f"  Тип границы узла: {border_type}")
    print(f"  Узел находится внутри thyroid: {'Да' if node_in_thyroid else 'Нет'}")
    print(f"  Узел находится рядом с carotis: {'Да' if node_near_carotis else 'Нет (или маска carotis отсутствует)'}")

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(full_image, cmap='gray')

    plt.contour(full_mask_thyroid, colors='green', levels=[0.5])
    if full_mask_carotis is not None:
        plt.contour(full_mask_carotis, colors='blue', levels=[0.5])

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='green', lw=2, label='Thyroid'),
        Line2D([0], [0], color='blue', lw=2, label='Carotis') if full_mask_carotis is not None else None
    ]

    legend_elements = [elem for elem in legend_elements if elem is not None]
    plt.legend(handles=legend_elements, loc='upper right')

    plt.title(f"Исходное изображение {image_number}: Thyroid, Carotis")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cropped_image, cmap='gray')
    plt.contour(cropped_mask, colors='red', levels=[0.5])

    plt.legend(handles=[Line2D([0], [0], color='red', lw=2, label='Node')], loc='upper right')
    plt.title(f"Обрезанное изображение {image_number}: Узел")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


while True:
    user_input = input("Введите номер фото (или 'q' для выхода): ")

    if user_input.lower() == 'q':
        print("Программа завершена.")
        break

    if not user_input.isdigit():
        print("Пожалуйста, введите корректный номер фото.")
        continue

    image_number = int(user_input)
    process_image_by_number(image_number)
