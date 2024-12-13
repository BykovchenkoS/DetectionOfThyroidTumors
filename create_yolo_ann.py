import base64
import zlib
import json
import cv2
import numpy as np
import os

IMG_DIR = "screen foto/dataset 2024-04-21 14_33_36/img"
JSON_DIR = "screen foto/dataset 2024-04-21 14_33_36/shifted_json"
YOLO1_DIR = "screen foto/dataset 2024-04-21 14_33_36/yolo1_ann"
YOLO2_DIR = "screen foto/dataset 2024-04-21 14_33_36/yolo2_ann"

os.makedirs(YOLO1_DIR, exist_ok=True)
os.makedirs(YOLO2_DIR, exist_ok=True)

def normalize_bbox(x_min, y_min, x_max, y_max, img_width, img_height):
    x_center = (x_min + x_max) / 2 / img_width
    y_center = (y_min + y_max) / 2 / img_height
    w = (x_max - x_min) / img_width
    h = (y_max - y_min) / img_height
    return x_center, y_center, w, h

json_files = [f for f in os.listdir(JSON_DIR) if f.endswith('.json')]

for json_filename in json_files:
    json_path = os.path.join(JSON_DIR, json_filename)

    with open(json_path, 'r') as file:
        data = json.load(file)

    img_name = os.path.splitext(json_filename)[0]
    img_path = os.path.join(IMG_DIR, img_name + ".jpg")

    img = cv2.imread(img_path)
    if img is None:
        print(f"Image not found: {img_path}")
        continue
    img_height, img_width = img.shape[:2]

    yolo1_annotations = []
    yolo2_annotations = []

    for obj in data['objects']:
        class_name = obj.get('classTitle', 'unknown_class')

        if obj['geometryType'] == 'polygon':
            points = obj['points']['exterior']
            if len(points) > 2:
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                x_min, y_min = min(x_coords), min(y_coords)
                x_max, y_max = max(x_coords), max(y_coords)

                bbox = normalize_bbox(x_min, y_min, x_max, y_max, img_width, img_height)

                if class_name in ["sagital", "longitudinal"]:
                    class_id = 0
                    yolo1_annotations.append(f"{class_id} " + " ".join(map(str, bbox)))

        elif obj['geometryType'] == 'bitmap' and 'bitmap' in obj:
            class_name = obj.get('classTitle', 'unknown_class')
            base64_data = obj['bitmap']['data']
            decoded_data = base64.b64decode(base64_data)

            try:
                decompressed_data = zlib.decompress(decoded_data)

                mask_array = np.frombuffer(decompressed_data, dtype=np.uint8)
                mask = cv2.imdecode(mask_array, cv2.IMREAD_GRAYSCALE)

                x_offset, y_offset = obj['bitmap']['origin']
                mask_height, mask_width = mask.shape[:2]
                x_min, y_min = x_offset, y_offset
                x_max, y_max = x_min + mask_width, y_min + mask_height

                bbox = normalize_bbox(x_min, y_min, x_max, y_max, img_width, img_height)

                if class_name == "Thyroid tissue":
                    class_id = 1  # Щитовидка
                    yolo1_annotations.append(f"{class_id} " + " ".join(map(str, bbox)))
                elif class_name == "Carotis":
                    class_id = 2  # Сонная артерия
                    yolo1_annotations.append(f"{class_id} " + " ".join(map(str, bbox)))
                elif class_name == "Node":
                    class_id = 0  # Узел
                    yolo2_annotations.append(f"{class_id} " + " ".join(map(str, bbox)))
                elif class_name == "Jugular":
                    class_id = 1  # Яремная вена
                    yolo2_annotations.append(f"{class_id} " + " ".join(map(str, bbox)))

            except Exception as e:
                print(f"Error decompressing bitmap for object {obj['id']}: {e}")

    yolo1_path = os.path.join(YOLO1_DIR, img_name + ".txt")
    with open(yolo1_path, "w") as f:
        f.write("\n".join(yolo1_annotations))
    print(f"Annotations for YOLO1 saved: {yolo1_path}")

    yolo2_path = os.path.join(YOLO2_DIR, img_name + ".txt")
    with open(yolo2_path, "w") as f:
        f.write("\n".join(yolo2_annotations))
    print(f"Annotations for YOLO2 saved: {yolo2_path}")
