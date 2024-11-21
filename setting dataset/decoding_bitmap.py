import base64
import zlib
import json
import cv2
import numpy as np
import os

output_dir = "../screen foto/dataset 2024-04-21 14_33_36/masks"
os.makedirs(output_dir, exist_ok=True)

json_dir = "../screen foto/dataset 2024-04-21 14_33_36/clear_ann"

json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]

for json_filename in json_files:
    json_path = os.path.join(json_dir, json_filename)

    with open(json_path, 'r') as file:
        data = json.load(file)

    json_name = os.path.splitext(os.path.basename(json_filename))[0]

    for obj in data['objects']:
        if 'bitmap' in obj:
            base64_data = obj['bitmap']['data']
            decoded_data = base64.b64decode(base64_data)

            try:
                decompressed_data = zlib.decompress(decoded_data)
                print(f"Data decompressed successfully for object ID: {obj['id']}")

                image_array = np.frombuffer(decompressed_data, dtype=np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

                class_name = obj.get('classTitle', 'unknown_class').replace(' ', '_')

                output_filename = f"{json_name}_{class_name}_{obj['id']}.png"
                output_path = os.path.join(output_dir, output_filename)

                cv2.imwrite(output_path, image)
                print(f"Mask saved as: {output_path}")

                cv2.imshow(f'Decompressed Image - {obj["id"]}', image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            except Exception as e:
                print(f"Error decompressing the data for object ID {obj['id']}: {e}")
