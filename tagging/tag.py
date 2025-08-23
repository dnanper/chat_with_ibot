import onnxruntime as rt
from PIL import Image
import numpy as np
import csv
import os
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(__file__)) 
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.onnx")
TAGS_PATH = os.path.join(BASE_DIR, "model", "selected_tags.csv")
IMAGE_FOLDER = os.path.join(BASE_DIR, "dataset", "images_processed")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "dataset", "images_with_tags")
THRESHOLD = 0.35
TRIGGER_WORD = "ladycin_style"

tag_names = []
with open(TAGS_PATH, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        tag_names.append(row[1])

session = rt.InferenceSession(MODEL_PATH, providers=rt.get_available_providers())
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
model_height = session.get_inputs()[0].shape[2]

def preprocess_image(image):
    ratio = float(model_height) / max(image.size)
    new_size = tuple([int(x * ratio) for x in image.size])
    image = image.resize(new_size, Image.LANCZOS)
    new_im = Image.new("RGB", (model_height, model_height), "white")
    new_im.paste(image, ((model_height - new_size[0]) // 2, (model_height - new_size[1]) // 2))
    img_array = np.array(new_im, dtype=np.float32)
    # img_array = img_array.transpose((2, 0, 1))
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
i = 0
for filename in tqdm(image_files, desc="Processing dataset"):
    try:
        image_path = os.path.join(IMAGE_FOLDER, filename)
        img = Image.open(image_path).convert("RGB")
        filename = "image_" + str(i).zfill(4) + ".jpg"
        i += 1
        img.save(os.path.join(OUTPUT_FOLDER, filename))
        
        processed_img = preprocess_image(img)
        probs = session.run([output_name], {input_name: processed_img})[0][0]
        
        filtered_tags = [tag_names[i] for i, prob in enumerate(probs) if prob > THRESHOLD]
        
        final_caption = f"{TRIGGER_WORD}, " + ", ".join(filtered_tags)
        
        output_filename = os.path.splitext(filename)[0] + ".txt"
        with open(os.path.join(OUTPUT_FOLDER, output_filename), 'w', encoding='utf-8') as f:
            f.write(final_caption)
            
    except Exception as e:
        print(f"Lỗi xử lý file {image_path}: {e}")

print(f"Hoàn tất! Dataset đã được xử lý và lưu tại: {OUTPUT_FOLDER}")