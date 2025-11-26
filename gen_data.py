import numpy as np
import os
from PIL import Image
import json

os.makedirs("quickdraw_dataset/images", exist_ok=True)

classes = ["circle", "square", "triangle"]
dataset = []

def drawing_to_image(drawing):
    arr = np.array(drawing)
    # If flattened (1D) and length is a perfect square, reshape to square
    if arr.ndim == 1:
        length = arr.size
        side = int(np.sqrt(length))
        if side * side == length:
            arr = arr.reshape((side, side))
        else:
            raise ValueError(f"Can't reshape 1D array of length {length} into square image")

    # If floats in [0,1], scale to [0,255]
    if np.issubdtype(arr.dtype, np.floating):
        if arr.max() <= 1.0:
            arr = (arr * 255.0).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    else:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    img = Image.fromarray(arr)
    # Ensure RGB for compatibility with most tools
    if img.mode == 'L':
        img = img.convert('RGB')
    return img, arr

for cls in classes:
    data = np.load(f"data/{cls}.npy")  # скачанные файлы
    print(f"Loaded '{cls}' -> shape={data.shape}, dtype={data.dtype}")
    for i, drawing in enumerate(data[:2000]):  # возьми 2000 примеров
        try:
            img, arr = drawing_to_image(drawing)
        except Exception as e:
            print(f"Skipping {cls}_{i}: {e}")
            continue

        path = f"quickdraw_dataset/images/{cls}_{i}.png"
        img.save(path)

        # Print info for the first 3 examples of each class
        if i < 3:
            print(f"Saved example {i} for {cls}: shape={arr.shape}, min={arr.min()}, max={arr.max()}")

        dataset.append({
            "image": path,
            "text": f"a hand-drawn {cls}"
        })

# Save JSONL
with open("quickdraw_dataset/data.jsonl", "w") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")
