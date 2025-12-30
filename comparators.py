import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os
# ---THE PHYSICS PART ---
# To apply fog layers
def add_realistic_fog(image, beta=4.5): 
    img_f = image.astype(np.float32) / 255.0
    (h, w) = img_f.shape[:2]
    A = 0.9 # Atmospheric Light (White)
    
    # --- FIXED DEPTH MAP ---
    # Top of image (row 0) = Far = 1.0
    # Bottom of image (row h) = Close = 0.0
    # We use 'linspace' to create a perfect gradient from 1 down to 0
    # Assuming far things the fog is heavy and nearby things it is lighter
    row_indices = np.linspace(1.0, 0.0, h) 
    depth_map = np.tile(row_indices.reshape(h, 1), (1, w))
    
    # Beer-Lambert: Fog gets thicker as depth increases
    transmission = np.exp(-beta * depth_map)
    transmission = transmission[:, :, np.newaxis]
    
    foggy_image = img_f * transmission + A * (1 - transmission)
    return np.clip(foggy_image * 255, 0, 255).astype(np.uint8)

image_path = "fog_image.png"
custom_model_path = "fog_final_best.pt" 

og_img = cv2.imread(image_path)
if og_img is None:
    print("Error: Upload 'traffic.png' first")
    exit()
foggy_img = add_realistic_fog(og_img, beta=3.0)

# Load Models
model_std = YOLO('yolov8n.pt') 
if os.path.exists(custom_model_path):
    model_custom = YOLO(custom_model_path)
else:
    model_custom = model_std

# Filter for Cars(2) and Trucks(7) from COCO dataset
target_classes =[2,7] 

# Run Detection
res_std = model_std(foggy_img, verbose=False,conf=0.10,classes=target_classes)
res_custom = model_custom(foggy_img, verbose=False,conf=0.10,classes=target_classes)

# Visualization
img_std_rgb = cv2.cvtColor(res_std[0].plot(),cv2.COLOR_BGR2RGB)
img_custom_rgb = cv2.cvtColor(res_custom[0].plot(),cv2.COLOR_BGR2RGB)

count_std = len(res_std[0].boxes)
count_custom = len(res_custom[0].boxes)

plt.figure(figsize=(16, 8))

plt.subplot(1, 2, 1)
plt.imshow(img_std_rgb)
plt.title(f"Standard Model:{count_std}",fontsize=16,color='red')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img_custom_rgb)
plt.title(f"Custom Model:{count_custom}",fontsize=16,color='green')
plt.axis('off')

plt.tight_layout()
plt.show()

print(f"Standard:{count_std}")

print(f"Custom:{count_custom}")
