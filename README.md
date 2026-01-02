Robust Object Detection 
Overview
Standard object detection models like YOLOv8 are trained on clear, high-quality images (like the COCO dataset). When these models encounter foggy or hazy weather, their accuracy drops significantly—a problem known as "Domain Shift." This project addresses that failure point. I have fine-tuned a YOLOv8 model specifically to detect objects in degraded visual environments (fog), ensuring reliable performance where standard models fail
How It Works (The AI Part)
The core of this project is Transfer Learning to bridge the gap between clear and foggy domains
The Problem: Fog reduces contrast and blurs edges, which are the primary features Convolutional Neural Networks (CNNs) use to identify objects. Standard models miss cars and pedestrians in these conditions because they haven't seen enough fog during training
The Solution: I used a pre-trained YOLOv8 model as a baseline and performed fine-tuning using physics based rendering (Beer Lamberts Law) on coco dataset. This process updates the model's weights to recognize features even when there are natural disturbances in the image
The Result: The model fog_final_best.pt has learned to "see through" the noise, maintaining high confidence scores for detections even in low-visibility scenarios.
Comparison Logic
To prove the improvement, I built a comparator script that runs a side-by-side analysis
It runs a standard,YOLO model on atest image to establish a baseline (often showing missed detections)
It runs my trained fog_final_best.pt model on the exact same input
Visualization: The script outputs a comparison where you can clearly see the custom model detecting objects that the standard model completely ignores.infact the model works good in foggy environment than in clear images i have solved this problem to an extent by inlcuding foggy as well as clear images in the dataset, it still continues but has improved a lot
Tech Stack
Framework: Ultralytics YOLOv8 (PyTorch).
Language: Python.
Computer Vision: OpenCV (cv2) for image processing and drawing bounding boxes.
Model format: PyTorch serialized weights (.pt).
File Structure
comparators.py: The main analysis script. It loads both the baseline model and my custom trained model, runs inference on input data, and generates a visual comparison of the performance.
fog_final_best.pt: The custom model weights. This file contains the parameters of the YOLOv8 neural network after being fine-tuned on the foggy dataset. It is the "brain" of the robust detector.
