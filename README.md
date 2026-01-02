Robust Object Detection (YOLOv8)
This project focuses on detecting objects in "degraded environments" (like foggy, blurry, or low-light images) where standard models usually fail
The Problem: Most object detection models work perfectly on clear images. But when you test them on a blurry CCTV feed or a foggy road, they stop detecting things. This is called Domain Shift
My Solution: I used YOLOv8 and fine-tuned it to handle these specific conditions. The goal is to make a model that is robust enough to see cars, people, and objects even when the image quality is low
Challenges Faced:
One of the problem is when I trained the model to detect better in foggy or blurry images, it stopped detecting objects in the normal, clear environment.It was like the model got confused—as soon as it got better at the "hard" images, it got worse at the "easy" ones. I had to balance it to work for both types
Tech Stack:
Model: YOLOv8 (Ultralytics)
Language: Python
Libraries: OpenCV, PyTorch, NumPy
Requirements :pip install ultralytics opencv-python
Future Plans:
Optimize the code to run faster on live video.
Update the code for Domain shift like the day and nightlight for which i should work on datasets 
and also deal with motion shift 
