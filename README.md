# 🎨 Gesture Canvas (Hand Gesture Drawing using MediaPipe + OpenCV)

This project lets users **draw on a virtual canvas using hand gestures**.  
It is built with **MediaPipe** for hand tracking and **OpenCV** for rendering drawings.  

---

## 🖐️ About MediaPipe Hand Tracking  

**MediaPipe** is a framework used for rapid prototyping of AI-powered perception pipelines.  
In this project, it is used to detect and track hand movements in real-time.  

MediaPipe offers **two main models** for hand detection:  

1. **Palm Detection Model** → Finds hands within an input image.  
2. **Hand Landmarks Detection Model** → Identifies 21 key landmark points of the detected hand.  

---

## 🔍 Hand Landmark Detection Model  

This model detects key points of a hand and provides:  
- Hand landmarks in **image coordinates**  
- Hand landmarks in **world coordinates**  
- The handedness (left or right hand)  

### ⚙️ Key Parameters  

- `static_image_mode` → If `False`, treats input as a continuous video stream.  
- `max_num_hands` → Maximum hands to detect (default = 2).  
- `model_complexity` → Accuracy level (0 or 1, default = 1).  
- `min_detection_confidence` → Minimum detection confidence (default = 0.5).  
- `min_tracking_confidence` → Minimum tracking confidence (default = 0.5).  

---

## 🖌️ Gesture Canvas Features  

The project provides a **drawing board controlled by hand gestures**.  

- A **top panel** allows selection of brushes (Blue, Orange, Purple, Red) and an **Eraser**.  
- The header updates based on the user’s selection.  
- Predefined images are used to visually show the active tool (brush color or eraser).  

### 🖍️ Brush & Eraser Settings  

- Brush thickness → `10`  
- Eraser thickness → `40`  
- These values are fixed (can only be changed manually in the code).  

---

## 🖐️ Gesture Controls  

Hand gestures are identified using the tips of the five fingers, stored in the list:  
fingers[] = [4, 8, 12, 16, 20]
<img width="470" height="168" alt="Screenshot 2025-09-02 225123" src="https://github.com/user-attachments/assets/5d47855f-0d93-4fbe-b0f2-a042e6c65b29" />


- **4 → Thumb tip**  
- **8 → Index finger tip**  
- **12 → Middle finger tip**  
- **16 → Ring finger tip**  
- **20 → Pinky finger tip**  

### ✋ Modes of Operation  

- **All five fingers up** → Clear the entire board.  
- **Index + Middle finger up** → **Selection mode** (choose brush color or eraser).  
- **Only Index finger up** → **Drawing mode** (draw on canvas).  

Drawing is handled using **OpenCV’s `cv2.line()`**, which connects finger positions frame by frame.  
The drawings are maintained on a separate canvas and then overlaid on the webcam feed.  

---

## 📷 Webcam Setup  

- Default resolution: **640x480**  
- Make sure to adjust resolution values in the code if your webcam uses different settings.  

---

## 🛠️ Tech Stack  

- **Python**  
- **MediaPipe** → Hand detection & landmark recognition  
- **OpenCV** → Drawing and real-time video processing  

---
this is the screenshot of canvas after running the code:
<img width="791" height="637" alt="Screenshot 2025-09-02 224950" src="https://github.com/user-attachments/assets/d2416007-f56e-42ff-85f5-68374188ef4e" />


