# Real-Time Emotion Detection Using CNN & OpenCV

> A real-time emotion recognition system that trains a Convolutional Neural Network (CNN) on facial-expression images (FER-2013) and performs live emotion detection with a webcam.

---

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [File / Folder Structure](#file--folder-structure)
- [Requirements](#requirements)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Real-time Detection](#real-time-detection)
- [Usage](#usage)
- [Monitoring & Visualization](#monitoring--visualization)
- [Troubleshooting & Tips](#troubleshooting--tips)
- [Future Improvements](#future-improvements)
- [Author](#author)
- [License](#license)

---

## Overview
This project builds and trains a CNN to classify seven facial emotions and uses the trained model for live predictions on webcam video. The pipeline:

1. Load and augment dataset (48×48 grayscale images).  
2. Train a CNN (Conv → Pool → Dense) with `categorical_crossentropy` loss and `Adam` optimizer.  
3. Save model (`emotion_detector_model.h5`).  
4. Load model and run inference on webcam frames using OpenCV’s Haar cascades for face detection.

---

## Dataset
**FER-2013 (Facial Expression Recognition 2013)** — grayscale 48×48 face images with 7 classes:

| Label Index | Emotion    |
|-------------|------------|
| 0           | Angry      |
| 1           | Disgust    |
| 2           | Fear       |
| 3           | Happy      |
| 4           | Sad        |
| 5           | Surprise   |
| 6           | Neutral    |

> Ensure your dataset folder is organized.

---

## File / Folder Structure
```
project-root/
├── archive/
│   ├── train/
│   │   ├── Angry/
│   │   ├── Disgust/
│   │   └── ...
│   └── test/
│       ├── Angry/
│       └── ...
├── train_model.py        # training script
├── emotion_live.py       # real-time detection script
├── requirements.txt
└── README.md
```

---

## Requirements

Install with pip:
```bash
pip install -r requirements.txt
```

`requirements.txt` example:
```
tensorflow>=2.6
opencv-python
numpy
matplotlib
```

---

## Model Architecture

```python
model = Sequential([
    Input(shape=(48, 48, 1)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(256, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])
```

**Compile:**
```python
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

- **Optimizer (Adam):** adaptive, usually fast & stable.  
- **Loss (Categorical Crossentropy):** appropriate for multi-class classification with softmax.  
- **Metric (Accuracy):** percent correctly classified.

---

## Training

**Sample training call:**
```python
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=25
)
model.save('emotion_detector_model.h5')
```

**Important notes**
- **Batch size:** updates weights after each batch.  
- **Validation:** `validation_generator` is used *only* to evaluate performance after each epoch (no weight updates).  
- **Augmentation:** use `ImageDataGenerator` to reduce overfitting (rotation, shift, zoom, flip).  
- **Monitor:** track `loss`, `val_loss`, `accuracy`, `val_accuracy` for underfitting/overfitting.

---

## Real-time Detection (Core steps)
1. Load saved model:
   ```python
   model = load_model('emotion_detector_model.h5')
   ```
2. Initialize face detector:
   ```python
   face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
   ```
3. Open webcam:
   ```python
   cap = cv2.VideoCapture(0)
   ```
4. For each frame:
   - Convert to grayscale.
   - Detect faces via `detectMultiScale`.
   - Crop, resize face to 48×48, normalize, expand dims.
   - `prediction = model.predict(roi)` → `label = emotion_labels[np.argmax(prediction)]`.
   - Display bounding box and label via `cv2.putText`.
5. Exit with `q`, then `cap.release()` and `cv2.destroyAllWindows()`.

---

## Usage

### Train
```bash
python train_model.py
```

### Run live detection
```bash
python emotion_live.py
```
Press **q** to quit the live window.

---

## Monitoring & Visualization

```python
import matplotlib.pyplot as plt

plt.figure()
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.title('Accuracy')
plt.legend()
plt.show()

plt.figure()
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Loss')
plt.legend()
plt.show()
```

---

## Troubleshooting & Tips
- **Low validation accuracy but high training accuracy → Overfitting**
  - Use stronger augmentation, dropout, more data, or reduce model complexity.
- **Both accuracies low → Underfitting**
  - Increase model capacity (more filters/layers), train longer, check data quality.
- **Training is unstable**
  - Lower learning rate: `Adam(learning_rate=1e-4)`; check normalization `rescale=1./255`.
- **Face detection misses faces**
  - Tune `detectMultiScale` parameters or use a more robust detector (dlib, MTCNN).
- **Speed**
  - Reduce input resolution or use a lighter model (MobileNet) for real-time on CPU.

---

## Collapsible: Quick Hyperparameters (click to expand)
<details>
<summary><strong>Hyperparameters used</strong></summary>

- Image size: `48 x 48` (grayscale)  
- Batch size: `64` (example)  
- Epochs: `25` (start)  
- Optimizer: `Adam` (default LR)  
- Loss: `categorical_crossentropy`  
- Dropout: `0.5` in FC layer

</details>

---

## Future Improvements
- Add multi-face tracking and smoother label smoothing over frames.  
- Making it in a fun game of making faces based on provided emoji

---

## Author
**Sachin Vardhan** — B.Tech | Deep Learning & Computer Vision Enthusiast  
GitHub: Sachin9vardhan

---

## License
This project is licensed