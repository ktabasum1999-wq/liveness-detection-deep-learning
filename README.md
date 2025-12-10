# Face Anti-Spoofing (Liveness Detection) using MobileNetV2 + Focal Loss

A deep-learning–based **face liveness detection / anti-spoofing system** built using **MobileNetV2**, trained on a curated subset of **CelebA-Spoof**.  
The model achieves high accuracy and robustness against common spoof attacks such as **printed photos**, **replay attacks**, and **screen displays**.

This project was fully developed and trained on **Kaggle**, and includes full evaluation metrics, Grad-CAM visual explanations, and a deployable model.

---

## 🚀 Features

- **Real vs Spoof Classification** using MobileNetV2  
- **Focal Loss** for spoof-focused learning  
- **~99% Accuracy**, **High AUC**, **Low EER & ACER**  
- **Grad-CAM Heatmaps** for visual explanation  
- **CSV Predictions for reproducibility**  
- **Optional LBP + CNN Fusion (Texture + Deep Features)**  
- Fully reproducible Kaggle notebook  

---
## 📁 Project Structure
face-antispoofing/
│
├── README.md
├── face_antispoofing.ipynb # Main Kaggle notebook
│
├── model/
│ └── antispoof_big_mobilenet_best.h5 # Trained model
│
├── predictions/
│ └── predictions_test.csv # Test scores output
│
├── gradcam_samples/ # Visual explanations
│ ├── real_1.png
│ ├── real_2.png
│ ├── spoof_1.png
│ └── spoof_2.png
│
├── report/
│ └── project_report.md # Short technical summary
│
└── requirements.txt # Dependencies


---

## 📦 Dataset

### **CelebA Spoof Dataset (subset)**  
Used for:
- Training  
- Validation  
- Testing  

Classes:
- **Real (live)**  
- **Spoof (print, replay, display attacks)**  

Images were resized to **224×224**, normalized, and sampled evenly.

---

## 🏗 Model Architecture



MobileNetV2 (ImageNet pretrained)
→ GlobalAveragePooling2D
→ Dense(128, activation='relu')
→ Dropout(0.3)
→ Dense(1, activation='sigmoid')


### **Loss Function**
- Focal Loss (γ=2, α=0.25)

### **Optimizer**
- Adam (learning_rate = 1e-4)

### **Callbacks**
- EarlyStopping  
- ReduceLROnPlateau  
- ModelCheckpoint  

---

## 📊 Evaluation Metrics

The notebook computes:

- **Accuracy**
- **Precision, Recall, F1**
- **Confusion Matrix**
- **ROC Curve**
- **AUC Score**
- **EER (Equal Error Rate)**
- **ACER (Average Classification Error Rate)**

Example performance:

| Metric | Score |
|--------|-------|
| Accuracy | ~0.99 |
| AUC | ~0.997 |
| EER | Very Low |
| ACER | Very Low |

---

## 🔥 Grad-CAM Visualization

Visual heatmaps generated using the last conv layer (`out_relu`) show where the model focuses when detecting spoof vs real.

Upload sample images like:



gradcam_samples/
real_1.png
real_2.png
spoof_1.png
spoof_2.png


Real faces → highlighted on skin texture  
Spoof faces → highlighted on edges, glare, print noise  

---

## 🧪 How to Use the Model

```python
from tensorflow.keras.models import load_model
import cv2
import numpy as np

model = load_model("model/antispoof_big_mobilenet_best.h5", compile=False)

img = cv2.imread("test.png")
img = cv2.resize(img, (224, 224))
img = img[..., ::-1] / 255.0   # BGR → RGB
x = np.expand_dims(img, 0)

pred = model.predict(x)[0][0]

print("REAL FACE" if pred < 0.5 else "SPOOF FACE")

🛠 Installation

Install dependencies:

pip install -r requirements.txt


requirements.txt

tensorflow==2.18.0
opencv-python
numpy
pandas
matplotlib
seaborn
scikit-learn
tqdm
scipy

📝 Short Project Report

A short technical report is included in report/project_report.md, summarizing:

Problem definition

Dataset

Methodology

Metrics

Grad-CAM analysis

Conclusion

📜 License

Apache 2.0 (matches Kaggle licensing)

👤 Author

Tabasum Khan
AI & Deep Learning Enthusiast
Kaggle: https://www.kaggle.com/tabasumusman
