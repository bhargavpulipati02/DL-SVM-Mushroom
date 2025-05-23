# 🍄 Mushroom Classification using CNN and CNN+SVM

## 📚 CS 6375 - Deep Learning Project 3  


This project explores image-based mushroom classification using:
1. A custom-built **Convolutional Neural Network (CNN)** with a softmax classifier.
2. A **CNN+SVM pipeline**, where CNN is used for feature extraction and **Support Vector Machine (SVM)** performs final classification.

---

## 🧠 Objective
To compare deep learning (CNN) vs hybrid (CNN+SVM) approaches on a 9-class mushroom dataset and evaluate their accuracy and generalization ability.

---

## 🗃️ Dataset
- 9 Mushroom Classes:  
  `Agaricus`, `Amanita`, `Boletus`, `Cortinarius`, `Entoloma`, `Hygrocybe`, `Lactarius`, `Russula`, `Suillus`
- Images are organized in folders, one per class.
- Format: `.jpg`, RGB

---

## 🧼 Data Preprocessing
- Images are decoded and resized to **224×224** pixels.
- Normalized to pixel values in `[0, 1]`.
- Split: **80% Training** / **20% Validation**, with a fixed seed for reproducibility.
- Augmentations (Training only):
  - Random rotation
  - Random zoom

---

## 🏗️ Model Architectures

### 1️⃣ CNN with Softmax Classifier
```text
Input: 224 × 224 × 3

→ Conv2D(32, 3×3) + ReLU  
→ MaxPooling  
→ Conv2D(64, 3×3) + ReLU  
→ MaxPooling  
→ Flatten  
→ Dense(128) + ReLU  
→ Dense(9) + Softmax
