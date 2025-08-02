# 🛑 Smart Traffic Sign Recognition System

### 🔍 Overview

**Smart Traffic Sign Recognition System** is an AI-powered application that classifies traffic signs from uploaded images using a deep learning model. Built using **Streamlit** and **TensorFlow**, this system is ideal for autonomous vehicle systems, road safety monitoring, and smart traffic solutions.

---

### 📂 Dataset

The model is trained on the [German Traffic Sign Recognition Benchmark (GTSRB)](https://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset), a widely used dataset in traffic sign recognition research.

---

### ✨ Features

* ✅ **Deep Learning-Powered** – Built with Convolutional Neural Networks (CNNs) for high performance
* ✅ **High Accuracy** – Achieves 96.5% accuracy on test data
* ✅ **Streamlit UI** – Interactive, user-friendly web interface
* ✅ **Auto Preprocessing** – Grayscale conversion, histogram equalization, normalization
* ✅ **Instant Results** – Real-time predictions on uploaded images
* ✅ **Optimized Input Pipeline** – Uses OpenCV for efficient image processing
* ✅ **Lightweight & Fast** – Quick deployment without requiring a heavy backend

---

### 🛠️ Tech Stack

* **Frontend/UI:** Streamlit
* **Backend/Model:** TensorFlow, Keras
* **Image Processing:** OpenCV, NumPy, Pillow
* **Deployment Ready:** Can run locally with minimal setup

---

### 🚀 How It Works

1. **Upload** an image of a traffic sign using the file uploader.
2. The system **preprocesses** the image (grayscale, normalize, resize).
3. The trained CNN model **classifies** the image.
4. The **predicted sign name** is displayed instantly.

---

### 🧠 Model Info

* Architecture: Convolutional Neural Network (CNN)
* Input Size: 32x32 (grayscale)
* Layers: Conv2D, MaxPooling, Dropout, Dense
* Optimizer: Adam
* Loss Function: Categorical Crossentropy

---

### 📌 Future Improvements

* 🧭 Real-time webcam support
* 📱 Mobile app version
* 🔁 Retraining option with custom datasets
* 📈 Add prediction probability graph (bar chart)
