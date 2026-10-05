# ✍️ Handwritten Digit Recognizer

An interactive deep-learning application that recognizes handwritten digits (0–9) using a trained MNIST-based neural network.

## 🚀 Live Demo

👉 **[Try the Handwritten Digit Recognizer](https://handwritten-digit-recognizer-smyfrwj9qdtodhqbc7t3zn.streamlit.app/)**

## 💻 GitHub Repository

👉 **[View Source Code](https://github.com/Ashu13640/handwritten-digit-recognizer)**

---

## 📌 Project Overview

This project uses a trained deep-learning model to recognize handwritten digits from either:

- ✏️ A digit drawn directly on the canvas
- 📤 An uploaded handwritten digit image

The application preprocesses the input image, converts it into the format expected by the model, and returns the predicted digit along with the top-3 probability scores.

## ✨ Features

- ✏️ Draw handwritten digits using an interactive canvas
- 📤 Upload handwritten digit images
- 🧠 Deep-learning based digit classification
- 🔢 Recognition of digits from 0–9
- 📊 Top-3 prediction probabilities
- 🖼️ Visualized processed 28×28 input
- ⚡ Interactive Streamlit interface
- ☁️ Deployed on Streamlit Community Cloud

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- Streamlit
- OpenCV
- NumPy
- Pillow
- Matplotlib
- Streamlit Drawable Canvas

## 🔄 Workflow

```text
User Input
   ↓
Draw / Upload Image
   ↓
Image Preprocessing
   ↓
Grayscale Conversion
   ↓
Thresholding
   ↓
Digit Detection & Cropping
   ↓
Resize to 28×28
   ↓
Normalization
   ↓
Trained Neural Network
   ↓
Prediction
   ↓
Top-3 Probabilities