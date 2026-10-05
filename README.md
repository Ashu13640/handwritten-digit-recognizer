# 🔢 AI Handwritten Digit Recognizer

A Streamlit-based AI application that recognizes handwritten digits from **0 to 9** using a trained **TensorFlow/Keras neural network**.

The application allows users to draw a digit on an interactive canvas and instantly displays the model's prediction, confidence score, top-3 predictions, and probability distribution.

## 🚀 Features

* ✏️ Interactive handwritten digit drawing canvas
* 🧠 TensorFlow/Keras neural network prediction
* 🖼️ Automatic image preprocessing using OpenCV
* 📐 Converts drawings into standardized **28 × 28** images
* 🎯 Predicted digit with confidence score
* 📊 Top-3 predictions
* 📈 Probability distribution for all 10 digits
* 🖼️ Displays the processed image sent to the model
* 🌐 Interactive Streamlit web interface

## 🛠️ Tech Stack

* **Python**
* **TensorFlow / Keras**
* **OpenCV**
* **NumPy**
* **Pillow**
* **Matplotlib**
* **Streamlit**
* **Streamlit Drawable Canvas**

## 🔄 How It Works

```text
User Drawing
     ↓
Canvas Image
     ↓
Grayscale Conversion
     ↓
Thresholding
     ↓
Digit Detection & Cropping
     ↓
Aspect-Ratio Preserving Resize
     ↓
Centering on 28 × 28 Image
     ↓
Pixel Normalization
     ↓
TensorFlow/Keras Model
     ↓
Prediction Probabilities
     ↓
Digit + Confidence + Top 3 Results
```

## 🧠 Image Preprocessing

The application performs several preprocessing operations before sending the image to the neural network:

1. Converts the RGBA canvas image to grayscale.
2. Applies thresholding to isolate the handwritten digit.
3. Detects the digit's bounding box.
4. Crops the digit from the canvas.
5. Resizes it while preserving its aspect ratio.
6. Places the digit in the center of a **28 × 28** image.
7. Normalizes pixel values between `0` and `1`.
8. Sends the processed image to the trained model.

## 📁 Project Structure

```text
handwritten-digit-recognizer/
│
├── app.py
├── requirements.txt
├── .gitignore
│
├── model/
│   └── handwritten_digit_model.keras
│
├── .venv/
└── venv312/
```

Virtual environments are excluded from GitHub using `.gitignore`.

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd handwritten-digit-recognizer
```

### 2. Create a Python virtual environment

Python **3.12** is recommended for this project.

```bash
py -3.12 -m venv .venv
```

### 3. Install dependencies

```bash
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 4. Run the application

```bash
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The application will open in your browser.

## 🎮 Usage

1. Open the Streamlit application.
2. Draw a digit from **0 to 9** on the canvas.
3. The application preprocesses your drawing.
4. The neural network analyzes the image.
5. View:

   * Predicted digit
   * Confidence percentage
   * Top-3 predictions
   * Probability distribution
   * Processed 28 × 28 image

## 🔮 Future Improvements

* Improve model accuracy with additional training and augmentation.
* Add support for uploading handwritten digit images.
* Add prediction history.
* Add a model performance dashboard.
* Improve mobile responsiveness.
* Deploy the application publicly.
* Add support for batch digit recognition.

## 👨‍💻 Author

**Rajeet Pratap Mall**

B.Tech — Computer Science & Engineering (Artificial Intelligence & Machine Learning)

GitHub: **Rajeet495**

---

⭐ If you found this project useful, consider giving the repository a star!
