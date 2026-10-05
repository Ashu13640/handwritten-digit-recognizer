import streamlit as st
import numpy as np
import cv2
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="AI Handwritten Digit Recognizer",
    page_icon="🔢",
    layout="wide"
)


# ---------------------------------------------------------
# CSS
# ---------------------------------------------------------

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 700;
    text-align: center;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #777;
    margin-bottom: 30px;
}

.result-card {
    padding: 25px;
    border-radius: 15px;
    border: 1px solid #ddd;
    text-align: center;
}

.digit-result {
    font-size: 60px;
    font-weight: 700;
}

.confidence-result {
    font-size: 30px;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.markdown(
    '<div class="main-title">🔢 AI Handwritten Digit Recognizer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Draw or upload a handwritten digit and let a neural network recognize it.'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

@st.cache_resource
def load_model():

    model_path = (
        Path(__file__).parent
        / "model"
        / "handwritten_digit_model.keras"
    )

    return tf.keras.models.load_model(model_path)


model = load_model()


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:

    st.header("🧠 About the Model")

    st.write(
        "This application uses a trained neural network "
        "to classify handwritten digits from 0 to 9."
    )

    st.markdown("### Model Pipeline")

    st.write(
        """
        ✏️ Drawing / 📤 Upload
        ↓
        🖼️ Image Preprocessing
        ↓
        📐 28 × 28 Image
        ↓
        🧠 Neural Network
        ↓
        🎯 Prediction
        """
    )

    st.markdown("### Technologies")

    st.write(
        """
        • Python
        • TensorFlow / Keras
        • OpenCV
        • NumPy
        • Pillow
        • Streamlit
        • Matplotlib
        """
    )


# ---------------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------------

def preprocess_image(image):

    if len(image.shape) == 3:

        if image.shape[2] == 4:

            gray = cv2.cvtColor(
                image.astype(np.uint8),
                cv2.COLOR_RGBA2GRAY
            )

        else:

            gray = cv2.cvtColor(
                image.astype(np.uint8),
                cv2.COLOR_RGB2GRAY
            )

    else:

        gray = image.astype(np.uint8)


    # Detect whether background is light or dark
    mean_value = np.mean(gray)

    if mean_value > 127:

        _, threshold = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

    else:

        _, threshold = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )


    # Find digit
    coords = cv2.findNonZero(threshold)

    if coords is None:
        return None


    # Bounding box
    x, y, w, h = cv2.boundingRect(coords)

    digit = threshold[y:y + h, x:x + w]


    # Resize while preserving aspect ratio
    size = 20

    if w > h:

        new_width = size
        new_height = max(
            1,
            int(h * size / w)
        )

    else:

        new_height = size
        new_width = max(
            1,
            int(w * size / h)
        )


    digit = cv2.resize(
        digit,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )


    # Create 28 × 28 image
    final_image = np.zeros(
        (28, 28),
        dtype=np.uint8
    )


    # Center digit
    x_offset = (28 - new_width) // 2
    y_offset = (28 - new_height) // 2


    final_image[
        y_offset:y_offset + new_height,
        x_offset:x_offset + new_width
    ] = digit


    # Normalize
    return final_image.astype("float32") / 255.0


# ---------------------------------------------------------
# INPUT SELECTION
# ---------------------------------------------------------

st.subheader("📝 Choose Your Input")

input_method = st.radio(
    "Select how you want to provide the digit:",
    ["✏️ Draw Digit", "📤 Upload Image"],
    horizontal=True
)


# ---------------------------------------------------------
# DRAWING MODE
# ---------------------------------------------------------

if input_method == "✏️ Draw Digit":

    left_column, right_column = st.columns(2)

    with left_column:

        st.subheader("✏️ Draw Your Digit")

        st.write(
            "Use your mouse to draw a digit between **0 and 9**."
        )

        canvas_result = st_canvas(

            fill_color="black",
            stroke_width=18,
            stroke_color="white",
            background_color="black",

            height=320,
            width=320,

            drawing_mode="freedraw",

            key="canvas",

            update_streamlit=True,

            return_image_data=True
        )


        if st.button(
            "🧹 Clear Drawing",
            use_container_width=True
        ):

            st.rerun()


    if canvas_result.image_data is not None:

        processed = preprocess_image(
            canvas_result.image_data
        )

    else:

        processed = None


# ---------------------------------------------------------
# UPLOAD MODE
# ---------------------------------------------------------

else:

    left_column, right_column = st.columns(2)

    with left_column:

        st.subheader("📤 Upload Handwritten Digit")

        st.write(
            "Upload a PNG, JPG or JPEG image containing a handwritten digit."
        )

        uploaded_file = st.file_uploader(
            "Choose an image",
            type=["png", "jpg", "jpeg"]
        )

        processed = None

        if uploaded_file is not None:

            uploaded_image = Image.open(
                uploaded_file
            )

            st.image(
                uploaded_image,
                caption="Uploaded Image",
                width=300
            )

            image_array = np.array(
                uploaded_image
            )

            processed = preprocess_image(
                image_array
            )


# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------

if processed is not None:

    input_image = processed.reshape(
        1,
        28,
        28,
        1
    )


    prediction = model.predict(
        input_image,
        verbose=0
    )


    predicted_digit = int(
        np.argmax(prediction)
    )


    confidence = float(
        np.max(prediction) * 100
    )


    probabilities = prediction[0] * 100


    # -----------------------------------------------------
    # RESULT
    # -----------------------------------------------------

    with right_column:

        st.subheader("🎯 AI Prediction")

        st.markdown(
            f"""
            <div class="result-card">

                <div>Predicted Digit</div>

                <div class="digit-result">
                    {predicted_digit}
                </div>

                <div>Confidence</div>

                <div class="confidence-result">
                    {confidence:.2f}%
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        st.subheader("📊 Top 3 Predictions")

        top_3_indices = np.argsort(
            probabilities
        )[-3:][::-1]


        for rank, digit in enumerate(
            top_3_indices,
            start=1
        ):

            probability = float(
                probabilities[digit]
            )

            st.write(
                f"**#{rank} — Digit {digit}** "
                f"• {probability:.2f}%"
            )

            st.progress(
                probability / 100
            )


    # -----------------------------------------------------
    # ANALYSIS
    # -----------------------------------------------------

    st.divider()

    chart_column, image_column = st.columns(
        [1.5, 1]
    )


    with chart_column:

        st.subheader("📈 Prediction Probability")

        digits = np.arange(10)

        fig, ax = plt.subplots(
            figsize=(8, 4)
        )

        ax.bar(
            digits,
            probabilities
        )

        ax.set_xlabel("Digit")
        ax.set_ylabel("Probability (%)")

        ax.set_title(
            "Model Confidence Across All Digits"
        )

        ax.set_xticks(digits)

        ax.set_ylim(
            0,
            max(
                100,
                float(np.max(probabilities) + 5)
            )
        )

        st.pyplot(fig)

        plt.close(fig)


    with image_column:

        st.subheader("🖼️ Processed Image")

        processed_image = Image.fromarray(
            (processed * 255).astype(np.uint8)
        )

        st.image(
            processed_image,
            width=200,
            caption="28 × 28 image sent to the model"
        )


# ---------------------------------------------------------
# HOW IT WORKS
# ---------------------------------------------------------

st.divider()

st.subheader("💡 How It Works")

st.write(
    """
    The application accepts either a handwritten digit drawn
    on the canvas or an uploaded image.

    OpenCV converts the image to grayscale, detects the digit,
    crops it, preserves its aspect ratio, resizes it, centers
    it inside a 28 × 28 image, and normalizes the pixel values.

    The processed image is then passed to a trained
    TensorFlow/Keras neural network. The model generates
    probabilities for all ten digits and displays the
    predicted digit, confidence score, top-3 predictions,
    and probability distribution.
    """
)


st.caption(
    "Built with Python • TensorFlow • OpenCV • Streamlit"
)