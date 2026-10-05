import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


# --------------------------------------------------
# Paths
# --------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent

MODEL_PATH = PROJECT_DIR / "models" / "mobilenetv2_finetuned_best.keras"
CLASSES_PATH = APP_DIR / "classes.json"
ADVISORY_PATH = APP_DIR / "advisory.json"


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Agro Care AI",
    page_icon="🌱",
    layout="centered"
)


# --------------------------------------------------
# Load model and data
# --------------------------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_classes():
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_advisory():
    with open(ADVISORY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


model = load_model()
classes = load_classes()
advisory = load_advisory()

# MobileNetV2 backbone for Grad-CAM
mobilenet_backbone = model.get_layer(
    "mobilenetv2_1.00_224"
)


# --------------------------------------------------
# Grad-CAM
# --------------------------------------------------

def make_gradcam_heatmap(
    img_array,
    model,
    backbone,
    last_conv_layer_name,
    pred_index
):
    # Create a model that returns:
    # 1. MobileNetV2 convolutional feature maps
    # 2. MobileNetV2 backbone output
    grad_model = tf.keras.models.Model(
        inputs=backbone.inputs,
        outputs=[
            backbone.get_layer(last_conv_layer_name).output,
            backbone.output
        ]
    )

    # Apply the same preprocessing used during training
    backbone_input = model.get_layer(
        "data_augmentation"
    )(img_array)

    backbone_input = tf.keras.applications.mobilenet_v2.preprocess_input(
        backbone_input
    )

    with tf.GradientTape() as tape:

        conv_outputs, backbone_features = grad_model(
            backbone_input,
            training=False
        )

        # Classification head
        pooled_features = model.get_layer(
            "global_average_pooling2d"
        )(backbone_features)

        dropout_output = model.get_layer(
            "dropout"
        )(pooled_features, training=False)

        predictions = model.get_layer(
            "dense"
        )(dropout_output)

        class_channel = predictions[:, pred_index]

    # Calculate gradients
    grads = tape.gradient(
        class_channel,
        conv_outputs
    )

    # Average gradients over spatial dimensions
    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    # Remove batch dimension
    conv_outputs = conv_outputs[0]

    # Weight feature maps
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

    heatmap = tf.squeeze(heatmap)

    # Keep positive influence
    heatmap = tf.maximum(
        heatmap,
        0
    )

    # Normalize
    max_value = tf.reduce_max(heatmap)

    heatmap = tf.where(
        max_value > 0,
        heatmap / max_value,
        heatmap
    )

    return heatmap.numpy()

LAST_CONV_LAYER = "out_relu"


# --------------------------------------------------
# UI
# --------------------------------------------------

st.title("🌱 Agro Care AI")
st.subheader("AI-Based Crop Disease Detection and Advisory System")

st.write(
    "Upload a crop leaf image to identify the possible disease "
    "and receive an advisory."
)


# Language selection
language = st.selectbox(
    "Select Advisory Language",
    ["English", "ಕನ್ನಡ"]
)

language_code = "en" if language == "English" else "kn"


# Image uploader
uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Leaf Image",
        use_container_width=True
    )

    # Resize image
    resized_image = image.resize((224, 224))

    # Convert to NumPy array
    image_array = np.array(
        resized_image,
        dtype=np.float32
    )

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # Model prediction
    predictions = model.predict(
        image_array,
        verbose=0
    )

    predicted_index = int(
        np.argmax(predictions[0])
    )

    confidence = float(
        predictions[0][predicted_index]
    )

    predicted_class = classes[predicted_index]

    # Generate Grad-CAM heatmap
    heatmap = make_gradcam_heatmap(
        image_array,
        model,
        mobilenet_backbone,
        LAST_CONV_LAYER,
        predicted_index
    )

    # --------------------------------------------------
    # Confidence threshold
    # --------------------------------------------------

    if confidence < 0.70:

        st.warning(
            f"⚠️ Low confidence prediction "
            f"({confidence:.2%})."
        )

        st.info(
            "Please upload a clearer leaf image or consult "
            "a local agriculture officer."
        )

    else:

        # Convert class name to readable format
        display_name = predicted_class.replace(
            "___", " — "
        ).replace(
            "_", " "
        )

        st.success(
            f"Prediction: {display_name}"
        )

        st.metric(
            "Confidence",
            f"{confidence:.2%}"
        )

        # --------------------------------------------------
        # Grad-CAM Visualization
        # --------------------------------------------------

        heatmap_uint8 = np.uint8(255 * heatmap)

        heatmap_resized = cv2.resize(
            heatmap_uint8,
            (image.width, image.height)
        )

        heatmap_colored = cv2.applyColorMap(
            heatmap_resized,
            cv2.COLORMAP_JET
        )

        original_image = np.array(image)

        original_image_bgr = cv2.cvtColor(
            original_image,
            cv2.COLOR_RGB2BGR
        )

        overlay = cv2.addWeighted(
            original_image_bgr,
            0.6,
            heatmap_colored,
            0.4,
            0
        )

        overlay_rgb = cv2.cvtColor(
            overlay,
            cv2.COLOR_BGR2RGB
        )

        st.markdown("### 🔍 Grad-CAM Explanation")

        st.image(
            overlay_rgb,
            caption="Regions that contributed to the model prediction",
            use_container_width=True
        )


        # --------------------------------------------------
        # Advisory
        # --------------------------------------------------

        if predicted_class in advisory:

            advice = advisory[predicted_class][
                language_code
            ]

            st.markdown("### 📋 Advisory")

            st.markdown(
                f"**About:** {advice['about']}"
            )

            st.markdown(
                f"**Treatment:** {advice['treatment']}"
            )

            st.markdown(
                f"**Prevention:** {advice['prevention']}"
            )

            if language_code == "en":
                st.info(
                    "For field-specific treatment decisions, "
                    "consult your local agriculture officer."
                )
            else:
                st.info(
                    "ನಿರ್ದಿಷ್ಟ ಹೊಲದ ಚಿಕಿತ್ಸಾ ನಿರ್ಧಾರಗಳಿಗಾಗಿ "
                    "ನಿಮ್ಮ ಸ್ಥಳೀಯ ಕೃಷಿ ಅಧಿಕಾರಿಯನ್ನು ಸಂಪರ್ಕಿಸಿ."
    )

        else:

            st.warning(
                "Advisory information is not available "
                "for this prediction."
            )