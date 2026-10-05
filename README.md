@'
# Agro Care AI 🌱

## AI-Based Crop Disease Detection and Advisory System for Sustainable Farming

Agro Care AI is an AI-powered crop health application designed to help farmers identify crop diseases from leaf images and receive practical disease advisory in English and Kannada.

The system currently supports **Tomato, Potato, and Corn** and uses a fine-tuned **MobileNetV2** deep learning model for image classification.

The project is being developed with a real-world focus, with future improvements aimed at making crop disease detection more reliable under natural field conditions.

---

## 🌾 Current Features

- 📷 Crop disease detection from leaf images
- 🌱 Support for Tomato, Potato, and Corn
- 🤖 Fine-tuned MobileNetV2 deep learning model
- 📊 Confidence-based predictions
- 🔍 Grad-CAM visual explanation for model predictions
- 🩺 Disease advisory for supported diseases
- 🌐 English and Kannada advisory
- 💻 Streamlit web application
- 📱 Flutter Android application
- ⚡ TensorFlow Lite on-device inference
- 🌍 Real-world evaluation using field-style PlantDoc images

---

## 🧠 AI Model

The project compares a custom CNN baseline with MobileNetV2 transfer learning.

| Model | Test Accuracy |
|---|---:|
| Custom CNN | 87.27% |
| MobileNetV2 | 92.55% |
| Fine-tuned MobileNetV2 | **95.89%** |

The fine-tuned MobileNetV2 model is currently the final model used by the application.

### Evaluation Metrics

- Test Accuracy: **95.89%**
- Macro Precision: **94.88%**
- Macro Recall: **94.17%**
- Macro F1 Score: **94.35%**

These results are based on the PlantVillage test split. Real-world field performance is being evaluated separately because field images can differ significantly from controlled dataset images.

---

## 🌱 Supported Crops

### Tomato

- Bacterial Spot
- Early Blight
- Healthy
- Late Blight
- Leaf Mold
- Septoria Leaf Spot
- Target Spot
- Tomato Yellow Leaf Curl Virus

### Potato

- Early Blight
- Healthy
- Late Blight

### Corn

- Cercospora Leaf Spot / Gray Leaf Spot
- Common Rust
- Healthy
- Northern Leaf Blight

**Total: 15 classes**

---

## 🔍 Explainable AI

Agro Care AI uses **Grad-CAM (Gradient-weighted Class Activation Mapping)** to provide a visual explanation of the model prediction.

Grad-CAM highlights the regions of the leaf that contributed most strongly to the model's prediction.

This helps make the AI prediction more interpretable instead of providing only a disease name and confidence score.

---

## 🩺 Disease Advisory

After detecting a disease, the application provides:

- About the disease
- Treatment guidance
- Prevention guidance

Advisory information is available in:

- 🇬🇧 English
- 🇮🇳 Kannada

The advisory is intended as general guidance. Farmers should verify treatment decisions with a qualified agriculture officer, KVK, or other local agricultural expert.

---

## 📱 Applications

### Streamlit Web Application

The Python application provides:

- Leaf image upload
- AI disease prediction
- Confidence score
- Grad-CAM visualization
- Disease advisory
- English/Kannada language selection

Run the application with:

```bash
streamlit run app/app.py