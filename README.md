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

### Run the Web Application

#### Prerequisites

- Python 3.10 or a compatible version
- Git
- pip

#### Setup

Clone the repository:

```bash
git clone https://github.com/madan-gm/Agro-Care-AI.git
cd Agro-Care-AI
```

Install the required Python dependencies using the project's requirements file, if available:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app/app.py
```

Open the local URL shown in the terminal.

> Note: The exact Python version and dependency installation steps should match the project's configuration.

---

## 📱 Mobile Application

Agro Care AI also includes a Flutter-based Android application with on-device crop disease detection, voice advisory, scan history, weather information, and an AI Farming Assistant.

- **Source code:** [Agro Care AI Mobile](https://github.com/madan-gm/Agro-Care-AI-Mobile)
- **Android APK:** [Download the application](https://drive.google.com/file/d/1jpXEmbG6UVM3VkoHHNWDCAt-jl1QiDc5/view?usp=sharing)

---

## ⚠️ Limitations

- Model predictions depend on input image quality and the training data.
- Reported test metrics are based on the PlantVillage test split and may not represent real-world field performance.
- Grad-CAM visualizations provide model explanations, not proof that a predicted disease is correct.
- Advisory content is general guidance and should be verified with qualified agricultural professionals.
- Weather and AI assistant features depend on their configured services and internet connectivity.

---

## 🔮 Future Improvements

- Evaluate the model on more diverse field images.
- Improve disease classification reliability under natural lighting and background conditions.
- Expand supported crops and disease classes.
- Improve personalized advisory and regional language support.
- Add downloadable scan reports and additional farming resources.

---

## 👨‍💻 Developer

**Madan G M**  
RNS Institute of Technology (RNSIT), Bengaluru

- **GitHub:** [madan-gm](https://github.com/madan-gm)
- **Mobile Repository:** [Agro Care AI Mobile](https://github.com/madan-gm/Agro-Care-AI-Mobile)

--- 
## 📑 Structural Designer
**Kusuma S R**
- **linkdin:** [Kusuma SR](https://www.linkedin.com/in/kusuma-sr-8b7614335/?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3BK649gVo6RKK4gml7aflUaw%3D%3D)

## 📄 License

No license has been specified yet.

## 🚀 Demo and Project Links

- **🎥 Demo Video:** [Watch Agro Care AI Demo](https://drive.google.com/file/d/14tPjAdihy7OFb9ot_jXO_WgMuBF_2K4j/view?usp=sharing)
- **📱 Android APK:** [Download Agro Care AI](https://drive.google.com/file/d/1jpXEmbG6UVM3VkoHHNWDCAt-jl1QiDc5/view?usp=sharing)
- **💻 Web Application Source:** This repository
- **📱 Mobile Application Source:** [Agro Care AI Mobile](https://github.com/madan-gm/Agro-Care-AI-Mobile)
