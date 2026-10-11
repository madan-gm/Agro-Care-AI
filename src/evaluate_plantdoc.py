import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from PIL import Image


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "mobilenetv2_finetuned_best.keras"
CLASSES_PATH = PROJECT_ROOT / "app" / "classes.json"
DATASET_PATH = PROJECT_ROOT / "data" / "real_world"

REPORTS_PATH = PROJECT_ROOT / "reports"
REPORTS_PATH.mkdir(parents=True, exist_ok=True)

RESULTS_PATH = REPORTS_PATH / "plantdoc_results.json"
PREDICTIONS_PATH = REPORTS_PATH / "plantdoc_predictions.csv"
CONFUSION_MATRIX_PATH = REPORTS_PATH / "plantdoc_confusion_matrix.png"


# ---------------------------------------------------------
# PlantDoc class mapping
# ---------------------------------------------------------

PLANTDOC_TO_MODEL = {
    "Corn___Cercospora_leaf_spot_Gray_leaf_spot":
        "Corn___Cercospora_leaf_spot Gray_leaf_spot",

    "Corn___Common_rust":
        "Corn___Common_rust",

    "Corn___Northern_Leaf_Blight":
        "Corn___Northern_Leaf_Blight",

    "Potato___Early_blight":
        "Potato___Early_blight",

    "Potato___Late_blight":
        "Potato___Late_blight",

    "Tomato___Bacterial_spot":
        "Tomato___Bacterial_spot",

    "Tomato___Early_blight":
        "Tomato___Early_blight",

    "Tomato___Late_blight":
        "Tomato___Late_blight",

    "Tomato___Septoria_leaf_spot":
        "Tomato___Septoria_leaf_spot",

    "Tomato___Tomato_Yellow_Leaf_Curl_Virus":
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
}


IMAGE_SIZE = (224, 224)


# ---------------------------------------------------------
# Load classes
# ---------------------------------------------------------

with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    classes = json.load(f)

class_to_index = {
    name: index
    for index, name in enumerate(classes)
}


# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------

print("\nLoading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")
print(f"Number of classes: {len(classes)}")


# ---------------------------------------------------------
# Collect images
# ---------------------------------------------------------

image_paths = []
true_labels = []

for folder_name, model_class in PLANTDOC_TO_MODEL.items():

    folder_path = DATASET_PATH / folder_name

    if not folder_path.exists():
        print(f"WARNING: Missing folder: {folder_path}")
        continue

    true_index = class_to_index[model_class]

    for image_path in folder_path.iterdir():

        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp",
        }:
            continue

        image_paths.append(image_path)
        true_labels.append(true_index)


print(f"\nTotal PlantDoc images: {len(image_paths)}")


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

predicted_labels = []
predicted_confidences = []

records = []

print("\nRunning predictions...")

for i, (image_path, true_index) in enumerate(
    zip(image_paths, true_labels),
    start=1,
):

    try:

        image = Image.open(image_path).convert("RGB")
        image = image.resize(IMAGE_SIZE)

        # IMPORTANT:
        # Do NOT apply MobileNetV2 preprocess_input here.
        # The trained model already contains its preprocessing
        # pipeline internally.

        image_array = np.asarray(
            image,
            dtype=np.float32,
        )

        image_array = np.expand_dims(
            image_array,
            axis=0,
        )

        probabilities = model.predict(
            image_array,
            verbose=0,
        )[0]

        predicted_index = int(
            np.argmax(probabilities)
        )

        confidence = float(
            probabilities[predicted_index]
        )

        predicted_labels.append(predicted_index)
        predicted_confidences.append(confidence)

        records.append(
            {
                "image": str(
                    image_path.relative_to(PROJECT_ROOT)
                ),
                "true_class": classes[true_index],
                "predicted_class": classes[predicted_index],
                "confidence": confidence,
                "correct": predicted_index == true_index,
            }
        )

        if i % 10 == 0 or i == len(image_paths):
            print(
                f"Processed {i}/{len(image_paths)}"
            )

    except Exception as e:

        print(
            f"ERROR processing {image_path}: {e}"
        )


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

accuracy = accuracy_score(
    true_labels,
    predicted_labels,
)

precision = precision_score(
    true_labels,
    predicted_labels,
    average="macro",
    zero_division=0,
)

recall = recall_score(
    true_labels,
    predicted_labels,
    average="macro",
    zero_division=0,
)

f1 = f1_score(
    true_labels,
    predicted_labels,
    average="macro",
    zero_division=0,
)


# ---------------------------------------------------------
# Classification report
# ---------------------------------------------------------

used_indices = sorted(
    set(true_labels)
)

target_names = [
    classes[index]
    for index in used_indices
]

report = classification_report(
    true_labels,
    predicted_labels,
    labels=used_indices,
    target_names=target_names,
    output_dict=True,
    zero_division=0,
)


# ---------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------

cm = confusion_matrix(
    true_labels,
    predicted_labels,
    labels=used_indices,
)

fig, ax = plt.subplots(
    figsize=(14, 12)
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=target_names,
)

disp.plot(
    ax=ax,
    xticks_rotation=90,
    values_format="d",
    colorbar=False,
)

plt.title(
    "Agro Care AI - PlantDoc Real-World Evaluation"
)

plt.tight_layout()

plt.savefig(
    CONFUSION_MATRIX_PATH,
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# ---------------------------------------------------------
# Save predictions
# ---------------------------------------------------------

predictions_df = pd.DataFrame(
    records
)

predictions_df.to_csv(
    PREDICTIONS_PATH,
    index=False,
)


# ---------------------------------------------------------
# Confidence analysis
# ---------------------------------------------------------

mean_confidence = float(
    np.mean(predicted_confidences)
)

correct_mask = (
    np.array(predicted_labels)
    == np.array(true_labels)
)

if np.any(correct_mask):

    mean_correct_confidence = float(
        np.mean(
            np.array(predicted_confidences)[
                correct_mask
            ]
        )
    )

else:

    mean_correct_confidence = 0.0


if np.any(~correct_mask):

    mean_wrong_confidence = float(
        np.mean(
            np.array(predicted_confidences)[
                ~correct_mask
            ]
        )
    )

else:

    mean_wrong_confidence = 0.0


# ---------------------------------------------------------
# Save JSON
# ---------------------------------------------------------

results = {
    "dataset": "PlantDoc",

    "evaluation_type":
        "real_world_field_style_evaluation",

    "total_images":
        len(image_paths),

    "accuracy":
        round(float(accuracy), 4),

    "macro_precision":
        round(float(precision), 4),

    "macro_recall":
        round(float(recall), 4),

    "macro_f1":
        round(float(f1), 4),

    "mean_prediction_confidence":
        round(mean_confidence, 4),

    "mean_correct_prediction_confidence":
        round(mean_correct_confidence, 4),

    "mean_wrong_prediction_confidence":
        round(mean_wrong_confidence, 4),

    "classes_evaluated":
        target_names,

    "classification_report":
        report,

    "files": {
        "predictions_csv":
            str(
                PREDICTIONS_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),

        "confusion_matrix":
            str(
                CONFUSION_MATRIX_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
    },
}

with open(
    RESULTS_PATH,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        results,
        f,
        indent=2,
    )


# ---------------------------------------------------------
# Final summary
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("PLANTDOC REAL-WORLD EVALUATION")
print("=" * 60)

print(
    f"Images evaluated : {len(image_paths)}"
)

print(
    f"Accuracy         : {accuracy:.4f}"
)

print(
    f"Macro Precision  : {precision:.4f}"
)

print(
    f"Macro Recall     : {recall:.4f}"
)

print(
    f"Macro F1         : {f1:.4f}"
)

print(
    f"Mean Confidence  : {mean_confidence:.4f}"
)

print(
    f"Correct Confidence : "
    f"{mean_correct_confidence:.4f}"
)

print(
    f"Wrong Confidence   : "
    f"{mean_wrong_confidence:.4f}"
)

print("\nSaved:")

print(
    f"  {RESULTS_PATH}"
)

print(
    f"  {PREDICTIONS_PATH}"
)

print(
    f"  {CONFUSION_MATRIX_PATH}"
)

print("=" * 60)