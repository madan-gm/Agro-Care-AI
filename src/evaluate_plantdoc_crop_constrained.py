import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "mobilenetv2_finetuned_best.keras"
)

CLASSES_PATH = (
    PROJECT_ROOT
    / "app"
    / "classes.json"
)

ORIGINAL_CSV = (
    PROJECT_ROOT
    / "reports"
    / "plantdoc_predictions.csv"
)

RESULTS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "plantdoc_crop_constrained_results.json"
)

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "plantdoc_crop_constrained_predictions.csv"
)


# =========================================================
# SETTINGS
# =========================================================

IMAGE_SIZE = (224, 224)


# =========================================================
# CROP CLASSES
# =========================================================

CROP_CLASSES = {

    "Corn": [
        "Corn___Cercospora_leaf_spot Gray_leaf_spot",
        "Corn___Common_rust",
        "Corn___Northern_Leaf_Blight",
        "Corn___healthy",
    ],

    "Potato": [
        "Potato___Early_blight",
        "Potato___Late_blight",
        "Potato___healthy",
    ],

    "Tomato": [
        "Tomato___Bacterial_spot",
        "Tomato___Early_blight",
        "Tomato___Late_blight",
        "Tomato___Leaf_Mold",
        "Tomato___Septoria_leaf_spot",
        "Tomato___Target_Spot",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
        "Tomato___healthy",
    ],
}


# =========================================================
# LOAD CLASSES
# =========================================================

print("\nLoading classes...")

with open(
    CLASSES_PATH,
    "r",
    encoding="utf-8"
) as f:
    classes = json.load(f)

class_to_index = {
    name: index
    for index, name in enumerate(classes)
}

print(
    f"Number of classes: {len(classes)}"
)


# =========================================================
# LOAD ORIGINAL 85-IMAGE CSV
# =========================================================

print("\nLoading original PlantDoc prediction list...")

if not ORIGINAL_CSV.exists():

    print(
        f"ERROR: Original prediction CSV not found:"
    )

    print(
        ORIGINAL_CSV
    )

    raise SystemExit(1)


original_df = pd.read_csv(
    ORIGINAL_CSV
)

print(
    f"Images in original CSV: "
    f"{len(original_df)}"
)


# =========================================================
# LOAD MODEL
# =========================================================

print("\nLoading model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print(
    "Model loaded successfully."
)


# =========================================================
# CREATE CROP INDEX MAP
# =========================================================

crop_indices = {}

for crop, crop_classes in CROP_CLASSES.items():

    crop_indices[crop] = [
        class_to_index[class_name]
        for class_name in crop_classes
    ]


# =========================================================
# PREPARE EXACT SAME 85 RECORDS
# =========================================================

records = []

for _, row in original_df.iterrows():

    image_path = (
        PROJECT_ROOT
        / str(row["image"])
    )

    true_class = str(
        row["true_class"]
    )

    # Determine crop from true class

    crop = None

    for crop_name, crop_classes in CROP_CLASSES.items():

        if true_class in crop_classes:

            crop = crop_name
            break

    if crop is None:

        print(
            f"WARNING: Could not determine crop for:"
        )

        print(
            true_class
        )

        continue

    records.append(
        {
            "path": image_path,
            "crop": crop,
            "true_class": true_class,
        }
    )


print(
    f"\nExact records selected: "
    f"{len(records)}"
)


# =========================================================
# IMPORTANT CHECK
# =========================================================

if len(records) != 85:

    print(
        "\nWARNING: Expected exactly 85 records."
    )

    print(
        f"Found: {len(records)}"
    )

    print(
        "Check reports/plantdoc_predictions.csv"
    )

    raise SystemExit(1)


# =========================================================
# RUN PREDICTIONS
# =========================================================

true_labels = []

predicted_labels = []

confidences = []

prediction_rows = []

failed_images = []


print(
    "\nRunning crop-constrained predictions..."
)


for number, record in enumerate(
    records,
    start=1
):

    image_path = record["path"]

    crop = record["crop"]

    true_class = record["true_class"]

    try:

        # -------------------------------------------------
        # LOAD IMAGE
        # -------------------------------------------------

        image = Image.open(
            image_path
        ).convert("RGB")

        # -------------------------------------------------
        # RESIZE
        # -------------------------------------------------

        image = image.resize(
            IMAGE_SIZE
        )

        # -------------------------------------------------
        # CONVERT TO NUMPY
        # -------------------------------------------------

        image_array = np.asarray(
            image,
            dtype=np.float32
        )

        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        # IMPORTANT:
        #
        # The trained model already contains its own
        # preprocessing.
        #
        # DO NOT call preprocess_input() here.

        # -------------------------------------------------
        # MODEL PREDICTION
        # -------------------------------------------------

        probabilities = model.predict(
            image_array,
            verbose=0
        )[0]

        # -------------------------------------------------
        # CROP-CONSTRAINED PREDICTION
        # -------------------------------------------------

        allowed_indices = crop_indices[
            crop
        ]

        allowed_probabilities = (
            probabilities[
                allowed_indices
            ]
        )

        best_position = int(
            np.argmax(
                allowed_probabilities
            )
        )

        predicted_index = (
            allowed_indices[
                best_position
            ]
        )

        confidence = float(
            probabilities[
                predicted_index
            ]
        )

        true_index = class_to_index[
            true_class
        ]

        # -------------------------------------------------
        # SAVE RESULT
        # -------------------------------------------------

        true_labels.append(
            true_index
        )

        predicted_labels.append(
            predicted_index
        )

        confidences.append(
            confidence
        )

        prediction_rows.append(
            {
                "image": str(
                    image_path.relative_to(
                        PROJECT_ROOT
                    )
                ),

                "crop": crop,

                "true_class": true_class,

                "predicted_class":
                    classes[predicted_index],

                "confidence":
                    confidence,

                "correct":
                    predicted_index == true_index,
            }
        )

        if (
            number % 10 == 0
            or number == len(records)
        ):

            print(
                f"Processed "
                f"{number}/{len(records)}"
            )

    except Exception as error:

        failed_images.append(
            {
                "image":
                    str(image_path),

                "error":
                    str(error),
            }
        )

        print(
            f"\nERROR processing:"
        )

        print(
            image_path
        )

        print(
            f"Reason: {error}"
        )


# =========================================================
# FINAL IMAGE COUNT
# =========================================================

print("\n" + "=" * 60)

print(
    f"Original CSV images : {len(original_df)}"
)

print(
    f"Records selected    : {len(records)}"
)

print(
    f"Images evaluated    : {len(true_labels)}"
)

print(
    f"Images failed       : {len(failed_images)}"
)

print("=" * 60)


# =========================================================
# STOP IF IMAGES FAILED
# =========================================================

if len(true_labels) != 85:

    print(
        "\nWARNING: Not all 85 images were evaluated."
    )

    print(
        "The failed images are listed above."
    )


# =========================================================
# METRICS
# =========================================================

accuracy = accuracy_score(
    true_labels,
    predicted_labels
)

precision = precision_score(
    true_labels,
    predicted_labels,
    average="macro",
    zero_division=0
)

recall = recall_score(
    true_labels,
    predicted_labels,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    true_labels,
    predicted_labels,
    average="macro",
    zero_division=0
)


# =========================================================
# CLASSIFICATION REPORT
# =========================================================

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
    zero_division=0
)


# =========================================================
# CONFIDENCE ANALYSIS
# =========================================================

confidence_array = np.array(
    confidences
)

correct_mask = (
    np.array(predicted_labels)
    ==
    np.array(true_labels)
)

mean_confidence = float(
    np.mean(
        confidence_array
    )
)

if np.any(correct_mask):

    mean_correct_confidence = float(
        np.mean(
            confidence_array[
                correct_mask
            ]
        )
    )

else:

    mean_correct_confidence = 0.0


if np.any(~correct_mask):

    mean_wrong_confidence = float(
        np.mean(
            confidence_array[
                ~correct_mask
            ]
        )
    )

else:

    mean_wrong_confidence = 0.0


# =========================================================
# PER-CROP ACCURACY
# =========================================================

prediction_df = pd.DataFrame(
    prediction_rows
)

crop_results = {}

for crop in CROP_CLASSES:

    crop_df = prediction_df[
        prediction_df["crop"] == crop
    ]

    if len(crop_df) == 0:

        continue

    crop_accuracy = (
        crop_df["correct"]
        .astype(bool)
        .mean()
    )

    crop_results[crop] = {

        "images":
            int(len(crop_df)),

        "accuracy":
            round(
                float(crop_accuracy),
                4
            ),
    }


# =========================================================
# SAVE CSV
# =========================================================

prediction_df.to_csv(
    PREDICTIONS_PATH,
    index=False
)


# =========================================================
# SAVE JSON
# =========================================================

results = {

    "dataset":
        "PlantDoc",

    "evaluation_type":
        "oracle_crop_constrained_evaluation",

    "description":
        (
            "The exact same 85 images used in "
            "plantdoc_predictions.csv were evaluated. "
            "Predictions were restricted to the "
            "known ground-truth crop classes. "
            "This is a diagnostic experiment and "
            "not an automatic crop classifier."
        ),

    "original_images":
        len(original_df),

    "images_selected":
        len(records),

    "images_evaluated":
        len(true_labels),

    "failed_images":
        len(failed_images),

    "accuracy":
        round(
            float(accuracy),
            4
        ),

    "macro_precision":
        round(
            float(precision),
            4
        ),

    "macro_recall":
        round(
            float(recall),
            4
        ),

    "macro_f1":
        round(
            float(f1),
            4
        ),

    "mean_prediction_confidence":
        round(
            mean_confidence,
            4
        ),

    "mean_correct_prediction_confidence":
        round(
            mean_correct_confidence,
            4
        ),

    "mean_wrong_prediction_confidence":
        round(
            mean_wrong_confidence,
            4
        ),

    "per_crop":
        crop_results,

    "classification_report":
        report,

    "failed_image_details":
        failed_images,

    "files": {

        "predictions_csv":
            str(
                PREDICTIONS_PATH.relative_to(
                    PROJECT_ROOT
                )
            )
    }
}


with open(
    RESULTS_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=2
    )


# =========================================================
# FINAL OUTPUT
# =========================================================

print("\n")

print("=" * 60)

print(
    "PLANTDOC CROP-CONSTRAINED EVALUATION"
)

print("=" * 60)

print(
    f"Images evaluated : "
    f"{len(true_labels)}"
)

print(
    f"Accuracy         : "
    f"{accuracy:.4f}"
)

print(
    f"Macro Precision  : "
    f"{precision:.4f}"
)

print(
    f"Macro Recall     : "
    f"{recall:.4f}"
)

print(
    f"Macro F1         : "
    f"{f1:.4f}"
)

print(
    f"Mean Confidence  : "
    f"{mean_confidence:.4f}"
)

print(
    f"Correct Confidence: "
    f"{mean_correct_confidence:.4f}"
)

print(
    f"Wrong Confidence: "
    f"{mean_wrong_confidence:.4f}"
)

print("\nPer-crop accuracy:")

for crop, values in crop_results.items():

    print(
        f"  {crop}: "
        f"{values['accuracy']:.4f} "
        f"({values['images']} images)"
    )

print("\nSaved:")

print(
    RESULTS_PATH
)

print(
    PREDICTIONS_PATH
)

print("=" * 60)