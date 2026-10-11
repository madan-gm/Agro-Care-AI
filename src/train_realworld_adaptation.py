import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


# =========================================================
# CONFIGURATION
# =========================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# PlantDoc is smaller than PlantVillage, so repeat it
# to give real-world images a stronger influence.
PLANTDOC_REPEAT = 3

EPOCHS = 10

LEARNING_RATE = 1e-5


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "mobilenetv2_finetuned_best.keras"
)

OUTPUT_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "mobilenetv2_realworld_adapted.keras"
)

CLASSES_PATH = (
    PROJECT_ROOT
    / "app"
    / "classes.json"
)

PLANTVILLAGE_TRAIN = (
    PROJECT_ROOT
    / "data"
    / "train"
)

PLANTVILLAGE_VAL = (
    PROJECT_ROOT
    / "data"
    / "val"
)

PLANTDOC_TRAIN = (
    Path(r"C:\Agro Care Ai")
    / "PlantDoc-Dataset"
    / "train"
)

REPORTS_PATH = (
    PROJECT_ROOT
    / "reports"
)

REPORTS_PATH.mkdir(
    parents=True,
    exist_ok=True
)

HISTORY_PATH = (
    REPORTS_PATH
    / "realworld_adaptation_history.json"
)


# =========================================================
# PLANTDOC → OUR MODEL CLASS MAPPING
# =========================================================

PLANTDOC_MAPPING = {

    "Corn Gray leaf spot":
        "Corn___Cercospora_leaf_spot Gray_leaf_spot",

    "Corn rust leaf":
        "Corn___Common_rust",

    "Corn leaf blight":
        "Corn___Northern_Leaf_Blight",

    "Potato leaf early blight":
        "Potato___Early_blight",

    "Potato leaf late blight":
        "Potato___Late_blight",

    "Tomato Early blight leaf":
        "Tomato___Early_blight",

    "Tomato leaf bacterial spot":
        "Tomato___Bacterial_spot",

    "Tomato leaf late blight":
        "Tomato___Late_blight",

    "Tomato Septoria leaf spot":
        "Tomato___Septoria_leaf_spot",

    "Tomato leaf yellow virus":
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
}


# =========================================================
# LOAD CLASS LIST
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
# IMAGE FILE EXTENSIONS
# =========================================================

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# =========================================================
# COLLECT PLANTVILLAGE TRAINING IMAGES
# =========================================================

print("\nCollecting PlantVillage training images...")

plantvillage_records = []


for class_name in classes:

    folder = (
        PLANTVILLAGE_TRAIN
        / class_name
    )

    if not folder.exists():

        print(
            f"WARNING: Missing PlantVillage folder:"
        )

        print(folder)

        continue

    class_index = class_to_index[
        class_name
    ]

    image_files = [
        p
        for p in folder.rglob("*")
        if (
            p.is_file()
            and p.suffix.lower()
            in VALID_EXTENSIONS
        )
    ]

    print(
        f"{class_name}: "
        f"{len(image_files)} images"
    )

    for image_path in image_files:

        plantvillage_records.append(
            (
                str(image_path),
                class_index,
                "plantvillage",
            )
        )


print(
    f"\nPlantVillage training images: "
    f"{len(plantvillage_records)}"
)


# =========================================================
# COLLECT PLANTDOC TRAINING IMAGES
# =========================================================

print(
    "\nCollecting PlantDoc real-world "
    "training images..."
)

plantdoc_records = []


for plantdoc_name, our_class in (
    PLANTDOC_MAPPING.items()
):

    folder = (
        PLANTDOC_TRAIN
        / plantdoc_name
    )

    if not folder.exists():

        print(
            f"WARNING: Missing PlantDoc folder:"
        )

        print(folder)

        continue

    class_index = class_to_index[
        our_class
    ]

    image_files = [
        p
        for p in folder.rglob("*")
        if (
            p.is_file()
            and p.suffix.lower()
            in VALID_EXTENSIONS
        )
    ]

    print(
        f"{plantdoc_name} -> "
        f"{our_class}: "
        f"{len(image_files)} images"
    )

    for image_path in image_files:

        plantdoc_records.append(
            (
                str(image_path),
                class_index,
                "plantdoc",
            )
        )


print(
    f"\nPlantDoc training images: "
    f"{len(plantdoc_records)}"
)


# =========================================================
# VERIFY EXPECTED COUNTS
# =========================================================

if len(plantdoc_records) != 1018:

    print(
        "\nWARNING:"
    )

    print(
        "Expected 1018 PlantDoc images."
    )

    print(
        f"Found {len(plantdoc_records)}."
    )

    print(
        "Training will stop so we don't "
        "silently use the wrong dataset."
    )

    raise SystemExit(1)


# =========================================================
# SPLIT PLANTDOC TRAINING DATA
#
# IMPORTANT:
# The existing PlantDoc TEST images are NOT touched.
# We create validation data only from PlantDoc TRAIN.
# =========================================================

plantdoc_paths = [
    item[0]
    for item in plantdoc_records
]

plantdoc_labels = [
    item[1]
    for item in plantdoc_records
]


(
    plantdoc_train_paths,
    plantdoc_val_paths,
    plantdoc_train_labels,
    plantdoc_val_labels,
) = train_test_split(

    plantdoc_paths,

    plantdoc_labels,

    test_size=0.20,

    random_state=SEED,

    stratify=plantdoc_labels,
)


# =========================================================
# CREATE DATA RECORDS
# =========================================================

plantdoc_train_records = [
    (
        path,
        label,
        "plantdoc",
    )
    for path, label
    in zip(
        plantdoc_train_paths,
        plantdoc_train_labels,
    )
]


plantdoc_val_records = [
    (
        path,
        label,
        "plantdoc",
    )
    for path, label
    in zip(
        plantdoc_val_paths,
        plantdoc_val_labels,
    )
]


# =========================================================
# REPEAT PLANTDOC TRAINING DATA
# =========================================================

plantdoc_train_records = (
    plantdoc_train_records
    * PLANTDOC_REPEAT
)


# =========================================================
# COMBINE TRAINING DATA
# =========================================================

combined_train_records = (
    plantvillage_records
    +
    plantdoc_train_records
)


# Shuffle

random.shuffle(
    combined_train_records
)


# =========================================================
# LOAD IMAGE FUNCTION
# =========================================================

def load_image(path, label):

    image = tf.io.read_file(
        path
    )

    image = tf.io.decode_image(
        image,
        channels=3,
        expand_animations=False,
    )

    image.set_shape(
        [None, None, 3]
    )

    image = tf.image.resize(
        image,
        IMAGE_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    )

    return image, label


# =========================================================
# CREATE TRAIN DATASET
# =========================================================

train_paths = [
    item[0]
    for item in combined_train_records
]

train_labels = [
    item[1]
    for item in combined_train_records
]


train_ds = tf.data.Dataset.from_tensor_slices(
    (
        train_paths,
        train_labels,
    )
)

train_ds = train_ds.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

train_ds = train_ds.shuffle(
    buffer_size=min(
        len(train_paths),
        5000
    ),
    seed=SEED,
    reshuffle_each_iteration=True,
)

train_ds = train_ds.batch(
    BATCH_SIZE
)

train_ds = train_ds.prefetch(
    tf.data.AUTOTUNE
)


# =========================================================
# CREATE PLANTDOC VALIDATION DATASET
# =========================================================

val_paths = [
    item[0]
    for item in plantdoc_val_records
]

val_labels = [
    item[1]
    for item in plantdoc_val_records
]


val_ds = tf.data.Dataset.from_tensor_slices(
    (
        val_paths,
        val_labels,
    )
)

val_ds = val_ds.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

val_ds = val_ds.batch(
    BATCH_SIZE
)

val_ds = val_ds.prefetch(
    tf.data.AUTOTUNE
)


# =========================================================
# DATASET SUMMARY
# =========================================================

print("\n" + "=" * 60)

print("REAL-WORLD ADAPTATION DATASET")

print("=" * 60)

print(
    f"PlantVillage train: "
    f"{len(plantvillage_records)}"
)

print(
    f"PlantDoc train before repeat: "
    f"{len(plantdoc_train_records) // PLANTDOC_REPEAT}"
)

print(
    f"PlantDoc train after repeat: "
    f"{len(plantdoc_train_records)}"
)

print(
    f"Combined training images: "
    f"{len(combined_train_records)}"
)

print(
    f"PlantDoc validation images: "
    f"{len(plantdoc_val_records)}"
)

print(
    f"PlantDoc test benchmark: 85"
)

print("=" * 60)


# =========================================================
# LOAD CURRENT BEST MODEL
# =========================================================

print(
    "\nLoading current best model..."
)

model = tf.keras.models.load_model(
    MODEL_PATH
)

print(
    "Current model loaded."
)


# =========================================================
# FINE-TUNING
# =========================================================

print(
    "\nPreparing model for fine-tuning..."
)

# Freeze everything first

for layer in model.layers:

    layer.trainable = False


# Find the MobileNetV2 backbone

backbone = None

for layer in model.layers:

    if (
        "mobilenetv2" in
        layer.name.lower()
    ):

        backbone = layer
        break


if backbone is None:

    print(
        "ERROR: MobileNetV2 backbone "
        "could not be found."
    )

    raise SystemExit(1)


print(
    f"Backbone found: "
    f"{backbone.name}"
)


# Unfreeze last ~30 layers,
# except BatchNormalization layers.

trainable_count = 0

for layer in backbone.layers[-30:]:

    if isinstance(
        layer,
        tf.keras.layers.BatchNormalization
    ):

        layer.trainable = False

    else:

        layer.trainable = True

        trainable_count += 1


print(
    f"Trainable backbone layers: "
    f"{trainable_count}"
)


# =========================================================
# COMPILE
# =========================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),

    loss=(
        "sparse_categorical_crossentropy"
    ),

    metrics=[
        "accuracy"
    ],
)


# =========================================================
# CALLBACKS
# =========================================================

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(

        filepath=str(
            OUTPUT_MODEL_PATH
        ),

        monitor="val_accuracy",

        mode="max",

        save_best_only=True,

        verbose=1,
    ),

    tf.keras.callbacks.EarlyStopping(

        monitor="val_accuracy",

        mode="max",

        patience=3,

        restore_best_weights=True,

        verbose=1,
    ),

    tf.keras.callbacks.ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.5,

        patience=2,

        min_lr=1e-7,

        verbose=1,
    ),
]


# =========================================================
# TRAIN
# =========================================================

print(
    "\nStarting real-world adaptation..."
)

print(
    "This may take some time."
)

history = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=EPOCHS,

    callbacks=callbacks,

)


# =========================================================
# SAVE FINAL MODEL
# =========================================================

model.save(
    OUTPUT_MODEL_PATH
)


# =========================================================
# SAVE HISTORY
# =========================================================

history_data = {
    key: [
        float(value)
        for value in values
    ]
    for key, values
    in history.history.items()
}


with open(
    HISTORY_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        history_data,
        f,
        indent=2
    )


# =========================================================
# FINAL OUTPUT
# =========================================================

print("\n" + "=" * 60)

print(
    "REAL-WORLD ADAPTATION COMPLETE"
)

print("=" * 60)

print(
    f"New model:"
)

print(
    OUTPUT_MODEL_PATH
)

print(
    "\nTraining history:"
)

print(
    HISTORY_PATH
)

print("=" * 60)