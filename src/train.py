import keras
import tensorflow as tf

from distort import augment, augment_dataset, load_image

TARGET_SIZE = (180, 320)  # model input size (square)
BATCH_SIZE = 32
NUM_CLASSES = 50

train_ds, val_ds = keras.utils.image_dataset_from_directory(
    "frames/",
    image_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    interpolation="nearest",
    subset="both",
    seed=12345,
)

train_ds = train_ds.map(
    lambda x, y: (
        tf.map_fn(
            lambda img: augment(tf.cast(img, tf.float32)),
            x,
            fn_output_signature=tf.float32,
        ),
        y,
    ),
    num_parallel_calls=tf.data.AUTOTUNE,
)

base_model = keras.applications.MobileNetV2(
    input_shape=(180, 320, 3), include_top=False, weights="imagenet"
)

base_model.trainable = False

inputs = keras.Input(shape=(180, 320, 3))

x = inputs
x = keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)

x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)

outputs = keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

model.fit(train_ds, validation_data=val_ds, epochs=10)

# Fine-tune
base_model.trainable = True

model.compile(
    optimizer=keras.optimizers.Adam(1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(train_ds, validation_data=val_ds, epochs=5)

model.save("models/ufo50_classifier.keras")
