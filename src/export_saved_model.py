import tensorflow as tf

model = tf.keras.models.load_model("models/ufo50_classifier.keras")

model.export("saved_model")
