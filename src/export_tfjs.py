import keras
import tensorflowjs as tfjs

tfjs.converters.save_keras_model(
    keras.models.load_model("models/ufo50_classifier.keras"), "web/model"
)
