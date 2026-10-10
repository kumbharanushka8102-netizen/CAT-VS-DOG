
import os

from flask import Flask, render_template, request
from PIL import Image  # type: ignore[reportMissingImports]
import numpy as np

try:
    from tensorflow.keras.models import load_model  # type: ignore[reportMissingImports]
except ImportError:
    try:
        from keras.models import load_model  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "TensorFlow/Keras is required to run this app. "
            "Install it with 'pip install tensorflow' or 'pip install keras'."
        ) from exc

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_CANDIDATES = [
    os.path.join(BASE_DIR, "dog_cat_model.keras"),
    os.path.join(BASE_DIR, "dog_cat_model.h5"),
    os.path.join(BASE_DIR, "model", "dog_cat_model.keras"),
    os.path.join(BASE_DIR, "model", "dog_cat_model.h5"),
]


def load_classifier_model():
    for model_path in MODEL_CANDIDATES:
        if os.path.exists(model_path):
            try:
                return load_model(model_path)
            except (OSError, ValueError) as exc:
                raise RuntimeError(f"The saved model at '{model_path}' is invalid: {exc}") from exc
    return None


model = load_classifier_model()

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    image_path = None
    error = None

    if request.method == "POST":
        if model is None:
            error = (
                "The trained model file is missing. Place dog_cat_model.keras "
                "in the project root or a model/ folder before using the app."
            )
            return render_template(
                "index.html",
                prediction=prediction,
                image_path=image_path,
                error=error,
            )

        file = request.files.get("image")

        if file and file.filename:
            try:
                image = Image.open(file.stream).convert("RGB")
                image = image.resize((128, 128))

                pixels = np.array(image, dtype=np.float32) / 255.0
                pixels = np.expand_dims(pixels, axis=0)

                pred = float(model.predict(pixels, verbose=0)[0][0])

                prediction = "Dog" if pred > 0.5 else "Cat"
                confidence = pred if pred > 0.5 else 1 - pred

                filename = "uploaded_image.jpg"
                image_path = os.path.join("uploads", filename)
                image.save(os.path.join(UPLOAD_FOLDER, filename))

                return render_template(
                    "index.html",
                    prediction=prediction,
                    confidence=round(confidence * 100, 2),
                    image_path=image_path,
                )
            except Exception as exc:  # pragma: no cover - defensive user-facing handling
                error = f"Could not process the image: {exc}"

    return render_template(
        "index.html",
        prediction=prediction,
        image_path=image_path,
        error=error,
    )


if __name__ == "__main__":
    app.run(debug=True)

