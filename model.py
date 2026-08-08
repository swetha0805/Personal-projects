from tensorflow import keras
from tensorflow.keras import layers

IMG_SIZE = 128
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
MODEL_PATH = "malaria_model.h5"
CLASS_NAMES = ("Uninfected", "Parasitized")


def build_model(input_shape=INPUT_SHAPE):
    """Return a compiled lightweight CNN for binary classification."""
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(16, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.GlobalAveragePooling2D(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="malaria_cnn",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_initial_model(path=MODEL_PATH):
    """Build an untrained model and persist it to disk."""
    model = build_model()
    model.save(path)
    return model


if __name__ == "__main__":
    model = save_initial_model()
    model.summary()
    print(f"\nSaved initialized model to {MODEL_PATH}")
    print("NOTE: weights are random - train on the NIH malaria dataset for real use.")
