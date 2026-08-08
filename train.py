import argparse
import os
import pathlib
import shutil

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from model import CLASS_NAMES, IMG_SIZE, MODEL_PATH, build_model

AUTOTUNE = tf.data.AUTOTUNE
JUNK_SUFFIXES = {".db", ".ini", ".txt", ".csv", ".html"}


def clean_dataset(data_dir):
    """Remove Thumbs.db and other non-image files that break the loader."""
    removed = 0
    for path in pathlib.Path(data_dir).rglob("*"):
        if path.is_file() and path.suffix.lower() in JUNK_SUFFIXES:
            path.unlink()
            removed += 1
    if removed:
        print(f"Removed {removed} non-image file(s).")
    return removed


def verify_layout(data_dir):
    """Fail loudly before a long training run if folders are wrong."""
    root = pathlib.Path(data_dir)
    if not root.is_dir():
        hint = ""
        candidates = [
            p
            for p in pathlib.Path(".").iterdir()
            if p.is_dir()
            and all((p / c).is_dir() for c in CLASS_NAMES)
        ]
        if candidates:
            hint = (
                "\n       Found a folder that looks right:\n"
                f"       python train.py --data-dir {candidates[0].name}"
            )
        raise SystemExit(f"ERROR: '{data_dir}' does not exist.{hint}")

    # Kaggle archives often unzip to <root>/cell_images/{Parasitized,Uninfected}.
    if not (root / "Parasitized").is_dir():
        for child in root.iterdir():
            if child.is_dir() and all((child / c).is_dir() for c in CLASS_NAMES):
                raise SystemExit(
                    f"ERROR: class folders are nested one level deeper.\n"
                    f"       Point --data-dir at that instead:\n"
                    f"       python train.py --data-dir {child.as_posix()}"
                )

    missing = [c for c in CLASS_NAMES if not (root / c).is_dir()]
    if missing:
        found = sorted(p.name for p in root.iterdir() if p.is_dir())
        raise SystemExit(
            f"ERROR: missing class folder(s): {missing}\n"
            f"       Found instead: {found}"
        )

    counts = {}
    for c in CLASS_NAMES:
        counts[c] = sum(
            1 for p in (root / c).iterdir()
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        if counts[c] == 0:
            raise SystemExit(f"ERROR: '{root / c}' contains no images.")
    print(f"Found images: {counts}")
    return counts


def make_datasets(data_dir, batch_size, val_split, seed):
    common = dict(
        directory=data_dir,
        labels="inferred",
        label_mode="binary",
        # Pinned explicitly: index 0 = Uninfected, 1 = Parasitized.
        # Alphabetical default would invert this and silently break the API.
        class_names=list(CLASS_NAMES),
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        validation_split=val_split,
        seed=seed,
    )
    train_ds = keras.utils.image_dataset_from_directory(subset="training", **common)
    val_ds = keras.utils.image_dataset_from_directory(subset="validation", **common)

    # Guard against a silent label inversion: Keras honours `class_names`, but
    # if that ever drifts from CLASS_NAMES the API would report the opposite
    # diagnosis with high confidence. Cheapest possible insurance.
    if list(train_ds.class_names) != list(CLASS_NAMES):
        raise SystemExit(
            f"ERROR: label order mismatch.\n"
            f"       loader gave {train_ds.class_names}\n"
            f"       expected    {list(CLASS_NAMES)}"
        )

    augment = keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.1),
        ],
        name="augment",
    )

    # Scale to 0-1 here, NOT inside the model: main.py already divides by 255
    # before calling predict. A Rescaling layer in the model would double-apply.
    def scale(x, y):
        return tf.cast(x, tf.float32) / 255.0, y

    train_ds = (
        train_ds.map(scale, num_parallel_calls=AUTOTUNE)
        .cache()
        .shuffle(1000, seed=seed, reshuffle_each_iteration=True)
        .map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=AUTOTUNE)
        .prefetch(AUTOTUNE)
    )
    val_ds = val_ds.map(scale, num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)
    return train_ds, val_ds


def evaluate(model, val_ds):
    probs, truth = [], []
    for batch_x, batch_y in val_ds:
        probs.append(model.predict(batch_x, verbose=0).ravel())
        truth.append(batch_y.numpy().ravel())
    probs = np.concatenate(probs)
    truth = np.concatenate(truth).astype(int)
    pred = (probs >= 0.5).astype(int)

    tp = int(((pred == 1) & (truth == 1)).sum())
    tn = int(((pred == 0) & (truth == 0)).sum())
    fp = int(((pred == 1) & (truth == 0)).sum())
    fn = int(((pred == 0) & (truth == 1)).sum())

    acc = (tp + tn) / max(len(truth), 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)

    print("\nValidation results")
    print("-" * 46)
    print(f"{'':<22}{'pred Uninf':>11}{'pred Para':>12}")
    print(f"{'actual Uninfected':<22}{tn:>11}{fp:>12}")
    print(f"{'actual Parasitized':<22}{fn:>11}{tp:>12}")
    print("-" * 46)
    print(f"accuracy   {acc:.4f}")
    print(f"precision  {prec:.4f}   (of predicted Parasitized, share correct)")
    print(f"recall     {rec:.4f}   (of true Parasitized, share caught)")
    print(f"f1         {f1:.4f}")
    print(f"score range {probs.min():.4f} - {probs.max():.4f}")

    if probs.max() - probs.min() < 0.20:
        print("\nWARNING: outputs barely vary - the model did not learn.")
    if acc < 0.80:
        print("\nWARNING: accuracy below 0.80 - check the class folder order.")
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default=MODEL_PATH)
    args = ap.parse_args()

    print(f"Class mapping: 0 -> {CLASS_NAMES[0]}, 1 -> {CLASS_NAMES[1]}\n")
    clean_dataset(args.data_dir)
    verify_layout(args.data_dir)

    train_ds, val_ds = make_datasets(
        args.data_dir, args.batch_size, args.val_split, args.seed
    )

    model = build_model()
    ckpt = args.output + ".ckpt.keras"
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=4, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            ckpt, monitor="val_accuracy", save_best_only=True, verbose=0
        ),
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)
    acc = evaluate(model, val_ds)

    if os.path.exists(ckpt):
        os.remove(ckpt)

    if os.path.exists(args.output):
        backup = args.output + ".untrained.bak"
        if not os.path.exists(backup):
            shutil.copy2(args.output, backup)
            print(f"\nBacked up previous weights to {backup}")

    model.save(args.output)
    print(f"Saved trained model to {args.output}  (val accuracy {acc:.4f})")
    print("Restart uvicorn to pick up the new weights.")


if __name__ == "__main__":
    main()
