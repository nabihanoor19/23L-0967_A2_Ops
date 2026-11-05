import tensorflow as tf
from tensorflow.keras import layers, models

IMG_SIZE = 224


def build_baseline(dropout=0.4, learning_rate=1e-3):
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image")
    x = layers.Rescaling(1.0 / 255, name="rescale")(inputs)

    for i, filters in enumerate([16, 32, 64, 128], start=1):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu", name=f"conv{i}")(x)
        x = layers.BatchNormalization(name=f"bn{i}")(x)
        x = layers.MaxPooling2D(name=f"pool{i}")(x)

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="prob_ai")(x)

    model = models.Model(inputs, outputs, name="baseline_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model

def _compile(model, learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )


def build_transfer(dropout=0.3, learning_rate=1e-3, img_size=IMG_SIZE):
    """Stage 1: frozen MobileNetV2 backbone + new head."""
    base = tf.keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3), include_top=False,
        weights="imagenet", name="backbone")
    base.trainable = False

    inputs = layers.Input(shape=(img_size, img_size, 3), name="image")
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="mobilenet_preprocess")(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="prob_ai")(x)

    model = models.Model(inputs, outputs, name="mobilenetv2_transfer")
    _compile(model, learning_rate)
    return model

def unfreeze_last_block(model, learning_rate=1e-5, first_layer="block_16_expand"):
    """Stage 2: unfreeze only the last block (block_16 + final conv), keep BatchNorm frozen."""
    base = model.get_layer("backbone")
    base.trainable = True
    reached = False
    for layer in base.layers:
        if layer.name == first_layer:
            reached = True
        layer.trainable = reached and not isinstance(layer, layers.BatchNormalization)
    _compile(model, learning_rate)       # must recompile after changing trainable
    return model

UNFREEZE_FROM = {0: None, 1: "block_16_expand", 2: "block_15_expand", 3: "block_14_expand"}


def build_tuned(dropout, learning_rate, unfrozen_blocks, img_size=IMG_SIZE):
    model = build_transfer(dropout=dropout, learning_rate=learning_rate, img_size=img_size)
    first = UNFREEZE_FROM[unfrozen_blocks]
    if first is not None:
        base = model.get_layer("backbone")
        base.trainable = True
        reached = False
        for layer in base.layers:
            if layer.name == first:
                reached = True
            layer.trainable = reached and not isinstance(layer, layers.BatchNormalization)
        _compile(model, learning_rate)
    return model