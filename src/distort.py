import tensorflow as tf
import matplotlib.pyplot as plt

IMG_WIDTH = 320
IMG_HEIGHT = 180
BATCH_SIZE = 64


def random_crop_pad(img):
    h = tf.shape(img)[0]
    w = tf.shape(img)[1]

    crop_pct = tf.random.uniform([], 0.0, 0.08)

    crop_h = tf.cast(tf.cast(h, tf.float32) * crop_pct, tf.int32)
    crop_w = tf.cast(tf.cast(w, tf.float32) * crop_pct, tf.int32)

    top = tf.random.uniform([], 0, crop_h + 1, dtype=tf.int32)
    bottom = tf.random.uniform([], 0, crop_h + 1, dtype=tf.int32)
    left = tf.random.uniform([], 0, crop_w + 1, dtype=tf.int32)
    right = tf.random.uniform([], 0, crop_w + 1, dtype=tf.int32)

    img = img[top : h - bottom, left : w - right]

    pad_pct = tf.random.uniform([], 0.0, 0.08)

    pad_h = tf.cast(tf.cast(h, tf.float32) * pad_pct, tf.int32)
    pad_w = tf.cast(tf.cast(w, tf.float32) * pad_pct, tf.int32)

    img = tf.image.pad_to_bounding_box(img, pad_h, pad_w, h + 2 * pad_h, w + 2 * pad_w)

    return img


def aspect_ratio_distort(img):
    scale_x = tf.random.uniform([], 0.95, 1.05)
    scale_y = tf.random.uniform([], 0.95, 1.05)

    h = tf.shape(img)[0]
    w = tf.shape(img)[1]

    new_h = tf.cast(tf.cast(h, tf.float32) * scale_y, tf.int32)
    new_w = tf.cast(tf.cast(w, tf.float32) * scale_x, tf.int32)

    img = tf.image.resize(img, [new_h, new_w])

    return img


def jpeg_artifacts(img):
    img_uint8 = tf.cast(tf.clip_by_value(img, 0, 255), tf.uint8)

    quality = tf.random.uniform([], 20, 70, dtype=tf.int32)

    encoded = tf.io.encode_jpeg(img_uint8, quality=quality)
    decoded = tf.io.decode_jpeg(encoded)

    return tf.cast(decoded, tf.float32)


def stream_blur(img):
    h = tf.shape(img)[0]
    w = tf.shape(img)[1]

    scale = tf.random.uniform([], 0.4, 0.8)

    small_h = tf.cast(tf.cast(h, tf.float32) * scale, tf.int32)
    small_w = tf.cast(tf.cast(w, tf.float32) * scale, tf.int32)

    img = tf.image.resize(img, [small_h, small_w])
    img = tf.image.resize(img, [h, w])

    return img


def color_jitter(img):
    img = tf.image.random_brightness(img, 0.15)
    img = tf.image.random_contrast(img, 0.8, 1.2)
    img = tf.image.random_saturation(img, 0.8, 1.2)

    return img


def add_scanlines(img):
    h = tf.shape(img)[0]
    w = tf.shape(img)[1]

    rows = tf.range(h)

    mask = tf.cast(rows % 2 == 0, tf.float32)
    mask = tf.reshape(mask, [h, 1, 1])

    strength = tf.random.uniform([], 0.6, 0.9)

    scanline_mask = mask * strength + (1.0 - mask)

    return img * scanline_mask


def add_black_bars(img):
    h = tf.shape(img)[0]
    w = tf.shape(img)[1]

    pct = tf.random.uniform([], 0.0, 0.1)

    bar_h = tf.cast(tf.cast(h, tf.float32) * pct, tf.int32)
    bar_w = tf.cast(tf.cast(w, tf.float32) * pct, tf.int32)

    if tf.random.uniform([]) < 0.5:
        img = tf.concat([tf.zeros([bar_h, w, 3]), img[bar_h:, :, :]], axis=0)

    if tf.random.uniform([]) < 0.5:
        img = tf.concat([img[:, :-bar_w, :], tf.zeros([h, bar_w, 3])], axis=1)

    return img


def add_rounded_corners(img):

    h = tf.shape(img)[0]
    w = tf.shape(img)[1]

    y = tf.cast(tf.range(h)[:, None], tf.float32)
    x = tf.cast(tf.range(w)[None, :], tf.float32)

    min_dim = tf.minimum(tf.cast(h, tf.float32), tf.cast(w, tf.float32))
    radius_pct = tf.random.uniform([], 0.1, 0.15)
    radius = radius_pct * min_dim

    tl = tf.sqrt((x - radius) ** 2 + (y - radius) ** 2)
    tr = tf.sqrt((x - (tf.cast(w, tf.float32) - radius)) ** 2 + (y - radius) ** 2)
    bl = tf.sqrt((x - radius) ** 2 + (y - (tf.cast(h, tf.float32) - radius)) ** 2)
    br = tf.sqrt(
        (x - (tf.cast(w, tf.float32) - radius)) ** 2
        + (y - (tf.cast(h, tf.float32) - radius)) ** 2
    )

    mask = tf.ones([h, w], tf.float32)

    mask = tf.where(
        (x < radius) & (y < radius),
        tf.cast(tl <= radius, tf.float32),
        mask,
    )
    mask = tf.where(
        (x > tf.cast(w, tf.float32) - radius) & (y < radius),
        tf.cast(tr <= radius, tf.float32),
        mask,
    )
    mask = tf.where(
        (x < radius) & (y > tf.cast(h, tf.float32) - radius),
        tf.cast(bl <= radius, tf.float32),
        mask,
    )
    mask = tf.where(
        (x > tf.cast(w, tf.float32) - radius) & (y > tf.cast(h, tf.float32) - radius),
        tf.cast(br <= radius, tf.float32),
        mask,
    )

    mask = tf.expand_dims(mask, axis=-1)

    return img * mask


def augment(img):

    if tf.random.uniform([]) < 0.9:
        img = add_rounded_corners(img)

    img = random_crop_pad(img)

    img = aspect_ratio_distort(img)

    img = img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH], method="nearest")

    if tf.random.uniform([]) < 0.8:
        img = jpeg_artifacts(img)

    if tf.random.uniform([]) < 0.7:
        img = stream_blur(img)

    if tf.random.uniform([]) < 0.8:
        img = color_jitter(img)

    if tf.random.uniform([]) < 0.3:
        img = add_scanlines(img)

    if tf.random.uniform([]) < 0.4:
        img = add_black_bars(img)

    img = tf.clip_by_value(img, 0, 255)

    return img


def load_image(path, label):

    img = tf.io.read_file(path)

    img = tf.io.decode_jpeg(img, channels=3)

    img = tf.cast(img, tf.float32)

    img = augment(img)

    img = img / 255.0

    return img, label


raw = tf.io.read_file("seasidedrive.png")
# raw = tf.io.decode_jpeg(raw, channels=3)
raw = tf.io.decode_png(raw, channels=3)
raw = tf.cast(raw, tf.float32)
fig, axes = plt.subplots(4, 4, figsize=(12, 8))

for ax in axes.flatten():

    aug = augment(raw)

    aug = tf.clip_by_value(aug / 255.0, 0, 1)

    ax.imshow(aug.numpy())

    ax.axis("off")

plt.tight_layout()

plt.savefig("same_frame_augments.png")


def get_dataset():

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    dataset = dataset.shuffle(10000)

    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    dataset = dataset.repeat()

    return dataset


# FINAL LAYER
# Dense(NUM_CLASSES, activation="softmax")

# model.compile(
#     optimizer="adam",
#     loss="sparse_categorical_crossentropy",
#     metrics=["accuracy"]
# )
