"""
Description: Preprocessing functions for the dataset.

Functions:
    - augment: Augment the image.
    - augment_fn: Augment the sample (tf.train.Example)
    - resize_rgb_image: Resize the RGB image to the given target size.
    - resize_hs_image: Resize the hyperspectral image to the given roi_size.
"""

import tensorflow as tf
from keras import ops



def resize_rgb_image(image: tf.Tensor, target_size: tuple = (192, 192)) -> tf.Tensor:
    """
    Resize the RGB image to the given square target size.

    Args:
        image (tf.Tensor): RGB image to resize.
        target_size (tuple, optional): Target size to resize the image to. Defaults to (192, 192).

    Returns:
        tf.Tensor: Resized RGB image.
    """

    image = ops.cast(image, dtype="float32") / 255.0
    shape = ops.shape(image)

    original_height = shape[0]
    original_width = shape[1]
    aspect_ratio = original_width / original_height

    new_height = target_size[0]
    new_width = target_size[1]

    aspect_ratio = original_width / original_height
    new_height = target_size[0]
    new_width = target_size[1]

    if aspect_ratio > 1:
        new_height = ops.cast(new_width / aspect_ratio, tf.int32)
    elif aspect_ratio < 1:
        new_width = ops.cast(new_height * aspect_ratio, tf.int32)
    else:
        new_height = new_width

    resized_image = ops.image.resize(image, [new_height, new_width])
    # pad image
    pad_top = (target_size[0] - new_height) // 2
    pad_bottom = target_size[0] - new_height - pad_top
    pad_left = (target_size[1] - new_width) // 2
    pad_right = target_size[1] - new_width - pad_left

    padded_image = tf.pad(
        resized_image,
        paddings=[
            [pad_top, pad_bottom],
            [pad_left, pad_right],
            [0, 0],
        ],
        mode="constant",
        constant_values=1.0,
    )

    image = tf.convert_to_tensor(padded_image)

    return image


def apply_unit_norm(data_cube: tf.Tensor) -> tf.Tensor:
    """
    Normalizes the data cube such that the sum of brightness across all bands in the pixel after normalization is 1.

    Args:
        data_cube (tf.Tensor): Data cube to apply unit norm to.

    Returns:
        tf.Tensor: Data cube with unit norm applied.
    """
    return data_cube / ops.sum(data_cube, axis=-1, keepdims=True)


def normalize_spectra(x):
    x = (x - ops.min(x)) / (ops.max(x) - ops.min(x))

    return x


def resize_hs_image(
    image: tf.Tensor,
    roi_size: tuple = None,
    target_size: tuple = None,
    keep_num_bands: int = None,
    resize: bool = False,
    central_crop: bool = False,
) -> tf.Tensor:
    """
    Resize the hyperspectral image to the given image_size.

    Args:
        image (tf.Tensor): Hyperspectral image to resize.
        target_size (tuple, optional): Target size to resize the image to.
        keep_num_bands (int, optional): Number of bands to keep. Defaults to 300.
        resize (bool, optional): Whether to resize the image. Defaults to True.
        central_crop (bool, optional): Whether to crop the image from the center. Defaults to False.
    Returns:
        tf.Tensor: Resized hyperspectral image.
    """

    if target_size is None and roi_size is None:
        raise ValueError("Either target_size or roi_size must be defined.")

    image_shape = ops.shape(image)

    if len(image_shape) != 3:
        raise ValueError(f"Image shape must be 3D, got {len(image_shape)}D.")

    if resize and target_size is not None:
        if image_shape[0] != target_size[0] or image_shape[1] != target_size[1]:
            image = tf.image.resize_with_pad(image, target_size[0], target_size[1])

    if central_crop and roi_size is not None:
        center = (image_shape[0] // 2, image_shape[1] // 2)

        image = image[
            center[0] - roi_size[0] // 2 : center[0] + roi_size[0] // 2,
            center[1] - roi_size[1] // 2 : center[1] + roi_size[1] // 2,
        ]

        image = ops.mean(image, axis=(0, 1))

    if keep_num_bands == 50:
        # take only every sixth band (last dim)
        image = image[..., ::6]
    elif keep_num_bands == 100:
        # take only every third band (last dim)
        image = image[..., ::3]
    elif keep_num_bands == 150:
        # take only every second band (last dim)
        image = image[..., ::2]
    return image
