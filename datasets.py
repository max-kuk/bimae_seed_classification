"""
Description: Module to handle the datasets.

Functions:
    - load_dataset: Loads the dataset from the given path.
    - parse_fn: Parses the sample.
    - get_dataset: Gets the dataset from the given path.
    - get_sc_dataset: Gets the dataset for the Supervised Classification task.
    - export_sc_dataset: Exports the dataset for the Supervised Classification task.
"""

import logging
import os

import tensorflow as tf

import dataset_utils as ds_utils

AUTOTUNE = tf.data.experimental.AUTOTUNE


def prepare_dataset(
    dataset,
    split: str = "train",
    downstream: bool = False,
    deterministic: bool = False,
    modality: str = "bimodal",
    preprocess_fn=None,
    cache: bool = False,
    subset: int = None,
    select_channels_strategy: str = "step_30",  # downstream only (select every 30th channel per default)
):
    if modality not in ["bimodal", "hs", "rgb"]:
        raise ValueError("Invalid 'modality' value. Use 'bimodal', 'hs' or 'rgb'.")

    if downstream:
        if select_channels_strategy == "all":
            select_channels = None
        if select_channels_strategy == "step_60":
            select_channels = 60
        if select_channels_strategy == "step_30":
            select_channels = 30
        elif select_channels_strategy == "top_10":
            select_channels = 10
        elif select_channels_strategy == "top_5":
            select_channels = 5
        elif select_channels_strategy == "bottom_10":
            select_channels = -10
        elif select_channels_strategy == "bottom_5":
            select_channels = -5

    if downstream and select_channels_strategy not in [
        "all",
        "step_60",
        "step_30",
        "top_10",
        "top_5",
        "bottom_10",
        "bottom_5",
    ]:
        raise ValueError(
            "Invalid 'select_channels_strategy' value. Use 'all', 'step_60', 'step_30', 'top_10', 'top_5', 'bottom_10' or 'bottom_5'."
        )

    if modality == "bimodal":
        if downstream is False:
            dataset = dataset.map(
                lambda x: (
                    preprocess_fn(x["hs_image"]),
                    x["rgb_image"],
                ),
                num_parallel_calls=AUTOTUNE,
                deterministic=deterministic,
            )
        else:
            if select_channels_strategy in ["step_60", "step_30"]:
                dataset = dataset.map(
                    lambda x: (
                        preprocess_fn(x["hs_image"])[..., ::select_channels],
                        x["rgb_image"],
                        x["class_id"],
                    ),
                    num_parallel_calls=AUTOTUNE,
                    deterministic=deterministic,
                )
            elif select_channels_strategy in ["top_10", "top_5"]:
                dataset = dataset.map(
                    lambda x: (
                        preprocess_fn(x["hs_image"])[..., :select_channels],
                        x["rgb_image"],
                        x["class_id"],
                    ),
                    num_parallel_calls=AUTOTUNE,
                    deterministic=deterministic,
                )
            elif select_channels_strategy in ["bottom_10", "bottom_5"]:
                dataset = dataset.map(
                    lambda x: (
                        preprocess_fn(x["hs_image"])[..., select_channels:],
                        x["rgb_image"],
                        x["class_id"],
                    ),
                    num_parallel_calls=AUTOTUNE,
                    deterministic=deterministic,
                )
            elif select_channels_strategy == "all":
                dataset = dataset.map(
                    lambda x: (
                        preprocess_fn(x["hs_image"]),
                        x["rgb_image"],
                        x["class_id"],
                    ),
                    num_parallel_calls=AUTOTUNE,
                    deterministic=deterministic,
                )

    elif modality == "hs":
        if downstream is False:
            dataset = dataset.map(
                lambda x: preprocess_fn(x["hs_image"]),
                num_parallel_calls=AUTOTUNE,
                deterministic=deterministic,
            )
        else:
            if select_channels_strategy in ["step_60", "step_30"]:
                dataset = dataset.map(
                    lambda x: (
                        preprocess_fn(x["hs_image"])[..., ::select_channels],
                        x["class_id"],
                    ),
                    num_parallel_calls=AUTOTUNE,
                    deterministic=deterministic,
                )

            elif select_channels_strategy in ["top_10", "top_5"]:
                dataset = dataset.map(
                    lambda x: (
                        preprocess_fn(x["hs_image"])[..., :select_channels],
                        x["class_id"],
                    ),
                    num_parallel_calls=AUTOTUNE,
                    deterministic=deterministic,
                )

            elif select_channels_strategy in ["bottom_10", "bottom_5"]:
                dataset = dataset.map(
                    lambda x: (
                        preprocess_fn(x["hs_image"])[..., select_channels:],
                        x["class_id"],
                    ),
                    num_parallel_calls=AUTOTUNE,
                    deterministic=deterministic,
                )
            elif select_channels_strategy == "all":
                dataset = dataset.map(
                    lambda x: (preprocess_fn(x["hs_image"]), x["class_id"]),
                    num_parallel_calls=AUTOTUNE,
                    deterministic=deterministic,
                )

    elif modality == "rgb":
        if downstream is False:
            dataset = dataset.map(
                lambda x: preprocess_fn(x["rgb_image"]),
                num_parallel_calls=AUTOTUNE,
                deterministic=deterministic,
            )
        else:
            dataset = dataset.map(
                lambda x: (x["rgb_image"], x["class_id"]),
                num_parallel_calls=AUTOTUNE,
                deterministic=deterministic,
            )

    if subset is not None:
        dataset = dataset.take(subset)

    if cache and split in ["train", "val"] and subset is None:
        if downstream:
            dataset = dataset.cache(
                f"/tmp/{downstream}_{modality}_{split}_{select_channels_strategy}_ds_cache"
            )
        else:
            dataset = dataset.cache(f"/tmp/{modality}_{split}_ds_cache")

    return dataset.prefetch(AUTOTUNE)


def min_max_scale(x):
    # apply min max normalization to the hyperspectral image for the batch
    min_vals = tf.math.reduce_min(x, keepdims=True, axis=[1, 2, 3])
    max_vals = tf.math.reduce_max(x, keepdims=True, axis=[1, 2, 3])
    range_nonzero = tf.where(min_vals != max_vals, max_vals - min_vals, 1.0)
    return (x - min_vals) / range_nonzero


def get_sc_dataset(
    src_dir: str = "/mnt/data/sc/",
    split: str = "train",
    batch_size: int = 32,
    data_type: str = "bimodal",
    deterministic: bool = True,
    random_seed: int = 42,
    compression: str = "GZIP",
    num_shards: int = 20,
    drop_remainder: bool = False,
) -> tf.data.Dataset:
    """
    Get the dataset for supervised classification.

    Args:
        src_dir (str): Path to the dataset directory.
        split (str, optional): Type of dataset. Defaults to "train". Can be "train", "val" or "test".
        batch_size (int, optional): Size of batches. Defaults to 32.
        data_type (str, optional): Type of dataset, one of 'bimodal', 'hs' or 'rgb'. Defaults to 'bimodal'.
        label_names (list): Names of the class labels. Used to map from the names to integer values.
        deterministic (bool, optional): Whether to use deterministic operations. Defaults to True.
        random_seed (int, optional): Random seed. Defaults to 42.
        compression (str, optional): Compression type. Defaults to "GZIP".
        num_shards (int, optional): Num of shards to split the dataset. Defaults to 4.


    Returns:
        tf.data.Dataset: Supervised classification dataset.
    """

    if split not in ["train", "val", "test"]:
        raise ValueError("Invalid 'split' value. Use 'train', 'val' or 'test'.")

    if data_type not in ["bimodal", "hs", "rgb"]:
        raise ValueError("Invalid 'data_type' value. Use 'bimodal', 'hs' or 'rgb'.")

    dataset = tf.data.Dataset.load(
        os.path.join(src_dir, split),
        compression=compression,
        reader_func=lambda x: ds_utils.reader_fn(
            x,
            num_shards=num_shards,
            deterministic=deterministic,
            random_seed=random_seed,
        ),
    ).batch(
        batch_size,
        drop_remainder=drop_remainder,
    )

    if data_type == "bimodal":
        dataset = dataset.map(
            lambda x: {
                "id": x["id"],
                "hs_image": x["hs_image"],
                "rgb_image": x["rgb_image"],
                "label": x["label"],
            },
            deterministic=deterministic,
            num_parallel_calls=AUTOTUNE,
        )
    elif data_type == "hs":
        dataset = dataset.map(
            lambda x: {
                "id": x["id"],
                # take only top 3 channels
                "hs_image": x["hs_image"][..., ::30],
                # map labels to integer values
                "label": x["label"],
            },
            deterministic=deterministic,
            num_parallel_calls=AUTOTUNE,
        )
    elif data_type == "rgb":
        dataset = dataset.map(
            lambda x: {"id": x["id"], "rgb_image": x["rgb_image"], "label": x["label"]},
            deterministic=deterministic,
            num_parallel_calls=AUTOTUNE,
        )
    dataset = dataset.map(
        ds_utils.add_label_id,
        num_parallel_calls=AUTOTUNE,
        deterministic=deterministic,
    )

    return dataset


def export_sc_datasets(
    src_dir: str,
    dst_dir: str = "/mnt/data/sc/hs1",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    deterministic: bool = True,
    shuffle: bool = False,
    random_seed: int = 42,
    compression: str = "GZIP",
    num_shards: int = 15,
):
    """
    Export train and val datasets for supervised classification.

    Args:
        src_dir (str): Path to tfrecord files.
        dst_dir (str, optional): Path to export datasets. Defaults to "/mnt/data/sc/".
        train_ratio (float): Proportion of the data to use for training dataset. Defaults to 0.8.
        compression (str, optional): Compression type. Defaults to "GZIP".
        random_seed (int, optional): Random seed. Defaults to 42.
        num_shards (int, optional): Num of shards to split the dataset. Defaults to 4.
    """
    if train_ratio > 1.0 or train_ratio < 0.0:
        raise ValueError("Invalid train_ratio. Value must be between 0 and 1.0")

    logging.info(
        f"Creating datasets for supervised classification. Seeds are: {ds_utils.get_classes(mode='sc', translate=False)}"
    )

    dataset = tf.data.TFRecordDataset.list_files(
        os.path.join(src_dir, "*.tfrecords"), seed=random_seed, shuffle=shuffle
    ).interleave(
        lambda x: tf.data.TFRecordDataset(x, compression_type="GZIP").map(
            map_func=ds_utils.parse_export_fn,
            num_parallel_calls=AUTOTUNE,
            deterministic=deterministic,
        ),
        cycle_length=4,
        num_parallel_calls=AUTOTUNE,
        deterministic=deterministic,
    )

    # determine number of samples in the dataset and the splits
    nb_samples = sum([1 for _ in dataset])

    logging.info(
        f"Total number of samples: {nb_samples}. Train/val/test ratio: {train_ratio}/{val_ratio}/{1-train_ratio-val_ratio}"
    )

    train_samples = int(nb_samples * train_ratio)
    logging.info(f"Number of training samples: {train_samples}")

    val_samples = int(nb_samples * val_ratio)
    logging.info(f"Number of validation samples: {val_samples}")

    test_samples = nb_samples - train_samples - val_samples
    logging.info(f"Number of test samples: {test_samples}")

    train_ds = dataset.take(train_samples)
    val_ds = dataset.skip(train_samples).take(val_samples)
    test_ds = dataset.skip(train_samples + val_samples).take(test_samples)

    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    ds_utils.save_dataset(
        dst_dir=dst_dir,
        dataset=train_ds,
        num_shards=num_shards,
        split="train",
        compression=compression,
        random_seed=random_seed,
    )
    ds_utils.save_dataset(
        dst_dir=dst_dir,
        dataset=val_ds,
        num_shards=num_shards,
        split="val",
        compression=compression,
        random_seed=random_seed,
    )

    ds_utils.save_dataset(
        dst_dir=dst_dir,
        dataset=test_ds,
        num_shards=num_shards,
        split="test",
        compression=compression,
        random_seed=random_seed,
    )


def get_data(
    src_dir,
    batch_size: int,
    modality: str = "bimodal",
    downstream: bool = False,
    preprocess_hs_fn=None,
    select_channels_strategy=None,
):
    train_ds = get_sc_dataset(
        src_dir,
        "train",
        batch_size=batch_size,
        data_type=modality,
    )

    valid_ds = get_sc_dataset(
        src_dir,
        "val",
        batch_size=batch_size,
    )
    test_ds = get_sc_dataset(
        src_dir,
        "test",
        batch_size=batch_size,
        data_type=modality,
    )

    train_ds = prepare_dataset(
        train_ds,
        "train",
        modality=modality,
        preprocess_fn=preprocess_hs_fn,
        cache=False,
        downstream=downstream,
        select_channels_strategy=select_channels_strategy,
    )
    valid_ds = prepare_dataset(
        valid_ds,
        "val",
        modality=modality,
        preprocess_fn=preprocess_hs_fn,
        cache=False,
        downstream=downstream,
        select_channels_strategy=select_channels_strategy,
    )

    test_ds = prepare_dataset(
        test_ds,
        "test",
        modality=modality,
        preprocess_fn=preprocess_hs_fn,
        cache=False,
        downstream=downstream,
        select_channels_strategy=select_channels_strategy,
    )

    return train_ds, valid_ds, test_ds
