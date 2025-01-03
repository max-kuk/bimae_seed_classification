import json
import logging
import os

import spectral.io.envi as envi
import tensorflow as tf
import yaml

import preprocessing

feature_description = {
    "id": tf.io.FixedLenFeature(
        (),
        tf.string,
    ),
    "hs_image": tf.io.VarLenFeature(tf.string),
    "hs_size": tf.io.FixedLenFeature(
        (3,),
        tf.int64,
    ),
    "rgb_image": tf.io.VarLenFeature(tf.string),
    "rgb_size": tf.io.FixedLenFeature(
        (3,),
        tf.int64,
    ),
    "class": tf.io.FixedLenFeature(
        (),
        tf.string,
    ),
}


def parse_fn(
    example_proto: tf.train.Example,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Function to parse a single sample from a tfrecord file.

    Args:
        example_proto (tf.train.Example): A single example from a tfrecord file.

    Returns:
        tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]: A single sample.
    """

    # Parse the input tf.Example proto using the dictionary above
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)

    file_id = parsed_example["id"]
    label = parsed_example["class"]

    hs_image = tf.sparse.to_dense(parsed_example["hs_image"])
    hs_image = tf.io.decode_raw(hs_image, "float32")
    hs_image = tf.reshape(hs_image, parsed_example["hs_size"])
    hs_image = preprocessing.resize_hs_image(
        hs_image, target_size=(24, 24), keep_num_bands=300, resize=True
    )

    rgb_image = tf.sparse.to_dense(parsed_example["rgb_image"])
    rgb_image = tf.io.decode_png(rgb_image[0], channels=3)
    rgb_image = tf.reshape(rgb_image, parsed_example["rgb_size"])

    return {
        "id": file_id,
        "hs_image": hs_image,
        "rgb_image": rgb_image,
        "label": label,
    }


def parse_export_fn(
    example_proto: tf.train.Example,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Function to parse a single sample from a tfrecord file.

    Args:
        example_proto (tf.train.Example): A single example from a tfrecord file.

    Returns:
        tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]: A single sample.
    """

    # Parse the input tf.Example proto using the dictionary above
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)

    file_id = parsed_example["id"]
    label = parsed_example["class"]

    hs_image = tf.sparse.to_dense(parsed_example["hs_image"])
    hs_image = tf.io.decode_raw(hs_image, "float32")
    hs_image = tf.reshape(hs_image, parsed_example["hs_size"])

    hs_image = preprocessing.resize_hs_image( 
        hs_image, target_size=(24, 24), keep_num_bands=300, resize=True
    )


    rgb_image = tf.sparse.to_dense(parsed_example["rgb_image"])
    rgb_image = tf.io.decode_png(rgb_image[0], channels=3)
    rgb_image = tf.reshape(rgb_image, parsed_example["rgb_size"])
    rgb_image = preprocessing.resize_rgb_image(rgb_image)

    return {
        "id": file_id,
        "hs_image": hs_image,
        "rgb_image": rgb_image,
        "label": label,
    }


def save_dataset(
    dst_dir: str,
    dataset: tf.data.Dataset,
    num_shards: int = 4,
    split: str = "train",
    compression="GZIP",
    random_seed: int = 42,
):
    """
    Save the dataset to the given path.

    Args:
        dst_dir (str): Path to save the dataset.
        dataset (tf.data.Dataset): Dataset to save.
        num_shards (int, optional): Num of shards to split the dataset. Defaults to 4.
        split (str, optional): Type of dataset. Defaults to "train". Can be "train", "val" or "test".
        compression (str, optional): Compression type. Defaults to "GZIP".
        random_seed (int, optional): Random seed. Defaults to 42.
    """
    logging.info(f"Saving {split} dataset...")

    dataset.save(
        path=os.path.join(dst_dir, split),
        compression=compression,
        shard_func=lambda x: shard_fn(
            sample=x, num_shards=num_shards, random_seed=random_seed
        ),
    )


def reader_fn(
    datasets: tf.data.Dataset, num_shards: int, deterministic: bool, random_seed: int
) -> tf.data.Dataset:
    """
    Function to read the dataset.

    Args:
        datasets (tf.data.Dataset): Dataset to read.
        num_shards (int): Num of shards to split the dataset.
        deterministic (bool): If True, deterministic order is used.
        random_seed (int): Random seed.

    Returns:
        tf.data.Dataset: Read dataset.
    """
    datasets = datasets.shuffle(num_shards, seed=random_seed)
    return datasets.interleave(
        lambda x: x,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=deterministic,
    )


def shard_fn(sample: dict, num_shards: int, random_seed: int = 42) -> tf.Tensor:
    """
    Function to shard the dataset.

    Args:
        sample (dict): Sample to shard.
        num_shards (int): Num of shards to split the dataset.
        random_seed (int, optional): Random seed. Defaults to 42.

    Returns:
        tf.Tensor: Shard index.
    """
    return tf.random.uniform(
        [1], minval=0, maxval=num_shards, dtype=tf.int64, seed=random_seed
    )


def convert_to_dict(file_id, hs_image, rgb_image, label):
    hs_image = preprocessing.resize_hs_image(
        hs_image, target_size=(24, 24), keep_num_bands=300, resize=True
    )
    hs_image.set_shape((24, 24, 300))

    rgb_image = preprocessing.resize_rgb_image(rgb_image)
    rgb_image.set_shape((192, 192, 3))

    return {
        "id": file_id,
        "hs_image": hs_image,
        "rgb_image": rgb_image,
        "label": label,
    }


def add_class_id(sample: dict, normal_label: str = "b. napus") -> dict:
    """
    Add class id to the sample (if it is normal class, class_id = 0, else anomalous class, class_id = 1).

    Args:
        sample (dict): Sample to add class id to.
        normal_label (str, optional): Label to filter from dataset. Defaults to "b. napus".

    Returns:
        dict: Sample with class id.
    """
    class_id = tf.where(tf.equal(sample["label"], normal_label), 0, 1)
    sample["class_id"] = tf.cast(class_id, tf.int32)
    return sample


def add_label_id(sample: dict) -> dict:
    label_mapping = get_label_mapping()

    class_id = label_mapping.lookup(tf.convert_to_tensor(sample["label"]))
    sample["class_id"] = tf.cast(class_id, tf.int8)
    return sample


# @tf.py_function(
#    Tout=[tf.string, tf.float32, tf.uint8, tf.string],
# )
def read_raw_files(src_dir: str) -> tuple[str, tf.Tensor, tf.Tensor, str]:
    """Reads the RGB and HS images from the given folder.

    Args:
        src_dir (str): Path to the folder containing the RGB and HS images.

    Returns:
        tuple[str, tf.Tensor, tf.Tensor, str]: File ID, HS image, RGB image.

    """

    src_dir = src_dir.numpy().decode("utf-8")
    file_id = os.path.basename(src_dir)

    rgb_img = tf.io.read_file(os.path.join(src_dir, "RGBSeed.png"))
    rgb_img = tf.image.decode_png(rgb_img, channels=3)

    hs_img = envi.open(
        os.path.join(src_dir, "HsSeed.hdr"), os.path.join(src_dir, "HsSeed.bil")
    )
    hs_img = hs_img.load()  # type: ignore

    with open(os.path.join(src_dir, "Meta.JSON"), encoding="utf-8") as json_file:
        meta_file = json.load(json_file)

    label = meta_file["species"]
    label = label.lower().strip()

    return file_id, hs_img, rgb_img, label


@tf.autograph.experimental.do_not_convert
def get_classes(mode: str = "od", translate: bool = False) -> list[str]:
    """
    Get the classes from the species file.

    Args:
        mode (str, optional): Mode of the dataset.
        Defaults to "od". Can be "od" (outlier detection) or "sc" (supervised classification).
        translate (bool, optional): Whether to translate the classes. Defaults to False.

    Raises:
        ValueError: Invalid mode.

    Returns:
        list[str]: Classes {class_name: class_id}
    """
    if mode == "od":
        with open("config/od_species.yaml", "r", encoding="utf-8") as file:
            dict_file = yaml.load(file, Loader=yaml.FullLoader)

    elif mode == "sc":
        with open("config/sc_species.yaml", "r", encoding="utf-8") as file:
            dict_file = yaml.load(file, Loader=yaml.FullLoader)
    else:
        raise ValueError(
            "Invalid mode. Use 'od' for outlier detection or 'sc' for supervised classification."
        )

    if translate:
        translation_dict = get_translations()
        dict_file = {translation_dict[k]: v for k, v in dict_file.items()}

        # sort by key alphabetically
        dict_file = dict(sorted(dict_file.items(), key=lambda x: x[0]))

    return list(dict_file.keys())


def get_translations(path: str = "config/translation.yaml") -> dict:
    """
    Get the translations from the translation file.

    Args:
        path (str, optional): Path to the translation file. Defaults to "config/translation.yaml".

    Returns:
        dict: Translations.
    """
    with open(path, "r", encoding="utf-8") as file:
        dict_file = yaml.load(file, Loader=yaml.FullLoader)
    return dict_file


def get_label_mapping():
    species = get_classes(mode="sc", translate=True)

    label_mapping = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            tf.constant(species),
            tf.constant(list(range(len(species)))),
        ),
        default_value=-1,
    )

    return label_mapping
