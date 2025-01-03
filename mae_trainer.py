import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import argparse

import tensorflow as tf
from keras import utils
from keras_cv import layers as layers_cv

import wandb
from datasets import get_data
from layers import MinMaxScaler
from models import get_mae_model
from optimizers import get_lr_schedule
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint


physical_devices = tf.config.list_physical_devices("GPU")
tf.config.set_visible_devices(physical_devices[0], "GPU")


preprocess_rgb_layers = [
    layers_cv.Resizing(192, 192),
]

preprocess_hs_layers = [MinMaxScaler()]

augmentation_layers = [
    layers_cv.RandomFlip("horizontal_and_vertical", seed=42),
]


def preprocess_hs_fn(x):
    for layer in preprocess_hs_layers:
        x = layer(x)
    return x


def augmenter(x):
    for layer in augmentation_layers:
        x = layer(x)
    return x


def main():
    parser = argparse.ArgumentParser(description="Train MAE model")
    parser.add_argument(
        "--src_dir",
        type=str,
        default="/mnt/data/sc/bimodal_24_192_300_bands",
        required=False,
        help="source directory",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mae_vit_tiny_patch24",
        required=False,
        help="model name",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        required=False,
        help="number of epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        required=False,
        help="batch size",
    )
    parser.add_argument(
        "--learning_rate_base",
        type=float,
        # default=5e-3,
        default=1e-4,
        required=False,
        help="learning rate base",
    )

    parser.add_argument(
        "--warmup_learning_rate",
        type=float,
        default=1e-6,
        required=False,
        help="warmup learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.05,
        required=False,
        help="weight decay",
    )

    parser.add_argument(
        "--warmup_epoch_percentage",
        type=float,
        default=0.10,
        required=False,
        help="warmup epoch percentage",
    )

    parser.add_argument(
        "--hs_image_size",
        type=int,
        default=24,
        required=False,
        help="hyperspectral image size",
    )

    parser.add_argument(
        "--patch_size",
        type=int,
        default=24,
        required=False,
        help="patch size",
    )

    parser.add_argument(
        "--hs_num_patches",
        type=int,
        default=300,
        required=False,
        help="number of patches",
    )

    parser.add_argument(
        "--hs_mask_proportion",
        type=float,
        default=0.95,
        required=False,
        help="mask proportion",
    )

    parser.add_argument(
        "--hs_decoder_mask_proportion",
        type=float,
        default=0.20,
        required=False,
        help="decoder mask proportion",
    )
    parser.add_argument(
        "--rgb_image_size",
        type=int,
        default=192,
        required=False,
        help="rgb image size",
    )

    parser.add_argument(
        "--rgb_num_patches",
        type=int,
        default=64,
        required=False,
        help="number of patches",
    )

    parser.add_argument(
        "--rgb_mask_proportion",
        type=float,
        default=0.75,
        required=False,
        help="mask proportion",
    )

    parser.add_argument(
        "--rgb_decoder_mask_proportion",
        type=float,
        default=0.5,
        required=False,
        help="decoder mask proportion",
    )

    parser.add_argument(
        "--seed", type=int, default=42, required=False, help="random seed"
    )

    args = parser.parse_args()

    utils.set_random_seed(args.seed)

    hs_num_mask = int(args.hs_num_patches * args.hs_mask_proportion)
    hs_dec_num_mask = int(hs_num_mask * args.hs_decoder_mask_proportion)

    rgb_num_patches = (args.rgb_image_size // args.patch_size) ** 2
    rgb_num_mask = int(rgb_num_patches * args.rgb_mask_proportion)
    rgb_dec_num_mask = int(rgb_num_mask * args.rgb_decoder_mask_proportion)

    config = dict(
        src_dir=args.src_dir,
        learning_rate_base=args.learning_rate_base,
        warmup_learning_rate=args.warmup_learning_rate,
        warmup_epoch_percentage=args.warmup_epoch_percentage,
        weight_decay=args.weight_decay,
        patch_size=args.patch_size,
        model=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hs_image_size=args.hs_image_size,
        hs_num_patches=args.hs_num_patches,
        hs_mask_proportion=args.hs_mask_proportion,
        hs_decoder_mask_proportion=args.hs_decoder_mask_proportion,
        hs_num_mask=hs_num_mask,
        hs_num_unmask=args.hs_num_patches - hs_num_mask,
        hs_dec_num_mask=hs_dec_num_mask,
        rgb_image_size=args.rgb_image_size,
        rgb_num_patches=rgb_num_patches,
        rgb_mask_proportion=args.rgb_mask_proportion,
        rgb_decoder_mask_proportion=args.rgb_decoder_mask_proportion,
        rgb_num_mask=rgb_num_mask,
        rgb_num_unmask=rgb_num_patches - rgb_num_mask,
        rgb_dec_num_mask=rgb_dec_num_mask,
        seed=args.seed,
    )

    wandb.init(project="pretraining_bi_mae_final", config=config)

    train_ds, valid_ds, test_ds = get_data(
        args.src_dir,
        args.batch_size,
        downstream=False,
        preprocess_hs_fn=preprocess_hs_fn,
    )

    scheduled_lrs = get_lr_schedule(train_ds, args)

    mae_model = get_mae_model(
        args,
        augmenter,
        rgb_num_patches,
        hs_dec_num_mask,
        rgb_dec_num_mask,
    )

    mae_model.build(
        [
            (None, args.hs_image_size, args.hs_image_size, args.hs_num_patches),
            (None, args.rgb_image_size, args.rgb_image_size, 3),
        ]
    )

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=scheduled_lrs,
        weight_decay=args.weight_decay,
    )
    optimizer.exclude_from_weight_decay(
        var_names=["global_token", "mask_token", "pos_emb"]
    )

    mae_model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=["mae"],
        jit_compile=False,
    )

    wandb.log({"model_summary": mae_model.summary()})

    mae_model.fit(
        train_ds,
        epochs=args.epochs,
        validation_data=valid_ds,
        callbacks=[
            WandbMetricsLogger(log_freq=5),
            WandbModelCheckpoint(
                filepath=f"ckpt/ckpt_{args.model}.weights.h5",
                save_weights_only=True,
                save_best_only=True,
            ),
        ],
        verbose=2,
    )

    loss, hs_mae, rgb_mae = mae_model.evaluate(test_ds)
    # wandb.log({"loss": loss, "hs_mae": hs_mae, "rgb_mae": rgb_mae})
    wandb.summary["test_loss"] = loss
    wandb.summary["test_hs_mae"] = hs_mae
    wandb.summary["test_rgb_mae"] = rgb_mae

    wandb.finish()


if __name__ == "__main__":
    main()
