import os

os.environ["KERAS_BACKEND"] = "tensorflow"
import argparse

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import ops, utils
from keras_cv import layers as layers_cv
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

import wandb
from dataset_utils import get_classes
from datasets import get_data
from layers import MinMaxScaler
from models import BimodalDownstreamModel, UnimodalDownstreamModel, get_mae_model
from optimizers import get_lr_schedule
from wandb.keras import WandbMetricsLogger

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


def plot_confusion_matrix(y_true, y_pred, args):
    class_names = get_classes(mode="sc", translate=True)
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    cm = np.around(cm, decimals=2)

    wandb.sklearn.plot_confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(1, 1, figsize=(15, 15), dpi=300)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", xticks_rotation="vertical", values_format=".2f", ax=ax)

    # save plot
    fig.savefig(
        f"plots/confusion_matrix/conf_matrix_{args.model}_{args.target_modalities}_{args.select_channels_strategy}.png",
        bbox_inches="tight",
        dpi=300,
    )

    wandb.log(
        {
            "conf_matrix": wandb.Image(
                f"plots/confusion_matrix/conf_matrix_{args.model}_{args.target_modalities}_{args.select_channels_strategy}.png"
            )
        }
    )

    plt.show()


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
        "--target_modalities",
        type=str,
        default="bimodal",  # hs, rgb or bimodal
        required=False,
        help="target modalities",
    )
    parser.add_argument(
        "--use_mean_pooling",
        type=bool,
        default=True,
        required=False,
        help="use mean pooling",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=19,
        required=False,
        help="number of classes",
    )
    parser.add_argument(
        "--from_scratch",
        type=bool,
        default=False,
        required=False,
        help="train from scratch",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
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
        "--norm_layer",
        type=str,
        default="layer_norm_eps_1e-6",
        required=False,
        help="normalization layer",
    )
    parser.add_argument(
        "--learning_rate_base",
        type=float,
        default=5e-4,
        required=False,
        help="learning rate",
    )

    parser.add_argument(
        "--warmup_learning_rate",
        type=float,
        default=1e-6,
        required=False,
        help="total steps",
    )

    parser.add_argument(
        "--warmup_epoch_percentage",
        type=float,
        default=0.05,
        required=False,
        help="warmup epoch percentage",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.05,
        required=False,
        help="weight decay",
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
        "--hs_num_channels",
        type=int,
        default=5,  # change to 10/300
        required=False,
        help="number of channels",
    )

    parser.add_argument(
        "--hs_num_patches",
        type=int,
        default=300,
        required=False,
        help="number of patches",
    )

    parser.add_argument(
        "--select_channels_strategy",
        type=str,
        default="step_60",  # Use  'all', 'step_60', 'step_30', 'top_10', 'top_5', 'bottom_10' or 'bottom_5'
        required=False,
        help="select channels strategy",
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
        default=0.2,
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
        from_scratch=args.from_scratch,
        learning_rate_base=args.learning_rate_base,
        warmup_learning_rate=args.warmup_learning_rate,
        warmup_epoch_percentage=args.warmup_epoch_percentage,
        weight_decay=args.weight_decay,
        patch_size=args.patch_size,
        model=args.model,
        use_mean_pooling=args.use_mean_pooling,
        target_modalities=args.target_modalities,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hs_image_size=args.hs_image_size,
        hs_num_patches=args.hs_num_patches,
        select_channels_strategy=args.select_channels_strategy,
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

    run = wandb.init(project="finetuning_bi_mae_final", config=config)

    train_ds, valid_ds, test_ds = get_data(
        args.src_dir,
        args.batch_size,
        modality=args.target_modalities,
        downstream=True,
        preprocess_hs_fn=preprocess_hs_fn,
        select_channels_strategy=args.select_channels_strategy,  # Use 'step_60', 'step_30', 'top_10', 'top_5', 'bottom_10' or 'bottom_5'
    )

    scheduled_lrs = get_lr_schedule(
        train_ds,
        args,
        downstream=True,
    )

    if args.target_modalities == "bimodal" or args.target_modalities == "hs":
        match args.select_channels_strategy:
            case "all":
                if args.hs_num_channels == 300:
                    hs_indices = ops.arange(0, args.hs_num_patches, step=1)
                else:
                    raise ValueError("hs_num_channels must be 300 for all strategy")
            case "step_60":
                if args.hs_num_channels == 5:
                    hs_indices = ops.arange(0, args.hs_num_patches, step=60)
                else:
                    raise ValueError(
                        "hs_num_channels must be 5 for step_60 strategy"
                    )
            case "step_30":
                if args.hs_num_channels == 10:
                    hs_indices = ops.arange(0, args.hs_num_patches, step=30)
                else:
                    raise ValueError(
                        "hs_num_channels must be 10 for step_30 strategy"
                    )
            case "top_10":
                if args.hs_num_channels == 10:
                    hs_indices = ops.arange(10)
                else:
                    raise ValueError("hs_num_channels must be 10 for top_10 strategy")
            case "top_5":
                if args.hs_num_channels == 5:
                    hs_indices = ops.arange(5)
                else:
                    raise ValueError("hs_num_channels must be 5 for top_5 strategy")
            case "bottom_10":
                if args.hs_num_channels == 10:
                    hs_indices = ops.arange(
                        args.hs_num_patches - 10, args.hs_num_patches
                    )
                else:
                    raise ValueError("hs_num_channels must be 10 for bottom_10 strategy")
            case "bottom_5":
                if args.hs_num_channels == 5:
                    hs_indices = ops.arange(
                        args.hs_num_patches - 5, args.hs_num_patches
                    )
                else:
                    raise ValueError("hs_num_channels must be 5 for bottom_5 strategy")
            case _:
                raise ValueError("Invalid select channels strategy")
    else:
        hs_indices = None

    backbone = get_mae_model(
        args,
        augmenter,
        rgb_num_patches,
        hs_dec_num_mask,
        rgb_dec_num_mask,
    )

    backbone.build(
        [
            (None, args.hs_image_size, args.hs_image_size, args.hs_num_channels),
            (None, args.rgb_image_size, args.rgb_image_size, 3),
        ]
    )

    print(backbone.summary())

    if args.from_scratch:
        print("Training from scratch...")
    else:
        print("Loading weights from pre-trained model...")

        artifact = run.use_artifact(
            "uni_l_ml_group/pretraining_bi_mae_final/run_xu05jd3u_model:v112"
        )
        artifact_dir = artifact.download()
        pretrained_model = os.path.join(artifact_dir, f"ckpt_{args.model}.weights.h5")

        backbone.load_weights(pretrained_model)

    if args.target_modalities == "bimodal":
        model = BimodalDownstreamModel(
            hs_patch_extractor=backbone.hs_patch_extractor,
            rgb_patch_extractor=backbone.rgb_patch_extractor,
            hs_patch_encoder=backbone.hs_patch_encoder,
            rgb_patch_encoder=backbone.rgb_patch_encoder,
            encoder=backbone.encoder,
            use_mean_pooling=args.use_mean_pooling,
            norm_layer=backbone.norm_layer,
            global_token=backbone.global_token,
            num_classes=args.num_classes,
            hs_indices=hs_indices,
        )

    elif args.target_modalities == "hs":
        model = UnimodalDownstreamModel(
            patch_extractor=backbone.hs_patch_extractor,
            patch_encoder=backbone.hs_patch_encoder,
            encoder=backbone.encoder,
            use_mean_pooling=args.use_mean_pooling,
            norm_layer=backbone.norm_layer,
            global_token=backbone.global_token,
            num_classes=args.num_classes,
            hs_indices=hs_indices,
            modality=args.target_modalities,
        )
        # model.build(
        #    (None, args.hs_image_size, args.hs_image_size, args.hs_num_channels)
        # )

    elif args.target_modalities == "rgb":
        model = UnimodalDownstreamModel(
            patch_extractor=backbone.rgb_patch_extractor,
            patch_encoder=backbone.rgb_patch_encoder,
            encoder=backbone.encoder,
            use_mean_pooling=args.use_mean_pooling,
            norm_layer=backbone.norm_layer,
            global_token=backbone.global_token,
            num_classes=args.num_classes,
            modality=args.target_modalities,
        )
        # model.build((None, args.rgb_image_size, args.rgb_image_size, 3))
    else:
        raise ValueError("Invalid target modalities")

    # delete backbone to free up memory
    del backbone

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=scheduled_lrs, beta_1=0.9, beta_2=0.999
    )

    optimizer.exclude_from_weight_decay(
        var_names=["global_token", "mask_token", "pos_emb"]
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        jit_compile=False,
    )

    print(model.summary())

    model.fit(
        train_ds,
        epochs=args.epochs,
        validation_data=valid_ds,
        verbose=2,
        callbacks=[
            WandbMetricsLogger(log_freq=5),
        ],
    )

    test_loss, test_accuracy = model.evaluate(test_ds)
    test_accuracy = round(test_accuracy * 100, 2)

    print(f"Test accuracy: {test_accuracy}%")
    print(f"Test loss: {test_loss}")

    wandb.summary["test_accuracy"] = test_accuracy
    wandb.summary["test_loss"] = test_loss

    # plot confusion matrix
    y_pred = model.predict(test_ds)
    y_pred = np.argmax(y_pred, axis=1)
    if args.target_modalities == "bimodal":
        y_true = np.concatenate([y for _, _, y in test_ds], axis=0)
    else:
        y_true = np.concatenate([y for _, y in test_ds], axis=0)

    plot_confusion_matrix(y_true, y_pred, args)

    model.save_weights(
        f"ckpt/checkpoint.{args.model}_{args.target_modalities}_{args.select_channels_strategy}.weights.h5"
    )

    model.export(
        f"ckpt/checkpoint.{args.model}_{args.target_modalities}_{args.select_channels_strategy}_export"
    )

    wandb.finish()


if __name__ == "__main__":
    main()
