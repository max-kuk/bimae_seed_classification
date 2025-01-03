import argparse
from typing import Callable, Optional

import tensorflow as tf
from keras import initializers, layers, losses, metrics, models, ops, saving, export

from factory import norm_layer_factory
from layers import (
    HS_PatchEncoder,
    HS_PatchExtractor,
    RGB_PatchEncoder,
    RGB_PatchExtractor,
    create_fusion_layer,
    create_mlp_decoder,
    create_transformer_encoder,
)

@saving.register_keras_serializable(name="UnimodalDownstreamModel")
class UnimodalDownstreamModel(models.Model):
    def __init__(
        self,
        patch_extractor,
        patch_encoder,
        encoder,
        use_mean_pooling,
        global_token=None,
        norm_layer: str = "layer_norm_eps_1e-6",
        num_classes: int = None,
        hs_indices=None,
        from_scratch=False,
        modality="hs",
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Extract the patchers.
        self.patch_extractor = patch_extractor
        self.patch_encoder = patch_encoder
        self.patch_encoder.downstream = True
        self.patch_encoder.indices = hs_indices

        self.encoder = encoder
        self.use_mean_pooling = use_mean_pooling

        if use_mean_pooling is False:
            self.global_token = global_token
            self.global_token.trainable = False

        norm_layer = norm_layer_factory(norm_layer)
        self.norm1 = norm_layer(name="norm1")

        self.num_classes = num_classes
        self.fc = layers.Dense(
            self.num_classes, activation="softmax", name="classification_head"
        )
        # if from_scratch is False:
        #    self.patch_extractor.trainable = False
        #    self.patch_encoder.trainable = False
        #    self.encoder.trainable = False
        self.from_scratch = from_scratch
        self.modality = modality

    def call(self, x, training=False):
        img = x

        if self.modality == "hs":
            img.set_shape((None, 24, 24, 300))
        elif self.modality == "rgb":
            img.set_shape((None, 192, 192, 3))

        x = self.patch_extractor(img, training=training)
        x = self.patch_encoder(x, training=training)

        if self.use_mean_pooling is False:
            batch_size = ops.shape(img)
            global_token = ops.tile(self.global_token, [batch_size, 1, 1])
            global_token = ops.cast(global_token, dtype="float32")

            # add global token to encoder outputs
            x = ops.concatenate(
                [x, global_token],
                axis=1,
            )

        x = self.encoder(x, training=training)
        x = self.norm1(x)
        if self.use_mean_pooling:
            x = ops.mean(x, axis=1)
        else:
            x = x[:, -1]
        x = self.fc(x, training=training)
        return x

    @tf.function
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(y=y, y_pred=y_pred)
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics (includes the metric that tracks the loss)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        loss = self.compute_loss(y=y, y_pred=y_pred)
        # Update metrics (includes the metric that tracks the loss)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def predict_step(self, data):
        x, _ = data
        return self(x, training=False)

    def get_config(self):
        return {
            "patch_extractor": saving.serialize_keras_object(self.patch_extractor),
            "patch_encoder": saving.serialize_keras_object(self.patch_encoder),
            "encoder": saving.serialize_keras_object(self.encoder),
            "use_mean_pooling": self.use_mean_pooling,
            "global_token": saving.serialize_keras_object(self.global_token),
            "num_classes": self.num_classes,
            "from_scratch": self.from_scratch,
            "modality": self.modality,
        }

    @classmethod
    def from_config(cls, config):
        config["patch_extractor"] = saving.deserialize_keras_object(
            config["patch_extractor"]
        )
        config["patch_encoder"] = saving.deserialize_keras_object(
            config["patch_encoder"]
        )
        config["encoder"] = saving.deserialize_keras_object(config["encoder"])

        if config["global_token"] is not None:
            config["global_token"] = saving.deserialize_keras_object(
                config["global_token"]
            )

        return cls(**config)


@saving.register_keras_serializable(name="BimodalDownstreamModel")
class BimodalDownstreamModel(models.Model):
    def __init__(
        self,
        hs_patch_extractor,
        rgb_patch_extractor,
        hs_patch_encoder,
        rgb_patch_encoder,
        encoder,
        use_mean_pooling,
        global_token=None,
        norm_layer: str = "layer_norm_eps_1e-6",
        num_classes: int = None,
        hs_indices=None,
        from_scratch=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Extract the patchers.
        self.hs_patch_extractor = hs_patch_extractor
        self.rgb_patch_extractor = rgb_patch_extractor

        self.hs_patch_encoder = hs_patch_encoder
        self.hs_patch_encoder.downstream = True
        self.hs_patch_encoder.indices = hs_indices

        self.rgb_patch_encoder = rgb_patch_encoder
        self.rgb_patch_encoder.downstream = True

        self.encoder = encoder

        self.use_mean_pooling = use_mean_pooling

        if use_mean_pooling is False:
            self.global_token = global_token
            self.global_token.trainable = False

        self.num_classes = num_classes

        norm_layer = norm_layer_factory(norm_layer)
        self.norm1 = norm_layer(name="norm1")

        self.fc = layers.Dense(
            num_classes, activation="softmax", name="classification_head"
        )

        # if from_scratch is False:
        #    self.hs_patch_extractor.trainable = False
        #    self.rgb_patch_extractor.trainable = False
        #    self.hs_patch_encoder.trainable = False
        #    self.rgb_patch_encoder.trainable = False
        #    self.encoder.trainable = False
        self.from_scratch = from_scratch

    def call(self, x, training=False):
        hs_img, rgb_img = x[0], x[1]

        hs_img.set_shape((None, 24, 24, 5))  # todo: change 5 to 300
        rgb_img.set_shape((None, 192, 192, 3))

        x1 = self.hs_patch_extractor(hs_img, training=training)
        x2 = self.rgb_patch_extractor(rgb_img, training=training)

        x1 = self.hs_patch_encoder(x1, training=training)
        x2 = self.rgb_patch_encoder(x2, training=training)

        x = ops.concatenate([x1, x2], axis=1)

        if self.use_mean_pooling is False:
            batch_size = ops.shape(hs_img)[0]
            global_token = ops.tile(self.global_token, [batch_size, 1, 1])
            global_token = ops.cast(global_token, dtype="float32")
            # add global token to encoder outputs

            x = ops.concatenate(
                [x, global_token],
                axis=1,
            )

        x = self.encoder(x, training=training)
        x = self.norm1(x)

        if self.use_mean_pooling:
            x = ops.mean(x, axis=1)
        else:
            x = x[:, -1]
        x = self.fc(x, training=training)
        return x

    @tf.function
    def train_step(self, data):
        hs_img, rgb_img, y = data
        with tf.GradientTape() as tape:
            y_pred = self([hs_img, rgb_img], training=True)
            loss = self.compute_loss(y=y, y_pred=y_pred)
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics (includes the metric that tracks the loss)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def test_step(self, data):
        hs_img, rgb_img, y = data
        y_pred = self([hs_img, rgb_img], training=False)
        loss = self.compute_loss(y=y, y_pred=y_pred)
        # Update metrics (includes the metric that tracks the loss)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def predict_step(self, data, inference=False):
        if inference:
            hs_img, rgb_img = data
        else:
            hs_img, rgb_img, _ = data
        return self([hs_img, rgb_img], training=False)

    @tf.function
    def sc_default_fn(
        self,
        hs_image,
        rgb_image,
    ) -> dict:

        y_pred = self.predict_step(
            data=(hs_image, rgb_image),
            inference=True,
        )

        preds = ops.argmax(y_pred, axis=-1)
        probs = ops.max(y_pred, axis=-1)
        probs = ops.cast(probs, dtype="float32")
        # round to 2 decimal places
        probs = ops.round(probs * 100, 2)

        return {
            "predictions": ops.cast(preds, dtype="int8"),
            "probabilities": probs,
        }

    @tf.function
    def sc_raw_preds_fn(self, hs_image, rgb_image):
        y_pred = self.predict_step(data=(hs_image, rgb_image), inference=True)

        return {
            "probabilities": ops.cast(y_pred, dtype="float32"),
        }

    def get_config(self):
        return {
            "hs_patch_extractor": saving.serialize_keras_object(
                self.hs_patch_extractor
            ),
            "rgb_patch_extractor": saving.serialize_keras_object(
                self.rgb_patch_extractor
            ),
            "hs_patch_encoder": saving.serialize_keras_object(self.hs_patch_encoder),
            "rgb_patch_encoder": saving.serialize_keras_object(self.rgb_patch_encoder),
            "encoder": saving.serialize_keras_object(self.encoder),
            "global_token": saving.serialize_keras_object(self.global_token),
            "use_mean_pooling": self.use_mean_pooling,
            "num_classes": self.num_classes,
            "from_scratch": self.from_scratch,
        }

    def export(self, filepath):
        export_archive = export.ExportArchive()
        export_archive.track(self)

        export_archive.add_endpoint(
            name="sc_default",
            fn=self.sc_default_fn,
            input_signature=[
                tf.TensorSpec(
                    shape=(
                        None,
                        24,
                        24,
                        5,
                    ),
                    dtype="float32",
                ),
                tf.TensorSpec(shape=(None, 192, 192, 3), dtype="float32"),
            ],
        )

        export_archive.add_endpoint(
            name="sc_raw_preds",
            fn=self.sc_raw_preds_fn,
            input_signature=[
                tf.TensorSpec(
                    shape=(
                        None,
                        24,
                        24,
                        5,
                    ),
                    dtype="float32",
                ),
                tf.TensorSpec(shape=(None, 192, 192, 3), dtype="float32"),
            ],
        )

        export_archive.write_out(filepath)

    @classmethod
    def from_config(cls, config):
        config["hs_patch_extractor"] = saving.deserialize_keras_object(
            config["hs_patch_extractor"]
        )
        config["rgb_patch_extractor"] = saving.deserialize_keras_object(
            config["rgb_patch_extractor"]
        )
        config["hs_patch_encoder"] = saving.deserialize_keras_object(
            config["hs_patch_encoder"]
        )
        config["rgb_patch_encoder"] = saving.deserialize_keras_object(
            config["rgb_patch_encoder"]
        )
        config["encoder"] = saving.deserialize_keras_object(config["encoder"])
        if config["global_token"] is not None:
            config["global_token"] = saving.deserialize_keras_object(
                config["global_token"]
            )
        return cls(**config)


@saving.register_keras_serializable(name="MaskedAutoencoderViT")
class MaskedAutoencoderViT(models.Model):
    def __init__(
        self,
        augmenter=None,
        patch_size: int = 24,
        encoder_dim: int = 256,
        encoder_num_heads: int = 6,
        depth: int = 12,
        mlp_ratio: int = 4,
        hs_mask_proportion: float = 0.9,
        rgb_mask_proportion: float = 0.75,
        decoder_dim: int = 128,
        decoder_num_heads: int = 8,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: str = "layer_norm_eps_1e-6",
        act_layer: str = "gelu",
        hs_dec_mask_proportion: float = 0.5,
        rgb_dec_mask_proportion: float = 0.5,
        rgb_image_size: int = 96,
        hs_dec_num_mask: int = None,
        rgb_dec_num_mask: int = None,
        hs_num_patches: int = 300,
        rgb_num_patches: int = 64,
        seed: int = 42,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.augmenter = augmenter
        self.patch_size = patch_size
        self.encoder_dim = encoder_dim
        self.encoder_num_heads = encoder_num_heads
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.hs_mask_proportion = hs_mask_proportion
        self.rgb_mask_proportion = rgb_mask_proportion
        self.decoder_dim = decoder_dim
        self.decoder_num_heads = decoder_num_heads
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.norm_layer = norm_layer
        self.act_layer = act_layer
        self.drop_path_rate = drop_path_rate
        self.hs_dec_mask_proportion = hs_dec_mask_proportion
        self.rgb_dec_mask_proportion = rgb_dec_mask_proportion
        self.seed = seed
        # calculate number of visble tokens for encoder
        self.num_global_tokens = 1
        self.hs_dec_num_mask = hs_dec_num_mask
        self.rgb_dec_num_mask = rgb_dec_num_mask

        self.rgb_image_size = rgb_image_size
        self.hs_num_patches = hs_num_patches
        self.rgb_num_patches = rgb_num_patches

        self.hs_patch_extractor = HS_PatchExtractor(name="hs_patch_extractor")
        self.rgb_patch_extractor = RGB_PatchExtractor(
            patch_size=patch_size, name="rgb_patch_extractor"
        )

        self.hs_patch_encoder = HS_PatchEncoder(
            num_patches=hs_num_patches,
            patch_size=patch_size,
            projection_dim=encoder_dim,
            mask_proportion=hs_mask_proportion,
            dec_mask_proportion=hs_dec_mask_proportion,
            name="hs_patch_encoder",
        )

        self.rgb_patch_encoder = RGB_PatchEncoder(
            num_patches=rgb_num_patches,
            patch_size=patch_size,
            projection_dim=encoder_dim,
            mask_proportion=rgb_mask_proportion,
            dec_mask_proportion=rgb_dec_mask_proportion,
            name="rgb_patch_encoder",
        )

        self.encoder = create_transformer_encoder(
            projection_dim=encoder_dim,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            depth=depth,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            act_layer=act_layer,
            norm_layer=norm_layer,
            name="encoder",
        )

        self.global_token = self.add_weight(
            shape=(1, self.num_global_tokens, self.encoder_dim),
            trainable=True,
            initializer=initializers.RandomNormal(seed=seed),
            dtype="float32",
            name="global_token",
        )

        self.fusion_layer = create_fusion_layer(
            projection_dim=encoder_dim,
            num_heads=decoder_num_heads,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            act_layer=act_layer,
            name="fusion_layer",
        )

        # self.rgb_fusion_layer = create_fusion_layer(
        #    projection_dim=encoder_dim,
        #    num_heads=decoder_num_heads,
        #    drop_rate=drop_rate,
        #    drop_path_rate=drop_path_rate,
        #    norm_layer=norm_layer,
        #    act_layer=act_layer,
        #    name="rgb_fusion_layer",
        # )

        # self.hs_fusion_layer = create_fusion_layer(
        #    projection_dim=encoder_dim,
        #    num_heads=decoder_num_heads,
        #    drop_rate=drop_rate,
        #    drop_path_rate=drop_path_rate,
        #    norm_layer=norm_layer,
        #    act_layer=act_layer,
        #    name="hs_fusion_layer",
        # )

        self.hs_decoder = create_mlp_decoder(
            input_dim=encoder_dim,
            projection_dim=decoder_dim,
            output_shape=(
                hs_dec_num_mask,
                patch_size,
                patch_size,
                1,
            ),
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            norm_layer=norm_layer,
            act_layer=act_layer,
            name="hs_decoder",
        )
        self.rgb_decoder = create_mlp_decoder(
            input_dim=encoder_dim,
            projection_dim=decoder_dim,
            output_shape=(
                rgb_dec_num_mask,
                patch_size,
                patch_size,
                3,
            ),
            drop_rate=drop_rate,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            act_layer=act_layer,
            name="rgb_decoder",
        )

        self.loss_tracker = metrics.Mean(name="loss")
        self.loss_fn = losses.MeanSquaredError()

        self.hs_mae = metrics.MeanAbsoluteError(name="mae_hs")
        self.rgb_mae = metrics.MeanAbsoluteError(name="mae_rgb")

    @property
    def metrics(self):
        return [self.loss_tracker, self.hs_mae, self.rgb_mae]

    def build(self, input_shape):
        self.hs_patch_extractor.build(input_shape[0])
        self.rgb_patch_extractor.build(input_shape[1])

        self.hs_patch_encoder.build(
            (
                None,
                self.hs_patch_extractor.num_patches,
                self.hs_patch_encoder.projection_dim,
            )
        )
        self.rgb_patch_encoder.build(
            (
                None,
                self.rgb_patch_extractor.num_patches,
                self.hs_patch_encoder.projection_dim,
            )
        )

        super().build(input_shape)

    def call(self, x, training=False):
        hs_img, rgb_img = x[0], x[1]

        if training and self.augmenter is not None:
            hs_img = self.augmenter(hs_img)
            rgb_img = self.augmenter(rgb_img)

        hs_img.set_shape((None, 24, 24, 300))
        rgb_img.set_shape((None, 192, 192, 3))

        hs_patches = self.hs_patch_extractor(hs_img)
        rgb_patches = self.rgb_patch_extractor(rgb_img)

        (
            hs_unmasked_embeddings,
            hs_masked_embeddings,
            hs_unmasked_positions,
            hs_mask_indices,
            _,
        ) = self.hs_patch_encoder(hs_patches, training=training)

        (
            rgb_unmasked_embeddings,
            rgb_masked_embeddings,
            rgb_unmasked_positions,
            rgb_mask_indices,
            _,
        ) = self.rgb_patch_encoder(rgb_patches, training=training)

        # concatenate the embeddings
        unmasked_embeddings = ops.concatenate(
            [hs_unmasked_embeddings, rgb_unmasked_embeddings], axis=1
        )

        # adapt global token to batch size
        batch_size = ops.shape(hs_img)[0]
        global_token = ops.tile(
            self.global_token, [batch_size, self.num_global_tokens, 1]
        )
        global_token = ops.cast(global_token, dtype="float32")
        unmasked_embeddings = ops.concatenate(
            [unmasked_embeddings, global_token], axis=1
        )

        encoder_outputs = self.encoder(unmasked_embeddings, training=training)

        unmasked_positions = ops.concatenate(
            [hs_unmasked_positions, rgb_unmasked_positions], axis=1
        )

        encoder_outputs_without_cls = encoder_outputs[:, : -self.num_global_tokens, :]
        # encoder_outputs_without_cls = encoder_outputs[:, :0]

        # print(f"encoder_outputs_without_cls: {encoder_outputs_without_cls.shape}")
        # print(f"unmasked_positions: {unmasked_positions.shape}")

        encoder_outputs = encoder_outputs_without_cls + unmasked_positions

        # add global token to encoder outputs back
        encoder_outputs = ops.concatenate([encoder_outputs, global_token], axis=1)

        masked_embeddings = ops.concatenate(
            [hs_masked_embeddings, rgb_masked_embeddings], axis=1
        )

        fusion_outputs = self.fusion_layer(
            [masked_embeddings, encoder_outputs], training=training
        )

        # hs_decoder_inputs = self.hs_fusion_layer(
        #    [hs_masked_embeddings, encoder_outputs], training=training
        # )

        # rgb_decoder_inputs = self.rgb_fusion_layer(
        #    [rgb_masked_embeddings, encoder_outputs], training=training
        # )

        hs_decoder_inputs = fusion_outputs[:, : self.hs_dec_num_mask, :]  # HS_NUM_MASK
        rgb_decoder_inputs = fusion_outputs[:, self.hs_dec_num_mask :, :]  # HS_NUM_MASK

        hs_decoder_outputs = self.hs_decoder(hs_decoder_inputs, training=training)
        rgb_decoder_outputs = self.rgb_decoder(rgb_decoder_inputs, training=training)

        hs_decoder_patches = self.hs_patch_extractor.decoder_call(hs_decoder_outputs)
        rgb_decoder_patches = self.rgb_patch_extractor.decoder_call(rgb_decoder_outputs)

        hs_loss_patches = tf.gather(hs_patches, hs_mask_indices, axis=1, batch_dims=1)
        rgb_loss_patches = tf.gather(
            rgb_patches, rgb_mask_indices, axis=1, batch_dims=1
        )

        return (
            hs_loss_patches,
            hs_decoder_patches,
            rgb_loss_patches,
            rgb_decoder_patches,
        )

    def calculate_loss(
        self, hs_loss_patches, hs_decoder_patches, rgb_loss_patches, rgb_decoder_patches
    ):
        hs_total_loss = self.loss_fn(hs_loss_patches, hs_decoder_patches)
        rgb_total_loss = self.loss_fn(rgb_loss_patches, rgb_decoder_patches)

        return hs_total_loss + rgb_total_loss

    @tf.function(jit_compile=False)
    def train_step(self, x):
        with tf.GradientTape() as tape:
            (
                hs_loss_patch,
                hs_loss_output,
                rgb_loss_patch,
                rgb_loss_output,
            ) = self(x, training=True)

            total_loss = self.calculate_loss(
                hs_loss_patch, hs_loss_output, rgb_loss_patch, rgb_loss_output
            )

        train_vars = [
            self.hs_patch_extractor.trainable_variables,
            self.rgb_patch_extractor.trainable_variables,
            self.hs_patch_encoder.trainable_variables,
            self.rgb_patch_encoder.trainable_variables,
            self.encoder.trainable_variables,
            self.fusion_layer.trainable_variables,
            # self.hs_fusion_layer.trainable_variables,
            # self.rgb_fusion_layer.trainable_variables,
            self.hs_decoder.trainable_variables,
            self.rgb_decoder.trainable_variables,
        ]

        grads = tape.gradient(total_loss, train_vars)
        tv_list = []
        for grad, var in zip(grads, train_vars):
            for g, v in zip(grad, var):
                tv_list.append((g, v))
        self.optimizer.apply_gradients(tv_list)

        self.loss_tracker.update_state(total_loss)
        self.hs_mae.update_state(hs_loss_patch, hs_loss_output)
        self.rgb_mae.update_state(rgb_loss_patch, rgb_loss_output)

        return {
            "loss": self.loss_tracker.result(),
            "mae_hs": self.hs_mae.result(),
            "mae_rgb": self.rgb_mae.result(),
        }

    @tf.function(jit_compile=False)
    def test_step(self, x):
        (
            hs_loss_patch,
            hs_loss_output,
            rgb_loss_patch,
            rgb_loss_output,
        ) = self(x, training=False)

        total_loss = self.calculate_loss(
            hs_loss_patch, hs_loss_output, rgb_loss_patch, rgb_loss_output
        )

        self.loss_tracker.update_state(total_loss)
        self.hs_mae.update_state(hs_loss_patch, hs_loss_output)
        self.rgb_mae.update_state(rgb_loss_patch, rgb_loss_output)

        return {
            "loss": self.loss_tracker.result(),
            "mae_hs": self.hs_mae.result(),
            "mae_rgb": self.rgb_mae.result(),
        }

    def load_own_variables(self, store):
        self.global_token.assign(store["global_token"])

    #    for i, v in enumerate(self.weights):
    #        v.assign(store[f"{i}"])

    def save_own_variables(self, store):
        # super().save_own_variables(store)
        store["global_token"] = self.global_token.numpy()

    def get_config(self):
        return {
            "augmenter": saving.serialize_keras_object(self.augmenter),
            "patch_size": self.patch_size,
            "encoder_dim": self.encoder_dim,
            "encoder_num_heads": self.encoder_num_heads,
            "depth": self.depth,
            "hs_mask_proportion": self.hs_mask_proportion,
            "rgb_mask_proportion": self.rgb_mask_proportion,
            "decoder_dim": self.decoder_dim,
            "decoder_num_heads": self.decoder_num_heads,
            "drop_rate": self.drop_rate,
            "attn_drop_rate": self.attn_drop_rate,
            "drop_path_rate": self.drop_path_rate,
            "norm_layer": self.norm_layer,
            "act_layer": self.act_layer,
            "hs_dec_mask_proportion": self.hs_dec_mask_proportion,
            "rgb_dec_mask_proportion": self.rgb_dec_mask_proportion,
            "seed": self.seed,
            "hs_dec_num_mask": self.hs_dec_num_mask,
            "rgb_dec_num_mask": self.rgb_dec_num_mask,
            "hs_num_patches": self.hs_num_patches,
            "rgb_num_patches": self.rgb_num_patches,
        }

    @classmethod
    def from_config(cls, config):
        config["augmenter"] = saving.deserialize_keras_object(config["augmenter"])
        return cls(**config)


def mae_vit_custom_patch24(
    augmenter: Optional[Callable] = None,
    hs_mask_proportion: float = 0.9,
    rgb_mask_proportion: float = 0.75,
    hs_dec_mask_proportion: float = 0.5,
    rgb_dec_mask_proportion: float = 0.5,
    hs_num_patches: int = 300,
    rgb_num_patches: int = 64,
    hs_dec_num_mask: int = None,
    rgb_dec_num_mask: int = None,
    **kwargs,
):
    model = MaskedAutoencoderViT(
        augmenter=augmenter,
        patch_size=24,
        encoder_dim=256,
        encoder_num_heads=12,
        depth=12,
        mlp_ratio=4,
        hs_mask_proportion=hs_mask_proportion,
        rgb_mask_proportion=rgb_mask_proportion,
        decoder_dim=128,
        decoder_num_heads=12,
        drop_rate=0.0,
        drop_path_rate=0.0,
        hs_dec_mask_proportion=hs_dec_mask_proportion,
        hs_dec_num_mask=hs_dec_num_mask,
        rgb_dec_mask_proportion=rgb_dec_mask_proportion,
        rgb_dec_num_mask=rgb_dec_num_mask,
        hs_num_patches=hs_num_patches,
        rgb_num_patches=rgb_num_patches,
        seed=42,
        name="mae_vit_custom_patch24",
        **kwargs,
    )

    return model


def mae_vit_tiny_patch24(
    augmenter: Optional[Callable] = None,
    hs_mask_proportion: float = 0.9,
    rgb_mask_proportion: float = 0.75,
    hs_dec_mask_proportion: float = 0.5,
    rgb_dec_mask_proportion: float = 0.5,
    hs_num_patches: int = 300,
    rgb_num_patches: int = 64,
    hs_dec_num_mask: int = None,
    rgb_dec_num_mask: int = None,
    **kwargs,
):
    """
    Masked Autoencoder ViT Tiny model with 24x24 patch size.

    Args:
        augmenter (_type_, optional): _description_. Defaults to None.
        hs_mask_proportion (float, optional): _description_. Defaults to 0.9.
        rgb_mask_proportion (float, optional): float. Defaults to 0.75.
        hs_dec_mask_proportion (float, optional): float. Defaults to 0.5.
        rgb_dec_mask_proportion (float, optional): float. Defaults to 0.5.
        hs_num_patches (int, optional): int. Defaults to 300.
        rgb_num_patches (int, optional): int. Defaults to 64.
        hs_dec_num_mask (int, optional): int. Defaults to None.
        rgb_dec_num_mask (int, optional): int. Defaults to None.

    Returns:
        _type_: _description_
    """

    model = MaskedAutoencoderViT(
        augmenter=augmenter,
        patch_size=24,
        encoder_dim=192,
        encoder_num_heads=3,
        depth=12,
        mlp_ratio=4,
        hs_mask_proportion=hs_mask_proportion,
        rgb_mask_proportion=rgb_mask_proportion,
        decoder_dim=128,
        decoder_num_heads=8,
        drop_rate=0.0,
        drop_path_rate=0.0,
        hs_dec_mask_proportion=hs_dec_mask_proportion,
        hs_dec_num_mask=hs_dec_num_mask,
        rgb_dec_mask_proportion=rgb_dec_mask_proportion,
        rgb_dec_num_mask=rgb_dec_num_mask,
        hs_num_patches=hs_num_patches,
        rgb_num_patches=rgb_num_patches,
        seed=42,
        name="mae_vit_tiny_patch24",
        **kwargs,
    )

    return model


def mae_vit_small_patch24(
    augmenter: Optional[Callable] = None,
    hs_mask_proportion: float = 0.9,
    rgb_mask_proportion: float = 0.75,
    hs_dec_mask_proportion: float = 0.5,
    rgb_dec_mask_proportion: float = 0.5,
    hs_num_patches: int = 300,
    rgb_num_patches: int = 64,
    hs_dec_num_mask: int = None,
    rgb_dec_num_mask: int = None,
    **kwargs,
):
    model = MaskedAutoencoderViT(
        augmenter=augmenter,
        patch_size=24,
        encoder_dim=384,
        encoder_num_heads=6,
        depth=12,
        mlp_ratio=4,
        hs_mask_proportion=hs_mask_proportion,
        rgb_mask_proportion=rgb_mask_proportion,
        decoder_dim=192,
        decoder_num_heads=3,
        drop_rate=0.0,
        drop_path_rate=0.0,
        hs_dec_mask_proportion=hs_dec_mask_proportion,
        hs_dec_num_mask=hs_dec_num_mask,
        rgb_dec_mask_proportion=rgb_dec_mask_proportion,
        rgb_dec_num_mask=rgb_dec_num_mask,
        hs_num_patches=hs_num_patches,
        rgb_num_patches=rgb_num_patches,
        seed=42,
        name="mae_vit_small_patch24",
        **kwargs,
    )

    return model


def mae_vit_base_patch24(
    augmenter: Optional[Callable] = None,
    hs_mask_proportion: float = 0.9,
    rgb_mask_proportion: float = 0.75,
    hs_dec_mask_proportion: float = 0.5,
    rgb_dec_mask_proportion: float = 0.5,
    hs_num_patches: int = 300,
    rgb_num_patches: int = 64,
    hs_dec_num_mask: int = None,
    rgb_dec_num_mask: int = None,
    **kwargs,
):
    model = MaskedAutoencoderViT(
        augmenter=augmenter,
        patch_size=24,
        encoder_dim=768,
        encoder_num_heads=12,
        depth=12,
        mlp_ratio=4,
        hs_mask_proportion=hs_mask_proportion,
        rgb_mask_proportion=rgb_mask_proportion,
        decoder_dim=384,
        decoder_num_heads=6,
        drop_rate=0.0,
        drop_path_rate=0.0,
        hs_dec_mask_proportion=hs_dec_mask_proportion,
        hs_dec_num_mask=hs_dec_num_mask,
        rgb_dec_mask_proportion=rgb_dec_mask_proportion,
        rgb_dec_num_mask=rgb_dec_num_mask,
        hs_num_patches=hs_num_patches,
        rgb_num_patches=rgb_num_patches,
        seed=42,
        name="mae_vit_base_patch24",
        **kwargs,
    )

    return model


def get_mae_model(
    args: argparse.Namespace,
    augmenter: Optional[Callable],
    rgb_num_patches: int,
    hs_dec_num_mask: int,
    rgb_dec_num_mask: int,
):
    match args.model:
        case "mae_vit_custom_patch24":
            model = mae_vit_custom_patch24(
                augmenter=augmenter,
                hs_num_patches=args.hs_num_patches,
                rgb_num_patches=rgb_num_patches,
                hs_mask_proportion=args.hs_mask_proportion,
                hs_dec_mask_proportion=args.hs_decoder_mask_proportion,
                hs_dec_num_mask=hs_dec_num_mask,
                rgb_mask_proportion=args.rgb_mask_proportion,
                rgb_dec_mask_proportion=args.rgb_decoder_mask_proportion,
                rgb_dec_num_mask=rgb_dec_num_mask,
            )
        case "mae_vit_tiny_patch24":
            model = mae_vit_tiny_patch24(
                augmenter=augmenter,
                hs_num_patches=args.hs_num_patches,
                rgb_num_patches=rgb_num_patches,
                hs_mask_proportion=args.hs_mask_proportion,
                hs_dec_mask_proportion=args.hs_decoder_mask_proportion,
                hs_dec_num_mask=hs_dec_num_mask,
                rgb_mask_proportion=args.rgb_mask_proportion,
                rgb_dec_mask_proportion=args.rgb_decoder_mask_proportion,
                rgb_dec_num_mask=rgb_dec_num_mask,
            )
        case "mae_vit_small_patch24":
            model = mae_vit_small_patch24(
                augmenter=augmenter,
                hs_num_patches=args.hs_num_patches,
                rgb_num_patches=rgb_num_patches,
                hs_mask_proportion=args.hs_mask_proportion,
                hs_dec_mask_proportion=args.hs_decoder_mask_proportion,
                hs_dec_num_mask=hs_dec_num_mask,
                rgb_mask_proportion=args.rgb_mask_proportion,
                rgb_dec_mask_proportion=args.rgb_decoder_mask_proportion,
                rgb_dec_num_mask=rgb_dec_num_mask,
            )
        case "mae_vit_base_patch24":
            model = mae_vit_base_patch24(
                augmenter=augmenter,
                hs_num_patches=args.hs_num_patches,
                rgb_num_patches=rgb_num_patches,
                hs_mask_proportion=args.hs_mask_proportion,
                hs_dec_mask_proportion=args.hs_decoder_mask_proportion,
                hs_dec_num_mask=hs_dec_num_mask,
                rgb_mask_proportion=args.rgb_mask_proportion,
                rgb_dec_mask_proportion=args.rgb_decoder_mask_proportion,
                rgb_dec_num_mask=rgb_dec_num_mask,
            )

    return model
