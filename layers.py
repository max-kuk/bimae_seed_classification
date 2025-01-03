from collections import OrderedDict

import keras_cv
import numpy as np
import tensorflow as tf
from keras import initializers, layers, models, ops, random, saving, utils
from matplotlib import pyplot as plt

from factory import act_layer_factory, norm_layer_factory


@saving.register_keras_serializable(package="my_custom_package")
class RGB_PatchExtractor(layers.Layer):
    def __init__(self, patch_size: int, **kwargs):
        super(RGB_PatchExtractor, self).__init__(**kwargs)
        self.patch_size = patch_size

    def build(self, input_shape):
        (_, self.image_size, _, _) = input_shape

        self.num_patches = (self.image_size // self.patch_size) ** 2
        super().build((None, self.image_size, self.image_size, 3))

    def call(self, x):
        x = ops.reshape(x, (-1, self.image_size, self.image_size, 3))
        x = ops.image.extract_patches(
            image=x,
            size=[self.patch_size, self.patch_size],
            strides=[1, self.patch_size, self.patch_size, 1],
            dilation_rate=1,
            padding="VALID",
        )

        x = ops.reshape(
            x,
            (
                -1,
                self.num_patches,
                self.patch_size * self.patch_size * 3,
            ),
        )
        return x

    def decoder_call(self, x):
        batch_size = ops.shape(x)[0]
        num_patches = ops.shape(x)[1]

        x = ops.reshape(
            x,
            (
                batch_size,
                num_patches,
                self.patch_size * self.patch_size * 3,
            ),
        )

        return x

    def show_patched_image(
        self,
        images,
        patches,
    ):
        # This is a utility function which accepts a batch of images and its
        # corresponding patches and help visualize one image and its patches
        # side by side.

        idx = np.random.choice(patches.shape[0])
        print(f"Index selected: {idx}.")

        plt.figure(figsize=(4, 4))
        plt.imshow(utils.array_to_img(images[idx]))
        plt.axis("off")
        plt.show()

        n = int(np.sqrt(patches.shape[1]))
        plt.figure(figsize=(4, 4))
        for i, patch in enumerate(patches[idx]):
            ax = plt.subplot(n, n, i + 1)
            patch_img = ops.reshape(patch, (self.patch_size, self.patch_size, 3))
            plt.imshow(utils.img_to_array(patch_img))
            plt.axis("off")
        plt.show()

        # Return the index chosen to validate it outside the method.
        return idx

    # taken from https://stackoverflow.com/a/58082878/10319735
    def reconstruct_from_patch(self, patch):
        # This utility function takes patches from a *single* image and
        # reconstructs it back into the image. This is useful for the train
        # monitor callback.
        num_patches = patch.shape[0]
        patch = ops.transpose(
            patch,
            axes=[1, 0],
        )
        reconstructed = ops.reshape(
            patch, (self.patch_size, self.patch_size, 3, num_patches)
        )
        reconstructed = ops.transpose(
            reconstructed,
            axes=[3, 0, 1, 2],
        )
        return reconstructed

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "patch_size": self.patch_size,
            }
        )
        return config

    def get_build_config(self):
        return {
            "image_size": self.image_size,
        }

    def build_from_config(self, config):
        self.build((None, config["image_size"], config["image_size"], 3))

    # def compute_output_shape(self, input_shape):
    #    return (input_shape[0], self.num_patches, self.patch_size * self.patch_size * 3)


@saving.register_keras_serializable(
    package="my_custom_package",
)
class RGB_PatchEncoder(layers.Layer):
    def __init__(
        self,
        num_patches: int,
        patch_size: int,
        projection_dim: int,
        mask_proportion: float = None,
        dec_mask_proportion=None,
        downstream: bool = False,
        seed: int = 42,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.projection_dim = projection_dim
        self.mask_proportion = mask_proportion
        self.dec_mask_proportion = dec_mask_proportion
        self.seed = seed
        self.downstream = downstream

    def build(self, input_shape):
        # self.cls_token = keras.Variable(
        #    shape=(1, self.projection_dim),
        #    trainable=True,
        #    initializer=initializers.RandomNormal(seed=self.seed),
        #    dtype="float32",
        #    name="cls_token",
        # )
        # This is a trainable mask token initialized randomly from a normal
        # distribution.
        self.mask_token = self.add_weight(
            shape=(1, self.patch_size * self.patch_size * 3),
            initializer=initializers.RandomNormal(seed=self.seed),
            trainable=True,
            dtype="float32",
            name="mask_token",
        )

        self.pos_embedding = layers.Embedding(
            input_dim=self.num_patches, output_dim=self.projection_dim, name="pos_emb"
        )

        # self.position_embedding.build((None, self.num_patches + 1))
        self.pos_embedding.build((None, self.num_patches))

        self.projection = layers.Dense(units=self.projection_dim, name="projection")

        self.projection.build((None, None, self.patch_size * self.patch_size * 3))

        self.num_mask = int(self.mask_proportion * self.num_patches)
        if self.dec_mask_proportion is not None:
            self.dec_num_mask = int(self.dec_mask_proportion * self.num_mask)

        super().build(input_shape)

    def call(self, patches):
        # Get the positional embeddings.

        batch_size = ops.shape(patches)[0]
        # cls_token = ops.tile(self.cls_token, repeats=[batch_size, 1])
        # cls_token = ops.reshape(cls_token, (batch_size, 1, self.projection_dim))

        # positions = ops.arange(start=0, stop=self.num_patches + 1, step=1)
        positions = ops.arange(start=0, stop=self.num_patches, step=1)
        pos_embeddings = self.pos_embedding(ops.expand_dims(positions, axis=0))

        pos_embeddings = ops.tile(
            pos_embeddings, [batch_size, 1, 1]
        )  # (B, num_patches, projection_dim)

        patch_embeddings = self.projection(patches)

        # Add the class token to the patch embeddings.
        # patch_embeddings = ops.concatenate(
        #    [cls_token, patch_embeddings], axis=1
        # )  # (B, num_patches + 1, projection_dim)

        # Embed the patches.
        patch_embeddings = patch_embeddings + pos_embeddings

        if self.downstream:
            return patch_embeddings
        else:
            mask_indices, unmask_indices = self.get_random_indices(batch_size)
            # The encoder input is the unmasked patch embeddings. Here we gather
            # all the patches that should be unmasked.
            unmasked_embeddings = tf.gather(
                patch_embeddings, unmask_indices, axis=1, batch_dims=1
            )  # (B, unmask_numbers, projection_dim)

            # Get the unmasked and masked position embeddings. We will need them
            # for the decoder.
            unmasked_positions = tf.gather(
                pos_embeddings, unmask_indices, axis=1, batch_dims=1
            )  # (B, unmask_numbers, projection_dim)
            masked_positions = tf.gather(
                pos_embeddings, mask_indices, axis=1, batch_dims=1
            )  # (B, mask_numbers, projection_dim)

            if self.dec_mask_proportion is not None:
                mask_tokens = ops.repeat(
                    self.mask_token, repeats=self.dec_num_mask, axis=0
                )
            else:
                # Repeat the mask token number of mask times.
                # Mask tokens replace the masks of the image.
                mask_tokens = ops.repeat(self.mask_token, repeats=self.num_mask, axis=0)

            mask_tokens = ops.repeat(
                ops.expand_dims(mask_tokens, axis=0), repeats=batch_size, axis=0
            )

            # Get the masked embeddings for the tokens.
            masked_embeddings = self.projection(mask_tokens) + masked_positions
            return (
                unmasked_embeddings,  # Input to the encoder.
                masked_embeddings,  # First part of input to the decoder.
                unmasked_positions,  # Added to the encoder outputs.
                mask_indices,  # The indices that were masked.
                unmask_indices,  # The indices that were unmaksed.
            )

    def get_random_indices(self, batch_size):
        # Create random indices from a uniform distribution and then split
        # it into mask and unmask indices.
        rand_indices = ops.argsort(
            random.uniform(shape=(batch_size, self.num_patches)), axis=-1
        )
        mask_indices = rand_indices[:, : self.num_mask]
        unmask_indices = rand_indices[:, self.num_mask :]

        subset_mask_indices = mask_indices[:, : self.dec_num_mask]
        return subset_mask_indices, unmask_indices

    def generate_masked_image(self, patches, unmask_indices, idx=None):
        if idx is None:
            # Choose a random patch and it corresponding unmask index.
            idx = np.random.choice(patches.shape[0])

        patch = patches[idx]
        unmask_index = unmask_indices[idx]

        # Build a numpy array of same shape as patch.
        new_patch = np.zeros_like(patch)

        # Iterate of the new_patch and plug the unmasked patches.
        count = 0
        for i in range(unmask_index.shape[0]):
            new_patch[unmask_index[i]] = patch[unmask_index[i]]
        return new_patch, idx

    # def load_own_variables(self, store):

    #    self.cls_token.assign(store["rgb_cls_token"])
    #    self.mask_token.assign(store["rgb_mask_token"])

    # Load the remaining weights
    # for i, v in enumerate(self.weights):
    #    v.assign(store[f"{i}"])

    # def save_own_variables(self, store):
    #    # super().save_own_variables(store)
    #    store["rgb_cls_token"] = self.cls_token.numpy()
    #    store["rgb_mask_token"] = self.mask_token.numpy()

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_patches": self.num_patches,
                "patch_size": self.patch_size,
                "projection_dim": self.projection_dim,
                "mask_proportion": self.mask_proportion,
                "dec_mask_proportion": self.dec_mask_proportion,
                "downstream": self.downstream,
                "seed": self.seed,
            }
        )
        return config

    def get_build_config(self):
        build_config = {
            "patch_size": self.patch_size,
        }
        return build_config

    def build_from_config(self, config):
        self.build((None, None, config["patch_size"] * config["patch_size"] * 3))


@saving.register_keras_serializable(package="my_custom_package")
class HS_PatchExtractor(layers.Layer):
    """
    Every channel of the input image is a patch.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape) -> None:
        (_, self.patch_size, self.patch_size, self.num_patches) = input_shape
        super().build((None, self.patch_size, self.patch_size, self.num_patches))

    def call(self, x):
        x = ops.transpose(x, axes=[0, 3, 1, 2])

        x = ops.reshape(
            x,
            (
                -1,
                self.num_patches,
                self.patch_size * self.patch_size,
            ),
        )
        return x

    def decoder_call(self, x):
        batch_size = ops.shape(x)[0]
        num_patches = ops.shape(x)[1]

        x = ops.reshape(
            x,
            (
                batch_size,
                num_patches,
                self.patch_size * self.patch_size,
            ),
        )

        return x

    def show_patched_image(self, images, patches):
        idx = np.random.choice(patches.shape[0])
        print(f"Index selected: {idx}.")

        n = 25
        plt.figure(figsize=(6, 4))
        plt.suptitle("Original Image")

        image = images[idx]
        num_channels = image.shape[-1]
        for i in range(num_channels):
            ax = plt.subplot(n, n, i + 1)

            plt.imshow(image[..., i])
            plt.axis("off")

        plt.figure(figsize=(6, 4))
        plt.suptitle("Patches")
        for i, patch in enumerate(patches[idx]):
            ax = plt.subplot(n, n, i + 1)
            patch_img = ops.reshape(patch, (self.patch_size, self.patch_size))
            plt.imshow(utils.img_to_array(patch_img))
            plt.axis("off")

        return idx

    def reconstruct_from_patch(self, patch):
        num_patches = patch.shape[0]

        patch = ops.transpose(
            patch,
            axes=[1, 0],
        )
        reconstructed = ops.reshape(
            patch, (self.patch_size, self.patch_size, num_patches)
        )
        return reconstructed

    def get_build_config(self):
        return {
            "patch_size": self.patch_size,
        }

    def build_from_config(self, config):
        self.build((None, config["patch_size"], config["patch_size"], None))


@saving.register_keras_serializable(package="my_custom_package")
class HS_PatchEncoder(layers.Layer):
    def __init__(
        self,
        num_patches,
        patch_size,
        projection_dim,
        mask_proportion=None,
        dec_mask_proportion=None,
        downstream=False,
        seed=42,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.projection_dim = projection_dim
        self.mask_proportion = mask_proportion
        self.dec_mask_proportion = dec_mask_proportion
        self.downstream = downstream
        self.from_scratch = False
        self.indices = None
        self.seed = seed

    def build(self, input_shape):
        # self.cls_token = keras.Variable(
        #    shape=[1, self.projection_dim],
        #    trainable=True,
        #    initializer=initializers.RandomNormal(seed=self.seed),
        #    dtype="float32",
        #    name="cls_token",
        # )

        self.mask_token = self.add_weight(
            shape=[1, self.patch_size * self.patch_size * 1],
            trainable=True,
            initializer=initializers.RandomNormal(seed=self.seed),
            dtype="float32",
            name="mask_token",
        )
        self.pos_embedding = layers.Embedding(
            input_dim=self.num_patches,
            output_dim=self.projection_dim,
            name="pos_emb",
        )

        self.pos_embedding.build((None, self.num_patches))

        self.num_mask = int(self.mask_proportion * self.num_patches)
        if self.dec_mask_proportion is not None:
            self.dec_num_mask = int(self.dec_mask_proportion * self.num_mask)

        self.projection = layers.Dense(units=self.projection_dim, name="projection")
        self.projection.build((None, None, self.patch_size * self.patch_size))
        super().build(input_shape)

    def call(self, patches, training=False):
        patches = ops.convert_to_tensor(patches)
        batch_size = ops.shape(patches)[0]

        # cls_token = ops.tile(self.cls_token, repeats=[batch_size, 1])
        # cls_token = ops.reshape(cls_token, (batch_size, 1, self.projection_dim))

        # positions = ops.arange(start=0, stop=self.num_patches + 1, step=1)
        positions = ops.arange(start=0, stop=self.num_patches, step=1)

        pos_embeddings = self.pos_embedding(
            ops.expand_dims(positions, axis=0), training=training
        )

        pos_embeddings = ops.tile(pos_embeddings, [batch_size, 1, 1])

        patch_embeddings = self.projection(patches, training=training)

        # patch_embeddings = ops.concatenate([cls_token, patch_embeddings], axis=1)

        if self.downstream:
            if self.indices is not None:
                pos_embeddings = ops.take(pos_embeddings, self.indices, axis=1)
            return patch_embeddings + pos_embeddings

        else:
            patch_embeddings = patch_embeddings + pos_embeddings
            mask_indices, unmask_indices = self.get_random_indices(batch_size)

            unmasked_embeddings = tf.gather(
                patch_embeddings, unmask_indices, axis=1, batch_dims=1
            )

            unmasked_positions = tf.gather(
                pos_embeddings, unmask_indices, axis=1, batch_dims=1
            )

            masked_positions = tf.gather(
                pos_embeddings, mask_indices, axis=1, batch_dims=1
            )

            if self.dec_mask_proportion is not None:
                mask_tokens = ops.repeat(
                    self.mask_token, repeats=self.dec_num_mask, axis=0
                )
            else:
                mask_tokens = ops.repeat(self.mask_token, repeats=self.num_mask, axis=0)

            mask_tokens = ops.repeat(
                ops.expand_dims(mask_tokens, axis=0), repeats=batch_size, axis=0
            )

            masked_embeddings = (
                self.projection(mask_tokens, training=training) + masked_positions
            )
            return (
                unmasked_embeddings,
                masked_embeddings,
                unmasked_positions,
                mask_indices,
                unmask_indices,
            )

    def get_random_indices(self, batch_size):
        rand_indices = ops.argsort(
            random.uniform(shape=(batch_size, self.num_patches), seed=self.seed),
            axis=-1,
        )

        mask_indices = rand_indices[:, : self.num_mask]
        unmask_indices = rand_indices[:, self.num_mask :]
        # Store the indices for the decoder.
        subset_mask_indices = mask_indices[:, : self.dec_num_mask]
        return subset_mask_indices, unmask_indices

    def generate_masked_image(self, patches, unmask_indices, idx=None):
        if idx is None:
            idx = np.random.choice(patches.shape[0])
        patch = patches[idx]
        unmask_index = unmask_indices[idx]
        new_patch = np.zeros_like(patch)
        count = 0
        for i in range(unmask_index.shape[0]):
            new_patch[unmask_index[i]] = patch[unmask_index[i]]
        return new_patch, idx

    def load_own_variables(self, store):
        #    self.cls_token.assign(store["hs_cls_token"])
        self.mask_token.assign(store["hs_mask_token"])

    # Load the remaining weights
    # for i, v in enumerate(self.weights):
    #    v.assign(store[f"{i}"])

    def save_own_variables(self, store):
        #    # super().save_own_variables(store)
        #    store["hs_cls_token"] = self.cls_token.numpy()
        store["hs_mask_token"] = self.mask_token.numpy()

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_patches": self.num_patches,
                "patch_size": self.patch_size,
                "projection_dim": self.projection_dim,
                "mask_proportion": self.mask_proportion,
                "dec_mask_proportion": self.dec_mask_proportion,
                "downstream": self.downstream,
                "seed": self.seed,
            }
        )
        return config


def create_transformer_encoder(
    projection_dim: int,
    num_heads: int,
    mlp_ratio: int,
    depth: int,
    drop_rate: float,
    attn_drop_rate: float,
    drop_path_rate: float,
    act_layer: str,
    norm_layer: str,
    name: str = "transformer_encoder",
    **kwargs,
):
    inputs = layers.Input(shape=(None, projection_dim))
    x = inputs
    for i in range(depth):
        x = ViTBlock(
            embed_dim=projection_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            use_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            act_layer=act_layer,
            name=f"transformer_block_{i}",
        )(x)

    norm_layer = norm_layer_factory(norm_layer)
    x = norm_layer(name="norm")(x)
    return models.Model(inputs, x, name=name)


def create_mlp_decoder(
    input_dim: int,
    projection_dim: int,
    output_shape,
    mlp_ratio: int,
    drop_rate: float,
    norm_layer: str,
    act_layer: str,
    name="mlp_decoder",
    **kwargs,
):
    inputs = layers.Input(shape=(output_shape[0], input_dim))
    x = inputs
    x = layers.Dense(units=projection_dim)(x)

    norm1 = norm_layer_factory(norm_layer)
    x = norm1(name="norm1")(x)
    x = MLP(
        hidden_dim=projection_dim * mlp_ratio,
        embed_dim=projection_dim,
        drop_rate=drop_rate,
        act_layer=act_layer,
    )(x)
    norm2 = norm_layer_factory(norm_layer)
    x = norm2(name="norm2")(x)
    x = layers.Flatten()(x)

    x = layers.Dense(
        units=output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3],
        activation="sigmoid",
        dtype="float32",
    )(x)

    if output_shape[3] == 1:
        x = layers.Reshape((output_shape[0:3]))(x)
    else:
        x = layers.Reshape((output_shape))(x)
    return models.Model(inputs, x, name=name)


def create_fusion_layer(
    projection_dim: int,
    num_heads: int,
    drop_rate: float,
    drop_path_rate: float,
    norm_layer: str,
    act_layer: str,
    name="fusion_layer",
    **kwargs,
):
    inputs = layers.Input(shape=(None, projection_dim))
    context = layers.Input(shape=(None, projection_dim))
    x = inputs

    norm1 = norm_layer_factory(norm_layer)
    x = norm1(name="norm1")(x)

    x = CrossAttention(
        num_heads=8,
        dim=projection_dim,
        attn_drop=drop_rate,
        norm_layer=norm_layer,
    )(x, context)

    norm2 = norm_layer_factory(norm_layer)
    x = norm2(name="norm2")(x)

    x = MLP(
        hidden_dim=projection_dim * 2,
        embed_dim=projection_dim,
        drop_rate=drop_rate,
        act_layer=act_layer,
    )(x)
    return models.Model([inputs, context], x, name=name)


class MLP(layers.Layer):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        hidden_dim: int,
        embed_dim: int,
        drop_rate: float,
        act_layer: str,
        kernel_initializer: str = "glorot_uniform",
        bias_initializer: str = "zeros",
        **kwargs,
    ):
        super().__init__(**kwargs)
        act_layer = act_layer_factory(act_layer)

        self.fc1 = layers.Dense(
            units=hidden_dim,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            name="fc1",
        )
        self.act = act_layer()
        self.drop1 = layers.Dropout(rate=drop_rate)
        self.fc2 = layers.Dense(
            units=embed_dim,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            name="fc2",
        )
        self.drop2 = layers.Dropout(rate=drop_rate)

    def call(self, x, training=False):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x, training=training)
        x = self.fc2(x)
        x = self.drop2(x, training=training)
        return x


class ViTBlock(layers.Layer):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float,
        use_bias: bool,
        drop_rate: float,
        attn_drop_rate: float,
        drop_path_rate: float,
        norm_layer: str,
        act_layer: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.use_bias = use_bias
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.norm_layer = norm_layer
        self.act_layer = act_layer

        norm_layer = norm_layer_factory(norm_layer)
        self.norm1 = norm_layer(name="norm1")

        self.attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim,
            dropout=attn_drop_rate,
            use_bias=use_bias,
            name="attn",
        )

        self.drop_path = keras_cv.layers.DropPath(rate=drop_path_rate)
        self.norm2 = norm_layer(name="norm2")
        self.mlp = MLP(
            hidden_dim=int(embed_dim * mlp_ratio),
            embed_dim=embed_dim,
            drop_rate=drop_rate,
            act_layer=act_layer,
            name="mlp",
        )

    def call(self, x, training=False, return_features=False):
        features = OrderedDict()
        shortcut = x
        x = self.norm1(x, training=training)
        x = self.attn(x, x, training=training, return_attention_scores=return_features)
        if return_features:
            x, mha_features = x
            features["attn"] = mha_features
        x = self.drop_path(x, training=training)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x, training=training)
        x = self.mlp(x, training=training)
        x = self.drop_path(x, training=training)
        x = x + shortcut
        return (x, features) if return_features else x


@saving.register_keras_serializable()
class MinMaxScaler(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        min_vals = ops.min(inputs, keepdims=True, axis=-1)
        max_vals = ops.max(inputs, keepdims=True, axis=-1)

        # Ensure that the range is not zero to avoid division by zero
        range_nonzero = ops.where(min_vals != max_vals, max_vals - min_vals, 1.0)

        # Normalize each pixel by subtracting the minimum and dividing by the range
        output = (inputs - min_vals) / range_nonzero

        return output


class ChannelReducer(layers.Layer):
    def __init__(self, keep_num_bands=100, num_bands=300, **kwargs):
        super(ChannelReducer, self).__init__(**kwargs)
        self.keep_num_bands = keep_num_bands
        self.num_bands = num_bands

    def call(self, inputs):
        if self.num_bands > self.keep_num_bands:
            # Select bands based on conditions
            match self.keep_num_bands:
                case 10:
                    output = self.select_bands(inputs, 30, 20, 10)
                case 50:
                    output = self.select_bands(inputs, 6, 4, 2)
                case 100:
                    output = self.select_bands(inputs, 3, 2)
                case 150:
                    output = self.select_bands(inputs, 2, 1)
                case _:
                    output = inputs
        else:
            output = inputs

        return output

    def select_bands(self, inputs, *factors):
        """
        Select bands based on the given factors for different cases.
        """
        if self.num_bands == 300:
            output = inputs[..., :: factors[0]]
        elif self.num_bands == 200:
            output = inputs[..., :: factors[1]]
        elif self.num_bands == 100:
            output = inputs[..., :: factors[2]]
        else:
            output = inputs

        return output

    def get_config(self):
        config = super().get_config()
        config.update({"keep_num_bands": self.keep_num_bands})
        return config

    # def compute_output_shape(self, input_shape):
    #    return (input_shape[0], input_shape[1], input_shape[2], self.keep_num_bands)


class UnitNorm(layers.Layer):
    def __init__(self, **kwargs):
        super(UnitNorm, self).__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        output = ops.math.l2_normalize(inputs, axis=-1)
        return output

    def get_config(self):
        config = super().get_config()
        return config

    # def compute_output_shape(self, input_shape):
    #    return input_shape


@saving.register_keras_serializable()
class Resizing(layers.Layer):
    def __init__(self, target_size, pad_to_aspect_ratio=True, **kwargs):
        super().__init__(**kwargs)
        self.target_size = target_size
        self.pad_to_aspect_ratio = pad_to_aspect_ratio

    def call(self, inputs):
        if self.pad_to_aspect_ratio:
            # check if image is smaller than target size
            image_shape = ops.shape(inputs)
            if (
                image_shape[1] < self.target_size[0]
                or image_shape[2] < self.target_size[1]
            ):
                # use pad and resize separately
                # pad image
                pad_width = self.target_size[0] - image_shape[1]
                pad_height = self.target_size[1] - image_shape[2]
                pad_width_left = pad_width // 2
                pad_width_right = pad_width - pad_width_left
                pad_height_top = pad_height // 2
                pad_height_bottom = pad_height - pad_height_top
                paddings = [
                    [0, 0],
                    [pad_width_left, pad_width_right],
                    [pad_height_top, pad_height_bottom],
                    [0, 0],
                ]
                inputs = ops.pad(inputs, paddings, "CONSTANT", constant_values=1)

        # resize image
        output = ops.image.resize(inputs, self.target_size)

        return output

    def get_config(self):
        config = super().get_config()
        config.update({"target_size": self.target_size, "pad_to_aspect_ratio": True})
        return config


@saving.register_keras_serializable(name="cross_attention")
class CrossAttention(layers.Layer):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        attn_drop: float,
        norm_layer: str,
        name="cross_attention",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.attn_drop = attn_drop
        self.norm_layer = norm_layer

        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=dim, dropout=attn_drop
        )
        norm_layer = norm_layer_factory(norm_layer)
        self.norm1 = norm_layer(name="norm1")
        self.add1 = layers.Add()

    def call(self, x, context):
        attn_output, attn_scores = self.mha(
            query=x, key=context, value=context, return_attention_scores=True
        )

        # Cache the attention scores for plotting later.
        self.last_attn_scores = attn_scores

        x = self.add1([x, attn_output])
        x = self.norm1(x)

        return x
