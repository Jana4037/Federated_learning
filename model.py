# """
# model.py - Neural Collaborative Filtering (NCF) model in TensorFlow/Keras.

# NCF combines two ideas:
#   1. GMF branch : element-wise product of user and book embeddings
#                   (like classic matrix factorisation).
#   2. MLP branch : concatenated embeddings through a small dense network
#                   to learn non-linear patterns.

# Both outputs are combined via Dense(1, sigmoid) and linearly rescaled to [1,10].

# WHY THESE CHANGES vs. the broken original
# ------------------------------------------
# Problem 1 — Flat RMSE ~7.75 every round (no learning)
#   Root cause : The Lambda(clip_by_value) layer silently caused Flower's weight
#                serialisation to produce mismatched tensor shapes on certain TF
#                versions. The server received updates, averaged them, but when
#                set_weights() loaded them back into a freshly-built model the
#                Lambda layer's "weights" were counted differently, so the real
#                trainable params (embeddings, Dense kernels) were overwritten
#                with the WRONG arrays. The model effectively restarted from
#                random weights every round — hence the identical RMSE each time.
#   Fix : Replace Lambda+clip with Dense(sigmoid) + Rescaling. Both layers are
#         pure Keras ops, have deterministic weight counts, and serialise safely.

# Problem 2 — Embedding explosion / dead gradients
#   Root cause : random_normal init with no regularisation → large initial norms
#                → gradients saturate early → no learning.
#   Fix : glorot_uniform init + L2 regulariser + gradient clipnorm=1.0.

# Problem 3 — Optimizer state reset kills momentum in federated rounds
#   Root cause : each client call to model.fit() after set_weights() starts a
#                fresh Adam state (m=0, v=0). With a cold optimizer and only
#                3 local epochs the update is dominated by the first step.
#   Fix : lower the default LR to 5e-4 (safer cold start) and add BatchNorm
#         so each layer sees normalised inputs regardless of optimizer warmth.
# """

# import numpy as np
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers, regularizers

# RATING_MIN = 1.0
# RATING_MAX = 10.0


# # ── Model ─────────────────────────────────────────────────────────────────────

# def build_ncf_model(
#     num_users: int,
#     num_books: int,
#     embed_dim: int = 16,
#     mlp_layers: list = None,
#     learning_rate: float = 5e-4,
#     l2_reg: float = 1e-4,
# ) -> keras.Model:
#     """
#     Build and compile the NCF model.

#     Output
#     ------
#     rating = RATING_MIN + (RATING_MAX - RATING_MIN) * sigmoid(logit)
#            = 1 + 9 * sigmoid(logit)   →  always in (1, 10)

#     This is fully differentiable, has no extra "phantom" weights, and
#     serialises correctly through Flower's NumPy parameter protocol.

#     Parameters
#     ----------
#     num_users     : total number of unique users (embedding vocab size)
#     num_books     : total number of unique books (embedding vocab size)
#     embed_dim     : embedding dimension shared by GMF and MLP branches
#     mlp_layers    : hidden-unit counts for MLP, e.g. [64, 32]
#     learning_rate : Adam LR — 5e-4 is a safe default for cold-start federated
#     l2_reg        : L2 regularisation on embedding tables
#     """
#     if mlp_layers is None:
#         mlp_layers = [64, 32]

#     reg = regularizers.l2(l2_reg)

#     # ── Inputs ────────────────────────────────────────────────────────────────
#     user_input = keras.Input(shape=(1,), dtype="int32", name="user_input")
#     book_input = keras.Input(shape=(1,), dtype="int32", name="book_input")

#     # ── GMF branch ────────────────────────────────────────────────────────────
#     gmf_u = layers.Embedding(num_users, embed_dim,
#                              embeddings_initializer="glorot_uniform",
#                              embeddings_regularizer=reg,
#                              name="gmf_user_emb")(user_input)   # (B,1,D)
#     gmf_b = layers.Embedding(num_books, embed_dim,
#                              embeddings_initializer="glorot_uniform",
#                              embeddings_regularizer=reg,
#                              name="gmf_book_emb")(book_input)

#     gmf_out = layers.Multiply(name="gmf_mul")([gmf_u, gmf_b])
#     gmf_out = layers.Flatten(name="gmf_flat")(gmf_out)          # (B, D)

#     # ── MLP branch ────────────────────────────────────────────────────────────
#     mlp_u = layers.Embedding(num_users, embed_dim,
#                              embeddings_initializer="glorot_uniform",
#                              embeddings_regularizer=reg,
#                              name="mlp_user_emb")(user_input)
#     mlp_b = layers.Embedding(num_books, embed_dim,
#                              embeddings_initializer="glorot_uniform",
#                              embeddings_regularizer=reg,
#                              name="mlp_book_emb")(book_input)

#     mlp_out = layers.Concatenate(name="mlp_cat")(
#         [layers.Flatten(name="mlp_u_flat")(mlp_u),
#          layers.Flatten(name="mlp_b_flat")(mlp_b)]
#     )                                                           # (B, 2*D)

#     for i, units in enumerate(mlp_layers):
#         mlp_out = layers.Dense(units, use_bias=False,
#                                kernel_initializer="glorot_uniform",
#                                kernel_regularizer=reg,
#                                name=f"mlp_dense_{i}")(mlp_out)
#         # BatchNorm stabilises training when optimizer state resets each round
#         mlp_out = layers.BatchNormalization(name=f"mlp_bn_{i}")(mlp_out)
#         mlp_out = layers.Activation("relu", name=f"mlp_relu_{i}")(mlp_out)
#         mlp_out = layers.Dropout(0.3, name=f"mlp_drop_{i}")(mlp_out)

#     # ── Combine & predict ─────────────────────────────────────────────────────
#     combined = layers.Concatenate(name="combine")([gmf_out, mlp_out])

#     # sigmoid logit → (0, 1)
#     logit = layers.Dense(1, activation="sigmoid",
#                          kernel_initializer="glorot_uniform",
#                          name="logit")(combined)                # (B, 1)

#     # Rescale (0,1) → (RATING_MIN, RATING_MAX) — pure Keras op, no Lambda
#     output = layers.Rescaling(
#         scale=RATING_MAX - RATING_MIN,
#         offset=RATING_MIN,
#         name="rescale",
#     )(logit)                                                     # (B, 1)

#     output = layers.Flatten(name="output")(output)              # (B,)

#     model = keras.Model(inputs=[user_input, book_input],
#                         outputs=output, name="NCFModel")

#     model.compile(
#         optimizer=keras.optimizers.Adam(
#             learning_rate=learning_rate,
#             clipnorm=1.0,       # prevents gradient explosions
#         ),
#         loss="mse",
#         metrics=[
#             keras.metrics.RootMeanSquaredError(name="rmse"),
#             keras.metrics.MeanAbsoluteError(name="mae"),
#         ],
#     )
#     return model


# # ── Flower weight helpers ─────────────────────────────────────────────────────

# def get_weights(model: keras.Model) -> list:
#     """
#     Return ALL model weights (trainable + non-trainable) as NumPy arrays.

#     Non-trainable weights include BatchNormalization running statistics
#     (moving_mean, moving_variance).  Including them ensures the global model
#     stays consistent — if we only sent trainable weights, BN statistics would
#     diverge across clients and the global model would evaluate poorly.
#     """
#     return model.get_weights()


# def set_weights(model: keras.Model, weights: list) -> None:
#     """
#     Load aggregated weights from the server into this model.

#     Weights are matched by position, so client and server models MUST be
#     built with identical arguments to build_ncf_model().
#     """
#     model.set_weights(weights)


# # ── tf.data factory ───────────────────────────────────────────────────────────

# def make_dataset(df, batch_size: int = 256, shuffle: bool = False) -> tf.data.Dataset:
#     """
#     Build a tf.data.Dataset from a DataFrame with (user_idx, book_idx, rating).

#     Returns
#     -------
#     A batched Dataset of ({"user_input": ..., "book_input": ...}, rating) pairs.

#     Note: .prefetch() is intentionally omitted. Inside Flower's single-process
#     simulation, background prefetch threads from multiple virtual clients can
#     race and cause non-deterministic behaviour. For the real server/client
#     split (server.py / client.py) the caller may add .prefetch() after this
#     call if desired.
#     """
#     users   = df["user_idx"].values.astype("int32")
#     books   = df["book_idx"].values.astype("int32")
#     ratings = df["rating"].values.astype("float32")

#     ds = tf.data.Dataset.from_tensor_slices(
#         ({"user_input": users, "book_input": books}, ratings)
#     )
#     if shuffle:
#         ds = ds.shuffle(buffer_size=min(len(df), 50_000),
#                         reshuffle_each_iteration=True)
#     return ds.batch(batch_size)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

RATING_MIN = 1.0
RATING_MAX = 10.0


def build_model(num_users, num_books, embed_dim=16):

    user_input = keras.Input(shape=(1,), name="user_input")
    book_input = keras.Input(shape=(1,), name="book_input")

    user_emb = layers.Embedding(num_users, embed_dim)(user_input)
    book_emb = layers.Embedding(num_books, embed_dim)(book_input)

    user_vec = layers.Flatten()(user_emb)
    book_vec = layers.Flatten()(book_emb)

    x = layers.Dot(axes=1)([user_vec, book_vec])
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.3)(x)          # add this — randomly drops 30% of neurons
    x = layers.Dense(1, activation="sigmoid")(x)

    output = layers.Rescaling(
        scale=RATING_MAX - RATING_MIN,
        offset=RATING_MIN
    )(x)

    model = keras.Model(inputs=[user_input, book_input], outputs=output)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"]
    )
    return model


def get_weights(model):
    return model.get_weights()

def set_weights(model, weights):
    model.set_weights(weights)


def make_dataset(df, batch_size=256, shuffle=False):
    users   = df["user_idx"].values.astype("int32")
    books   = df["book_idx"].values.astype("int32")
    ratings = df["rating"].values.astype("float32")

    ds = tf.data.Dataset.from_tensor_slices(
        ({"user_input": users, "book_input": books}, ratings)
    )
    if shuffle:
        ds = ds.shuffle(len(df))
    return ds.batch(batch_size)