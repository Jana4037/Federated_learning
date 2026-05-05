# """
# centralized.py - Train on ALL data combined (no federation, no privacy).

# This is the upper-bound baseline: what accuracy is achievable when every
# bookstore pools its data together?  We compare this to the federated result.
# If they are close, federated learning achieves privacy almost for free.

# Run with:
#     python centralized.py

# Recommended order:
#     python data_preparation.py   ← do once
#     python centralized.py        ← establish the baseline
#     python simulation.py         ← run federated and compare
# """

# import os
# import json
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt

# import tensorflow as tf
# from tensorflow import keras

# from data_preparation import load_all_partitions, load_meta, train_test_split_df
# from model import build_ncf_model, make_dataset

# # ── Settings ─────────────────────────────────────────────────────────────────
# EMBED_DIM     = 16
# MLP_LAYERS    = [64, 32]
# LR            = 5e-4       # same as simulation / client
# L2_REG        = 1e-4
# BATCH_SIZE    = 512
# NUM_EPOCHS    = 15         # more epochs OK here — we have all the data
# TEST_FRACTION = 0.2


# def main():
#     tf.random.set_seed(42)

#     df   = load_all_partitions()
#     meta = load_meta()

#     print(f"\nTotal ratings : {len(df):,}")
#     print(f"Users         : {meta['num_users']:,}")
#     print(f"Books         : {meta['num_books']:,}\n")

#     train_df, test_df = train_test_split_df(df, TEST_FRACTION)

#     # Add .prefetch here — safe in the standalone (non-simulation) context
#     train_ds = make_dataset(train_df, batch_size=BATCH_SIZE, shuffle=True).prefetch(tf.data.AUTOTUNE)
#     test_ds  = make_dataset(test_df,  batch_size=BATCH_SIZE * 4).prefetch(tf.data.AUTOTUNE)

#     model = build_ncf_model(
#         num_users=meta["num_users"],
#         num_books=meta["num_books"],
#         embed_dim=EMBED_DIM,
#         mlp_layers=MLP_LAYERS,
#         learning_rate=LR,
#         l2_reg=L2_REG,
#     )
#     model.summary()

#     os.makedirs("data", exist_ok=True)
#     callbacks = [
#         keras.callbacks.ModelCheckpoint(
#             filepath="data/centralized_model.keras",
#             monitor="val_rmse",
#             save_best_only=True,
#             verbose=1,
#         ),
#         keras.callbacks.EarlyStopping(
#             monitor="val_rmse",
#             patience=4,
#             restore_best_weights=True,
#             verbose=1,
#         ),
#         keras.callbacks.ReduceLROnPlateau(
#             monitor="val_rmse",
#             factor=0.5,
#             patience=2,
#             min_lr=1e-6,
#             verbose=1,
#         ),
#     ]

#     print("=" * 55)
#     print("  Centralized training")
#     print("=" * 55)

#     history = model.fit(
#         train_ds,
#         validation_data=test_ds,
#         epochs=NUM_EPOCHS,
#         callbacks=callbacks,
#         verbose=1,
#     )

#     print("\nFinal evaluation on test set:")
#     results    = model.evaluate(test_ds, verbose=1)
#     final_rmse = results[1]
#     final_mae  = results[2]
#     best_rmse  = min(history.history["val_rmse"])

#     print(f"\n  Best val RMSE : {best_rmse:.4f}")
#     print(f"  Final RMSE    : {final_rmse:.4f}")
#     print(f"  Final MAE     : {final_mae:.4f}")
#     print("  Model saved → data/centralized_model.keras")

#     with open("data/centralized_results.json", "w") as f:
#         json.dump({"best_rmse": best_rmse,
#                    "final_rmse": final_rmse,
#                    "final_mae": final_mae}, f, indent=2)

#     # ── Plot ──────────────────────────────────────────────────────────────────
#     epochs_ran  = range(1, len(history.history["rmse"]) + 1)
#     os.makedirs("data/plots", exist_ok=True)

#     fig, axes = plt.subplots(1, 2, figsize=(12, 4))

#     axes[0].plot(epochs_ran, history.history["rmse"],     marker="o", label="Train RMSE")
#     axes[0].plot(epochs_ran, history.history["val_rmse"], marker="s", label="Val RMSE")
#     axes[0].set_title("Centralized — RMSE")
#     axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("RMSE")
#     axes[0].legend(); axes[0].grid(True, alpha=0.3)

#     axes[1].plot(epochs_ran, history.history["mae"],     marker="o", label="Train MAE")
#     axes[1].plot(epochs_ran, history.history["val_mae"], marker="s", label="Val MAE")
#     axes[1].set_title("Centralized — MAE")
#     axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("MAE")
#     axes[1].legend(); axes[1].grid(True, alpha=0.3)

#     plt.tight_layout()
#     plt.savefig("data/plots/centralized_training.png", dpi=120)
#     plt.close()
#     print("  Plot saved → data/plots/centralized_training.png")


# if __name__ == "__main__":
#     main()

import json
import os
import tensorflow as tf
from tensorflow import keras

from data_preparation import load_all_partitions, load_meta
from model import build_model, make_dataset


# ── Config ─────────────────────────────
BATCH_SIZE = 512
EPOCHS     = 30


if __name__ == "__main__":

    tf.random.set_seed(42)

    df   = load_all_partitions()
    meta = load_meta()

    print(f"Total ratings: {len(df):,}")

    # ── Fix: calculate real vocab sizes from the actual data ───────────────
    # meta["num_users"] and meta["num_books"] only reflect the top-N cities
    # used during partitioning. The combined data may have higher indices.
    num_users = int(df["user_idx"].max()) + 1
    num_books = int(df["book_idx"].max()) + 1

    print(f"num_users (actual): {num_users:,}")
    print(f"num_books (actual): {num_books:,}")

    train_df = df.sample(frac=0.8, random_state=42)
    test_df  = df.drop(train_df.index)

    train_ds = make_dataset(train_df, BATCH_SIZE, shuffle=True)
    test_ds  = make_dataset(test_df,  BATCH_SIZE)

    # Use actual sizes instead of meta sizes
    model = build_model(num_users, num_books)

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    )

    print("\nStarting centralized training...\n")

    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=EPOCHS,
        callbacks=[early_stop],
        verbose=1,
    )

    loss, mae = model.evaluate(test_ds, verbose=0)
    best_val_mae = min(history.history["val_mae"])

    print("\n" + "=" * 40)
    print("Centralized Results")
    print("=" * 40)
    print(f"Stopped at epoch : {early_stop.stopped_epoch + 1}")
    print(f"Best val MAE     : {best_val_mae:.4f}")
    print(f"Final MAE        : {mae:.4f}")
    print(f"Final Loss (MSE) : {loss:.4f}")

    os.makedirs("data", exist_ok=True)

    result = {
        "best_mae":   float(best_val_mae),
        "final_mae":  float(mae),
        "final_loss": float(loss),
        "num_users":  num_users,
        "num_books":  num_books,
    }

    with open("data/centralized_results.json", "w") as f:
        json.dump(result, f, indent=2)

    print("\nSaved → data/centralized_results.json")