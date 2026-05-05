# """
# client.py - A federated learning client (one bookstore node) using TensorFlow.

# Each client:
#   1. Loads only its own local data — never shares raw ratings.
#   2. Receives the global model weights from the server each round.
#   3. Trains locally for LOCAL_EPOCHS using Keras model.fit().
#   4. Sends the updated weights back to the server.

# Run with (after starting server.py in a separate terminal):
#     python client.py --node-id city_london
#     python client.py --node-id city_paris

# List available node IDs:
#     python client.py --list-nodes
# """

# import argparse
# import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# import tensorflow as tf
# import flwr as fl

# from data_preparation import load_partition, load_meta, train_test_split_df
# from model import build_ncf_model, make_dataset, get_weights, set_weights

# # ── Settings — keep in sync with simulation.py and server.py ─────────────────
# EMBED_DIM     = 16
# MLP_LAYERS    = [64, 32]
# LR            = 5e-4       # matches model.py default — safe cold-start LR
# L2_REG        = 1e-4
# BATCH_SIZE    = 256
# LOCAL_EPOCHS  = 2          # 2 epochs: less local overfitting
# TEST_FRACTION = 0.2


# # ── Flower client ─────────────────────────────────────────────────────────────

# class BookstoreClient(fl.client.NumPyClient):
#     """
#     One bookstore = one Flower client.

#     The model is built once at startup and reused across all rounds.
#     Rebuilding it inside fit/evaluate would risk TF tracing the graph
#     differently and making set_weights() load values into the wrong tensors.
#     """

#     def __init__(self, node_id, model, train_ds, test_ds, n_train, n_test):
#         self.node_id  = node_id
#         self.model    = model       # persistent — created once in main()
#         self.train_ds = train_ds
#         self.test_ds  = test_ds
#         self.n_train  = n_train
#         self.n_test   = n_test

#     def get_parameters(self, config):
#         return get_weights(self.model)

#     def fit(self, parameters, config):
#         """Load global weights → train locally → return updated weights."""
#         set_weights(self.model, parameters)

#         hist = self.model.fit(self.train_ds, epochs=LOCAL_EPOCHS, verbose=0)

#         last_rmse = hist.history.get("rmse", [None])[-1]
#         last_mae  = hist.history.get("mae",  [None])[-1]
#         if last_rmse is not None:
#             print(f"  [{self.node_id}] {LOCAL_EPOCHS} epoch(s) — "
#                   f"train RMSE={last_rmse:.4f}  MAE={last_mae:.4f}")
#         else:
#             print(f"  [{self.node_id}] Local training complete ({LOCAL_EPOCHS} epochs).")

#         return get_weights(self.model), self.n_train, {}

#     def evaluate(self, parameters, config):
#         """Load global weights → evaluate on local test set → report metrics."""
#         set_weights(self.model, parameters)

#         results = self.model.evaluate(self.test_ds, verbose=0)
#         loss, rmse, mae = float(results[0]), float(results[1]), float(results[2])

#         print(f"  [{self.node_id}] Eval — RMSE={rmse:.4f}  MAE={mae:.4f}")
#         return loss, self.n_test, {"rmse": rmse, "mae": mae}


# # ── Main ──────────────────────────────────────────────────────────────────────

# def main():
#     parser = argparse.ArgumentParser(description="Federated bookstore client")
#     parser.add_argument("--node-id", default=None,
#                         help="Partition node ID, e.g. city_london")
#     parser.add_argument("--server-address", default="127.0.0.1:8081",
#                         help="Flower server address (default: 127.0.0.1:8081)")
#     parser.add_argument("--list-nodes", action="store_true",
#                         help="Print available node IDs and exit.")
#     args = parser.parse_args()

#     meta = load_meta()

#     if args.list_nodes:
#         print("\nAvailable node IDs:")
#         for nid in meta["node_ids"]:
#             print(f"  {nid}")
#         return

#     if args.node_id is None:
#         parser.error("--node-id is required. Use --list-nodes to see options.")
#     if args.node_id not in meta["node_ids"]:
#         raise ValueError(f"Unknown node_id '{args.node_id}'. "
#                          "Run with --list-nodes to see valid options.")

#     tf.random.set_seed(42)

#     df = load_partition(args.node_id)
#     train_df, test_df = train_test_split_df(df, TEST_FRACTION)

#     train_ds = make_dataset(train_df, batch_size=BATCH_SIZE, shuffle=True)
#     test_ds  = make_dataset(test_df,  batch_size=BATCH_SIZE * 4)

#     # Build the model ONCE — reused every round inside BookstoreClient
#     model = build_ncf_model(
#         num_users=meta["num_users"],
#         num_books=meta["num_books"],
#         embed_dim=EMBED_DIM,
#         mlp_layers=MLP_LAYERS,
#         learning_rate=LR,
#         l2_reg=L2_REG,
#     )

#     print(f"\n[{args.node_id}] {len(df):,} ratings "
#           f"({len(train_df):,} train / {len(test_df):,} test).")
#     print(f"[{args.node_id}] Connecting to {args.server_address} ...\n")

#     fl.client.start_numpy_client(
#         server_address=args.server_address,
#         client=BookstoreClient(
#             node_id=args.node_id,
#             model=model,
#             train_ds=train_ds,
#             test_ds=test_ds,
#             n_train=len(train_df),
#             n_test=len(test_df),
#         ),
#     )


# if __name__ == "__main__":
#     main()


import argparse
import tensorflow as tf
import flwr as fl

from data_preparation import load_partition, load_meta
from model import build_model, get_weights, set_weights, make_dataset


# ── Config ─────────────────────────────
LOCAL_EPOCHS = 10
BATCH_SIZE = 256


# ── Client ─────────────────────────────
class Client(fl.client.NumPyClient):

    def __init__(self, model, train_ds, test_ds, n_train, n_test):
        self.model = model
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.n_train = n_train
        self.n_test = n_test

    def get_parameters(self, config):
        return get_weights(self.model)

    def fit(self, parameters, config):
        set_weights(self.model, parameters)

        self.model.fit(self.train_ds, epochs=LOCAL_EPOCHS, verbose=0)

        return get_weights(self.model), self.n_train, {}

    def evaluate(self, parameters, config):
        set_weights(self.model, parameters)

        results = self.model.evaluate(self.test_ds, verbose=1)
        loss, mae = results[0], results[1]
        return loss, self.n_test, {"mae": mae}


# ── Main ───────────────────────────────
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--server-address", default="127.0.0.1:8081")
    args = parser.parse_args()

    tf.random.set_seed(42)

    # Load global metadata
    meta = load_meta()
    num_users = meta["num_users"]
    num_books = meta["num_books"]

    # Load this client's data
    df = load_partition(args.node_id)

    # Simple split
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)

    train_ds = make_dataset(train_df, BATCH_SIZE, shuffle=True)
    test_ds = make_dataset(test_df, BATCH_SIZE)

    # Build model ONCE
    model = build_model(num_users, num_books)

    print(f"[{args.node_id}] Training samples: {len(train_df)}")

    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=Client(
            model,
            train_ds,
            test_ds,
            len(train_df),
            len(test_df),
        ),
    )