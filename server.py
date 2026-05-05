# """
# server.py - The federated learning server.

# The server holds NO training data. Every round it:
#   1. Broadcasts current global model weights to all connected clients.
#   2. Receives locally-updated weights from each client.
#   3. Averages them (FedAvg).
#   4. Logs aggregated RMSE / MAE and saves the global Keras model.

# Start this first, then start clients in separate terminals:
#     python server.py
#     python client.py --node-id city_london
#     python client.py --node-id city_paris
# """

# import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# import json
# import tensorflow as tf
# import flwr as fl
# from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
# from flwr.server.strategy import FedAvg

# from data_preparation import load_meta
# from model import build_ncf_model, get_weights, set_weights

# # ── Settings — keep in sync with client.py ────────────────────────────────────
# NUM_ROUNDS  = 10
# MIN_CLIENTS = 2
# EMBED_DIM   = 16
# MLP_LAYERS  = [64, 32]
# LR          = 5e-4      # must match client.py
# L2_REG      = 1e-4


# # ── Strategy ──────────────────────────────────────────────────────────────────

# class FedAvgWithLogging(FedAvg):
#     """
#     FedAvg + weighted RMSE/MAE logging + Keras model checkpoint after each round.

#     The global model is built once (self._global_model) and reused for all
#     checkpointing calls. This avoids repeated TF graph tracing and keeps the
#     weight-slot layout stable.
#     """

#     def __init__(self, num_users, num_books, **kwargs):
#         super().__init__(**kwargs)
#         self.num_users = num_users
#         self.num_books = num_books
#         self.history   = []
#         self.best_rmse = float("inf")

#         # Build the server-side model once for checkpointing
#         self._global_model = build_ncf_model(
#             num_users, num_books, EMBED_DIM, MLP_LAYERS, LR, L2_REG
#         )

#     def aggregate_evaluate(self, server_round, results, failures):
#         if not results:
#             return None, {}

#         total    = sum(r.num_examples for _, r in results)
#         agg_rmse = sum(r.metrics["rmse"] * r.num_examples for _, r in results) / total
#         agg_mae  = sum(r.metrics["mae"]  * r.num_examples for _, r in results) / total

#         print(f"\n  [Server] Round {server_round:>2}/{NUM_ROUNDS}  |  "
#               f"RMSE={agg_rmse:.4f}  MAE={agg_mae:.4f}")
#         self.history.append({"round": server_round, "rmse": agg_rmse, "mae": agg_mae})

#         if agg_rmse < self.best_rmse:
#             self.best_rmse = agg_rmse
#             print(f"           ↳ New best RMSE: {self.best_rmse:.4f}")

#         loss_agg, _ = super().aggregate_evaluate(server_round, results, failures)
#         return loss_agg, {"rmse": agg_rmse, "mae": agg_mae}

#     def aggregate_fit(self, server_round, results, failures):
#         aggregated, metrics = super().aggregate_fit(server_round, results, failures)

#         if aggregated is not None:
#             os.makedirs("data", exist_ok=True)
#             # Load into the PERSISTENT model — same object every round
#             set_weights(self._global_model, parameters_to_ndarrays(aggregated))
#             self._global_model.save("data/global_model_latest.keras")
#             print(f"  [Server] Round {server_round} — global model saved.")

#         return aggregated, metrics


# # ── Main ──────────────────────────────────────────────────────────────────────

# def main():
#     tf.random.set_seed(42)
#     meta = load_meta()

#     # Initial weights — fresh model, same architecture as clients
#     seed_model     = build_ncf_model(meta["num_users"], meta["num_books"],
#                                      EMBED_DIM, MLP_LAYERS, LR, L2_REG)
#     initial_params = ndarrays_to_parameters(get_weights(seed_model))
#     del seed_model

#     strategy = FedAvgWithLogging(
#         num_users=meta["num_users"],
#         num_books=meta["num_books"],
#         fraction_fit=1.0,
#         fraction_evaluate=1.0,
#         min_fit_clients=MIN_CLIENTS,
#         min_evaluate_clients=MIN_CLIENTS,
#         min_available_clients=MIN_CLIENTS,
#         initial_parameters=initial_params,
#     )

#     print("=" * 55)
#     print("  Federated Book Recommender — Server")
#     print(f"  Rounds: {NUM_ROUNDS}  |  Min clients: {MIN_CLIENTS}")
#     print("  Waiting for clients on port 8081 ...")
#     print("=" * 55)

#     fl.server.start_server(
#         server_address="0.0.0.0:8081",
#         config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
#         strategy=strategy,
#     )

#     # ── Summary ───────────────────────────────────────────────────────────────
#     print("\n" + "=" * 55)
#     print(f"  Federation complete!  Best RMSE: {strategy.best_rmse:.4f}")
#     print("=" * 55)
#     print(f"\n{'Round':>6} | {'RMSE':>8} | {'MAE':>8}")
#     print("-" * 28)
#     for h in strategy.history:
#         print(f"{h['round']:>6} | {h['rmse']:>8.4f} | {h['mae']:>8.4f}")

#     os.makedirs("data", exist_ok=True)
#     with open("data/federated_history.json", "w") as f:
#         json.dump(strategy.history, f, indent=2)
#     print("\nHistory saved → data/federated_history.json")


# if __name__ == "__main__":
#     main()

import flwr as fl
import tensorflow as tf

from model import build_model, get_weights
from data_preparation import load_meta


# ── Config ─────────────────────────────
NUM_ROUNDS = 5
MIN_CLIENTS = 5


# ── Main ───────────────────────────────
if __name__ == "__main__":

    tf.random.set_seed(42)

    # Load global metadata
    meta = load_meta()
    num_users = meta["num_users"]
    num_books = meta["num_books"]

    # Build initial model
    model = build_model(num_users, num_books)

    # Convert weights → Flower format
    initial_parameters = fl.common.ndarrays_to_parameters(
        get_weights(model)
    )

    # Simple FedAvg
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        min_fit_clients=MIN_CLIENTS,
        min_available_clients=MIN_CLIENTS,
        initial_parameters=initial_parameters,
    )

    print("Starting server on port 8081...")

    fl.server.start_server(
        server_address="0.0.0.0:8081",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )