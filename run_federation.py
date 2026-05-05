import threading
import subprocess
import sys
import time

from data_preparation import load_meta


def start_server():
    subprocess.run([sys.executable, "server.py"])


def run_client(node_id: str):
    subprocess.run([
        sys.executable, "client.py",
        "--node-id", node_id,
        "--server-address", "127.0.0.1:8081",
    ])


if __name__ == "__main__":

    meta     = load_meta()
    node_ids = meta["node_ids"]

    print(f"Total nodes : {len(node_ids)}")
    print()

    # Step 1 — start server in background
    print("Starting server...")
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()

    # Step 2 — wait for server to be fully ready
    time.sleep(5)
    print("Server ready. Launching ALL clients now...\n")

    # Step 3 — launch ALL clients at the same time
    client_threads = []
    for node_id in node_ids:
        t = threading.Thread(target=run_client, args=(node_id,))
        t.start()
        client_threads.append(t)
        time.sleep(0.3)   # tiny gap so they don't all hit server at same millisecond

    print(f"All {len(node_ids)} clients launched. Waiting for federation...\n")

    # Step 4 — wait for all clients to finish
    for t in client_threads:
        t.join()

    print("All clients finished.")
    server_thread.join(timeout=30)
    print("Federation complete!")