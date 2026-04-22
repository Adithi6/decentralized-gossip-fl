import logging
import yaml
import time
import csv
import numpy as np
import torch

from data.loader import make_client_loaders
from gossip.node import GossipNode
from gossip.protocol import GossipProtocol
from server.fl_server import FederatedServer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("experiment.log"),
        logging.StreamHandler()
    ]
)


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def extract_sign_ms_from_return(ret):
    if ret is None:
        return None

    if isinstance(ret, (int, float)):
        return float(ret)

    if isinstance(ret, dict) and "sign_ms" in ret:
        return float(ret["sign_ms"])

    if isinstance(ret, (tuple, list)):
        for item in reversed(ret):
            if isinstance(item, (int, float)):
                return float(item)

    return None


def extract_sign_ms_from_node(node):
    attrs = ["sign_ms", "last_sign_ms", "latest_sign_ms", "last_sign_time_ms"]

    for attr in attrs:
        if hasattr(node, attr):
            value = getattr(node, attr)
            if isinstance(value, (int, float)):
                return float(value)

    if hasattr(node, "client"):
        for attr in attrs:
            if hasattr(node.client, attr):
                value = getattr(node.client, attr)
                if isinstance(value, (int, float)):
                    return float(value)

    return None


def print_experiment_summary(n_clients, sign_ms_list, verify_ms_list, total_time):
    logging.info("=" * 72)
    logging.info(f"Experiment summary for {n_clients} clients")
    logging.info("=" * 72)

    if sign_ms_list:
        logging.info(
            f"Avg sign   : {np.mean(sign_ms_list):.3f} ms "
            f"(min {np.min(sign_ms_list):.3f} max {np.max(sign_ms_list):.3f})"
        )
    else:
        logging.warning("No signing timings collected.")

    if verify_ms_list:
        logging.info(
            f"Avg verify : {np.mean(verify_ms_list):.3f} ms "
            f"(min {np.min(verify_ms_list):.3f} max {np.max(verify_ms_list):.3f})"
        )
    else:
        logging.warning("No verification timings collected.")

    logging.info(f"Total time : {total_time:.2f} s")
    logging.info("=" * 72)


def run_experiment(n_clients: int, config: dict) -> dict:
    N_ROUNDS = config["experiment"]["n_rounds"]
    LOCAL_EPOCHS = config["experiment"]["local_epochs"]

    GOSSIP_FANOUT = config["gossip"]["fanout"]
    GOSSIP_MAX_HOPS = config["gossip"]["max_hops"]

    USE_HASH = config["security"]["use_hash"]

    BATCH_SIZE = config["data"]["batch_size"]
    ALPHA = config["data"]["alpha"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    start_time = time.time()

    logging.info("=" * 80)
    logging.info(
        f"Starting experiment | clients={n_clients} | rounds={N_ROUNDS} | "
        f"local_epochs={LOCAL_EPOCHS} | batch_size={BATCH_SIZE} | alpha={ALPHA} | "
        f"fanout={GOSSIP_FANOUT} | max_hops={GOSSIP_MAX_HOPS} | use_hash={USE_HASH}"
    )
    logging.info("=" * 80)

    client_loaders, _ = make_client_loaders(
        n_clients=n_clients,
        batch_size=BATCH_SIZE,
        alpha=ALPHA,
    )

    server = FederatedServer(device)

    logging.info("Key generation started")
    nodes: list[GossipNode] = []
    for i in range(n_clients):
        node = GossipNode(
            f"client_{i}",
            client_loaders[i],
            device,
            use_hash=USE_HASH
        )
        server.register_client(node.client_id, node.pk)
        nodes.append(node)

    all_pub_keys = {n.client_id: n.pk for n in nodes}
    gossip = GossipProtocol(
        fanout=GOSSIP_FANOUT,
        max_hops=GOSSIP_MAX_HOPS,
        all_pub_keys=all_pub_keys,
    )

    initial_weights = server.global_weight_arrays()
    for node in nodes:
        node.local_train(initial_weights, epochs=0)

    all_sign_ms = []
    all_verify_ms = []

    for round_num in range(1, N_ROUNDS + 1):
        round_start = time.time()
        logging.info(f"Round {round_num}/{N_ROUNDS} started for {n_clients} clients")

        logging.info("Training phase started")
        for node in nodes:
            node.local_train(None, epochs=LOCAL_EPOCHS)

        logging.info("Signing phase started")
        round_sign_ms = []
        for node in nodes:
            ret = node.sign_update()

            sign_ms = extract_sign_ms_from_return(ret)
            if sign_ms is None:
                sign_ms = extract_sign_ms_from_node(node)

            if sign_ms is not None:
                round_sign_ms.append(sign_ms)

        all_sign_ms.extend(round_sign_ms)

        logging.info("Gossip propagation started")
        gossip.run_round(nodes)
        gossip.print_gossip_summary()

        round_verify_ms = [
            float(t["verify_ms"])
            for t in gossip.gossip_timings
            if isinstance(t, dict) and "verify_ms" in t
        ]
        all_verify_ms.extend(round_verify_ms)

        logging.info("Decentralized aggregation started")
        for node in nodes:
            local_submissions = node.get_all_submissions()
            if local_submissions:
                node.aggregate_local_updates(local_submissions, node.client.model)

        round_end = time.time()
        logging.info(
            f"Round {round_num} execution time for {n_clients} clients: "
            f"{round_end - round_start:.2f} seconds"
        )

        if round_sign_ms:
            logging.info(
                f"Round {round_num} avg sign time: {np.mean(round_sign_ms):.3f} ms"
            )
        else:
            logging.warning(
                f"Round {round_num}: no sign timings captured. "
                f"Store sign_ms inside sign_update() if needed."
            )

        if round_verify_ms:
            logging.info(
                f"Round {round_num} avg verify time: {np.mean(round_verify_ms):.3f} ms"
            )
        else:
            logging.warning(f"Round {round_num}: no verify timings captured.")

    total_time = time.time() - start_time

    avg_sign = float(np.mean(all_sign_ms)) if all_sign_ms else 0.0
    avg_verify = float(np.mean(all_verify_ms)) if all_verify_ms else 0.0

    print_experiment_summary(n_clients, all_sign_ms, all_verify_ms, total_time)

    result = {
        "n_clients": n_clients,
        "avg_sign_ms": avg_sign,
        "avg_verify_ms": avg_verify,
        "total_time_s": total_time,
    }

    logging.info(
        f"Completed experiment | clients={n_clients} | "
        f"avg_sign={avg_sign:.3f} ms | avg_verify={avg_verify:.3f} ms | "
        f"total_time={total_time:.2f} s"
    )

    return result


def main():
    config = load_config()
    logging.info("Configuration loaded successfully from config.yaml")

    client_counts = list(range(5, 101, 5))
    results = []

    for n_clients in client_counts:
        result = run_experiment(n_clients, config)
        results.append(result)

    logging.info("=" * 80)
    logging.info("FINAL SCALING RESULTS")
    logging.info("=" * 80)
    logging.info(
        f"{'Clients':<10} {'Avg Sign (ms)':<18} "
        f"{'Avg Verify (ms)':<18} {'Total Time (s)':<18}"
    )
    logging.info("-" * 70)

    for r in results:
        logging.info(
            f"{r['n_clients']:<10} "
            f"{r['avg_sign_ms']:<18.3f} "
            f"{r['avg_verify_ms']:<18.3f} "
            f"{r['total_time_s']:<18.2f}"
        )

    with open("scaling_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["n_clients", "avg_sign_ms", "avg_verify_ms", "total_time_s"]
        )
        writer.writeheader()
        writer.writerows(results)

    logging.info("Saved results to scaling_results.csv")


if __name__ == "__main__":
    main()