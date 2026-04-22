import logging
import yaml
import time
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


def print_timing_table(all_timings: list[dict]):
    if not all_timings:
        logging.info("No server timings available (fully decentralized mode).")
        return

    logging.info("=" * 64)
    logging.info("Crypto Timing Summary (sign + server-verify)")
    logging.info("=" * 64)
    logging.info(f"{'Round':<6} {'Client':<12} {'Sign (ms)':<14} {'Verify (ms)':<14} Valid")
    logging.info("-" * 58)

    for t in all_timings:
        logging.info(
            f"{t['round']:<6} {t['client_id']:<12} "
            f"{t['sign_ms']:<14} {t['verify_ms']:<14} {t['valid']}"
        )

    sign_ms = [t["sign_ms"] for t in all_timings]
    verify_ms = [t["verify_ms"] for t in all_timings]

    logging.info(
        f"Avg sign   : {np.mean(sign_ms):.3f} ms "
        f"(min {np.min(sign_ms):.3f} max {np.max(sign_ms):.3f})"
    )
    logging.info(
        f"Avg verify : {np.mean(verify_ms):.3f} ms "
        f"(min {np.min(verify_ms):.3f} max {np.max(verify_ms):.3f})"
    )
    logging.info("=" * 64)


def main():
    config = load_config()
    logging.info("Configuration loaded successfully from config.yaml")

    N_CLIENTS = config["experiment"]["n_clients"]
    N_ROUNDS = config["experiment"]["n_rounds"]
    LOCAL_EPOCHS = config["experiment"]["local_epochs"]
   

    GOSSIP_FANOUT = config["gossip"]["fanout"]
    GOSSIP_MAX_HOPS = config["gossip"]["max_hops"]

    USE_HASH = config["security"]["use_hash"]

    start_time = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logging.info(f"Device: {device}")
    logging.info(
        f"Config: {N_CLIENTS} clients | {N_ROUNDS} rounds | "
        f"{LOCAL_EPOCHS} local epoch(s) | "
        f"gossip fanout={GOSSIP_FANOUT} max_hops={GOSSIP_MAX_HOPS} | "
        f"use_hash={USE_HASH}"
    )

    BATCH_SIZE = config["data"]["batch_size"]
    ALPHA = config["data"]["alpha"]

    client_loaders, _ = make_client_loaders(
        n_clients=N_CLIENTS,
        batch_size=BATCH_SIZE,
        alpha=ALPHA,
    )
    server = FederatedServer(device)

    logging.info("Key generation started")
    nodes: list[GossipNode] = []
    for i in range(N_CLIENTS):
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

    for round_num in range(1, N_ROUNDS + 1):
        round_start = time.time()
        logging.info(f"Round {round_num}/{N_ROUNDS} started")

        logging.info("Training phase started")
        for node in nodes:
            node.local_train(None, epochs=LOCAL_EPOCHS)

        logging.info("Signing phase started")
        for node in nodes:
            node.sign_update()

        logging.info("Gossip propagation started")
        gossip.run_round(nodes)
        gossip.print_gossip_summary()

        logging.info("Decentralized aggregation started")
        for node in nodes:
            local_submissions = node.get_all_submissions()
            if local_submissions:
                node.aggregate_local_updates(local_submissions, node.client.model)

        round_end = time.time()
        logging.info(f"Round {round_num} execution time: {round_end - round_start:.2f} seconds")

    end_time = time.time()
    logging.info(f"Total execution time: {end_time - start_time:.2f} seconds")

    print_timing_table(server.all_timings)


if __name__ == "__main__":
    main()