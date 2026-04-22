import logging
import yaml
import time
import random
import json
import base64
import torch

from data.loader import make_client_loaders
from gossip.node import GossipNode
from gossip.protocol import GossipProtocol
from utils.weights import model_to_weight_arrays


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("experiment.log"),
        logging.StreamHandler()
    ]
)


REGISTRY_FILE = "client_registry.json"


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_public_keys_to_json(nodes, path=REGISTRY_FILE):
    registry = {}
    for node in nodes:
        registry[node.client_id] = base64.b64encode(node.pk).decode("utf-8")

    with open(path, "w") as f:
        json.dump(registry, f, indent=2)

    logging.info(f"Public key registry saved to {path}")


def load_public_keys_from_json(path=REGISTRY_FILE) -> dict[str, bytes]:
    with open(path, "r") as f:
        registry = json.load(f)

    all_pub_keys = {
        client_id: base64.b64decode(pk_b64.encode("utf-8"))
        for client_id, pk_b64 in registry.items()
    }

    logging.info(f"Public key registry loaded from {path}")
    return all_pub_keys


def choose_aggregator_node(nodes: list[GossipNode]) -> GossipNode:
    """
    Choose the aggregator node based on who has the most submissions.
    Tie-break: random choice among top nodes.
    """
    counts = []
    for node in nodes:
        submission_count = len(node.get_all_submissions())
        counts.append((node, submission_count))
        logging.info(
            f"[aggregator-selection] {node.client_id} has {submission_count} submission(s)"
        )

    max_count = max(count for _, count in counts)
    candidates = [node for node, count in counts if count == max_count]
    aggregator = random.choice(candidates)

    logging.info(
        f"Selected aggregator: {aggregator.client_id} "
        f"(received {max_count} submission(s))"
    )
    return aggregator


def main():
    config = load_config()
    logging.info("Configuration loaded successfully from config.yaml")

    N_CLIENTS = config["experiment"]["n_clients"]
    N_ROUNDS = config["experiment"]["n_rounds"]
    LOCAL_EPOCHS = config["experiment"]["local_epochs"]

    GOSSIP_FANOUT = config["gossip"]["fanout"]
    GOSSIP_MAX_HOPS = config["gossip"]["max_hops"]

    USE_HASH = config["security"]["use_hash"]

    BATCH_SIZE = config["data"]["batch_size"]
    ALPHA = config["data"]["alpha"]

    start_time = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logging.info(f"Device: {device}")
    logging.info(
        f"Config: {N_CLIENTS} clients | {N_ROUNDS} rounds | "
        f"{LOCAL_EPOCHS} local epoch(s) | "
        f"batch_size={BATCH_SIZE} | alpha={ALPHA} | "
        f"gossip fanout={GOSSIP_FANOUT} max_hops={GOSSIP_MAX_HOPS} | "
        f"use_hash={USE_HASH}"
    )

    client_loaders, _ = make_client_loaders(
        n_clients=N_CLIENTS,
        batch_size=BATCH_SIZE,
        alpha=ALPHA,
    )

    logging.info("Node creation and key generation started")
    nodes: list[GossipNode] = []
    for i in range(N_CLIENTS):
        node = GossipNode(
            f"client_{i}",
            client_loaders[i],
            device,
            use_hash=USE_HASH
        )
        nodes.append(node)

    # Save and reload public keys using JSON registry
    save_public_keys_to_json(nodes, REGISTRY_FILE)
    all_pub_keys = load_public_keys_from_json(REGISTRY_FILE)

    gossip = GossipProtocol(
        fanout=GOSSIP_FANOUT,
        max_hops=GOSSIP_MAX_HOPS,
        all_pub_keys=all_pub_keys,
    )

    # Random node initializes the starting global model
    initializer_node = random.choice(nodes)
    initial_weights = model_to_weight_arrays(initializer_node.client.model)

    logging.info(f"Random initializer selected: {initializer_node.client_id}")

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

        logging.info("Aggregator selection started")
        aggregator_node = choose_aggregator_node(nodes)

        logging.info("Leader-based aggregation started")
        aggregator_submissions = aggregator_node.get_all_submissions()

        if aggregator_submissions:
            aggregator_node.aggregate_local_updates(
                aggregator_submissions,
                aggregator_node.client.model
            )

            # Share aggregator's updated model to all nodes
            aggregated_weights = model_to_weight_arrays(aggregator_node.client.model)
            for node in nodes:
                node.local_train(aggregated_weights, epochs=0)

            logging.info(
                f"Aggregated model from {aggregator_node.client_id} "
                f"shared with all nodes"
            )
        else:
            logging.warning(
                f"No submissions available at aggregator {aggregator_node.client_id}"
            )

        round_end = time.time()
        logging.info(
            f"Round {round_num} execution time: {round_end - round_start:.2f} seconds"
        )

    end_time = time.time()
    logging.info(f"Total execution time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()