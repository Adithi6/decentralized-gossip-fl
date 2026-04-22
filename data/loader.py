import logging
from typing import Tuple, List

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner


def _partition_to_tensordataset(partition) -> TensorDataset:
    """
    Convert one Flower/Hugging Face partition into a PyTorch TensorDataset.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    images = []
    labels = []

    for item in partition:
        # item["image"] is a PIL image for MNIST in Flower Datasets
        img = transform(item["image"])
        label = int(item["label"])

        images.append(img)
        labels.append(label)

    x_tensor = torch.stack(images)
    y_tensor = torch.tensor(labels, dtype=torch.long)

    return TensorDataset(x_tensor, y_tensor)


def make_client_loaders(
    n_clients: int,
    batch_size: int = 32,
    alpha: float = 0.5,
) -> Tuple[List[DataLoader], DataLoader]:
    """
    Create non-IID client DataLoaders using Flower DirichletPartitioner.

    Args:
        n_clients: number of federated clients
        batch_size: DataLoader batch size
        alpha: Dirichlet concentration parameter
               smaller alpha -> more non-IID
               larger alpha -> more IID-like

    Returns:
        client_loaders: list of client train DataLoaders
        test_loader: DataLoader for full MNIST test set
    """
    partitioner = DirichletPartitioner(
        num_partitions=n_clients,
        partition_by="label",
        alpha=alpha,
        min_partition_size=10,
        self_balancing=True,
        seed=42,
    )

    fds = FederatedDataset(
        dataset="ylecun/mnist",
        partitioners={"train": partitioner},
    )

    client_loaders: List[DataLoader] = []

    logging.info(
        f"Creating {n_clients} Dirichlet-based client partitions | alpha={alpha}"
    )

    for client_id in range(n_clients):
        partition = fds.load_partition(client_id, "train")
        dataset = _partition_to_tensordataset(partition)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        client_loaders.append(loader)

        labels = [int(item["label"]) for item in partition]
        unique_labels = sorted(set(labels))
        logging.info(
            f"Client {client_id} | samples={len(labels)} | labels={unique_labels}"
        )

    test_partition = fds.load_split("test")
    test_dataset = _partition_to_tensordataset(test_partition)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    logging.info(
        f"Created {n_clients} non-IID client loaders using DirichletPartitioner"
    )

    return client_loaders, test_loader