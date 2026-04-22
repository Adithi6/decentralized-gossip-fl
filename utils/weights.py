import logging
import hashlib
import numpy as np
import torch
import torch.nn as nn


def weights_to_bytes(model: nn.Module) -> bytes:
    arrays = [p.data.cpu().numpy().flatten() for p in model.parameters()]
    flat = np.concatenate(arrays).astype(np.float32)
    data = flat.tobytes()

    logging.debug(f"Converted model weights to bytes | size={len(data)} bytes")
    return data


def bytes_to_weight_arrays(data: bytes, template_model: nn.Module) -> list[np.ndarray]:
    flat = np.frombuffer(data, dtype=np.float32).copy()
    shapes = [tuple(p.shape) for p in template_model.parameters()]

    arrays = []
    idx = 0
    for shape in shapes:
        n = int(np.prod(shape))
        arrays.append(flat[idx: idx + n].reshape(shape))
        idx += n

    logging.debug(f"Reconstructed weight arrays | total_elements={len(flat)}")
    return arrays


def apply_weight_arrays(model: nn.Module, arrays: list[np.ndarray]):
    for param, arr in zip(model.parameters(), arrays):
        param.data = torch.from_numpy(arr).to(param.device)


def model_to_weight_arrays(model: nn.Module) -> list[np.ndarray]:
    arrays = [p.data.cpu().numpy().copy() for p in model.parameters()]
    logging.debug("Converted model to weight arrays")
    return arrays


def hash_weights(model: nn.Module) -> bytes:
    return hashlib.sha256(weights_to_bytes(model)).digest()