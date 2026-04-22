import hashlib
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader

from model.cnn import SmallCNN
from crypto import dilithium_utils
from utils.weights import bytes_to_weight_arrays, apply_weight_arrays


def build_model(
    model_name: str,
    device: str,
    input_channels: int,
    num_classes: int,
    input_height: int,
    input_width: int,
    conv1_channels: int,
    conv2_channels: int,
    hidden_dim: int,
):
    model_name = model_name.lower()

    if model_name == "smallcnn":
        return SmallCNN(
            input_channels=input_channels,
            num_classes=num_classes,
            input_height=input_height,
            input_width=input_width,
            conv1_channels=conv1_channels,
            conv2_channels=conv2_channels,
            hidden_dim=hidden_dim,
        ).to(device)

    raise ValueError(f"Unsupported model: {model_name}")


class FederatedServer:
    def __init__(
        self,
        device: str,
        crypto_scheme: str,
        model_name: str,
        input_channels: int,
        num_classes: int,
        input_height: int,
        input_width: int,
        conv1_channels: int,
        conv2_channels: int,
        hidden_dim: int,
    ):
        self.device = device
        self.crypto_scheme = crypto_scheme
        self.model_name = model_name

        self.global_model = build_model(
            model_name=model_name,
            device=device,
            input_channels=input_channels,
            num_classes=num_classes,
            input_height=input_height,
            input_width=input_width,
            conv1_channels=conv1_channels,
            conv2_channels=conv2_channels,
            hidden_dim=hidden_dim,
        )

        self.client_keys: dict[str, bytes] = {}
        self.all_timings: list[dict] = []

        logging.info(
            f"[server] initialized | model={self.model_name} scheme={self.crypto_scheme}"
        )

    def register_client(self, client_id: str, public_key: bytes):
        self.client_keys[client_id] = public_key
        logging.info(f"[server] registered {client_id} | pk={len(public_key)}B")

    def global_weight_arrays(self) -> list:
        return [p.data.cpu().numpy().copy() for p in self.global_model.parameters()]

    def _compute_expected_payload(self, submission: dict) -> bytes:
        if "is_hashed" not in submission:
            raise KeyError("Submission missing required field: 'is_hashed'")

        if not submission["is_hashed"]:
            return submission["update_bytes"]

        if "hash_algorithm" not in submission:
            raise KeyError("Submission missing required field: 'hash_algorithm'")

        hash_algorithm = submission["hash_algorithm"].lower()

        if hash_algorithm == "sha256":
            return hashlib.sha256(submission["update_bytes"]).digest()

        raise ValueError(f"Unsupported hash algorithm: {hash_algorithm}")

    def verify_and_aggregate(self, submissions: list[dict], round_num: int) -> list[dict]:
        verified_weight_arrays = []
        round_timings = []

        for sub in submissions:
            cid = sub["client_id"]

            if cid not in self.client_keys:
                logging.warning(f"[server] missing public key for {cid}")
                is_valid = False
                verify_ms = 0.0
                status = "REJECTED (NO PUBLIC KEY)"
            else:
                pk = self.client_keys[cid]
                payload = sub["payload"]
                signature = sub["signature"]
                update_bytes = sub["update_bytes"]

                expected_payload = self._compute_expected_payload(sub)

                if expected_payload != payload:
                    is_valid = False
                    verify_ms = 0.0
                    status = "REJECTED (PAYLOAD MISMATCH)"
                else:
                    is_valid, verify_ms = dilithium_utils.verify(
                        pk,
                        payload,
                        signature,
                        self.crypto_scheme,
                    )
                    status = "VALID" if is_valid else "REJECTED (BAD SIGNATURE)"

                if is_valid:
                    arrays = bytes_to_weight_arrays(update_bytes, self.global_model)
                    verified_weight_arrays.append(arrays)

            logging.info(f"[server] verify {cid} | {verify_ms:.3f} ms | [{status}]")

            round_timings.append({
                "round": round_num,
                "client_id": cid,
                "sign_ms": round(float(sub["sign_ms"]), 3) if "sign_ms" in sub else 0.0,
                "verify_ms": round(float(verify_ms), 3),
                "valid": is_valid,
            })

        if verified_weight_arrays:
            averaged = [
                np.mean([w[i] for w in verified_weight_arrays], axis=0)
                for i in range(len(verified_weight_arrays[0]))
            ]
            apply_weight_arrays(self.global_model, averaged)
            logging.info(
                f"[server] FedAvg over {len(verified_weight_arrays)} verified update(s)"
            )
        else:
            logging.warning("[server] no valid updates, global model unchanged")

        self.all_timings.extend(round_timings)
        return round_timings

    def evaluate(self, test_loader: DataLoader) -> float:
        self.global_model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                preds = self.global_model(x).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += len(y)

        accuracy = 100.0 * correct / total if total > 0 else 0.0
        logging.info(f"[server] evaluation accuracy = {accuracy:.2f}%")
        return accuracy