import logging
import hashlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model.cnn import SmallCNN
from crypto import dilithium_utils
from utils.weights import apply_weight_arrays, weights_to_bytes


class FederatedClient:
    def __init__(self, client_id: str, dataloader: DataLoader, device: str, use_hash: bool = False):
        self.client_id = client_id
        self.dataloader = dataloader
        self.device = device
        self.use_hash = use_hash
        self.model = SmallCNN().to(device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        self.pk, self.sk, keygen_ms = dilithium_utils.keygen()
        logging.info(
            f"[{client_id}] keygen: {keygen_ms:.2f} ms "
            f"(pk={len(self.pk)}B sk={len(self.sk)}B)"
        )

    def local_train(self, global_weight_arrays=None, epochs=1):
        if global_weight_arrays is not None:
            apply_weight_arrays(self.model, global_weight_arrays)

        if epochs == 0:
            return

        self.model.train()
        total_loss = 0.0

        for _ in range(epochs):
            for batch_idx, (x, y) in enumerate(self.dataloader):
                x, y = x.to(self.device), y.to(self.device)

                self.optimizer.zero_grad()

                logits = self.model(x)
                loss = self.criterion(logits, y)

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

                # Debug logits only for first batch to avoid too much logging
                if batch_idx == 0:
                    pred = torch.argmax(logits, dim=1)
                    logging.info(
                        f"[{self.client_id}] logits sample: {logits[0].detach().cpu().numpy()} | "
                        f"pred={pred[0].item()} | actual={y[0].item()}"
                    )

        total_batches = len(self.dataloader) * epochs
        logging.info(f"[{self.client_id}] trained | loss: {total_loss / total_batches:.4f}")

    def sign_update(self) -> dict:
        update_bytes = weights_to_bytes(self.model)

        if self.use_hash:
            payload = hashlib.sha256(update_bytes).digest()
            mode = "HASHED"
        else:
            payload = update_bytes
            mode = "RAW"

        signature, sign_ms = dilithium_utils.sign(self.sk, payload)

        logging.info(
            f"[{self.client_id}] signed ({mode}) | {sign_ms:.3f} ms "
            f"input={len(payload)} B update={len(update_bytes)/1024:.1f} KB "
            f"sig={len(signature)} B"
        )

        return {
            "client_id": self.client_id,
            "update_bytes": update_bytes,
            "payload": payload,
            "signature": signature,
            "sign_ms": sign_ms,
        }