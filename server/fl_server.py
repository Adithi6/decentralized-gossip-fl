import hashlib
import numpy as np
import torch
from torch.utils.data import DataLoader

from model.cnn import SmallCNN
from crypto import dilithium_utils
from utils.weights import bytes_to_weight_arrays, apply_weight_arrays


class FederatedServer:
    def __init__(self, device: str):
        self.device = device
        self.global_model = SmallCNN().to(device)
        self.client_keys: dict[str, bytes] = {}
        self.all_timings: list[dict] = []

    def register_client(self, client_id: str, public_key: bytes):
        self.client_keys[client_id] = public_key
        print(f"  [server] registered {client_id}  pk={len(public_key)}B")

    def global_weight_arrays(self) -> list:
        return [p.data.cpu().numpy() for p in self.global_model.parameters()]

    def verify_and_aggregate(self, submissions: list[dict], round_num: int) -> list[dict]:
        verified_weight_arrays = []
        round_timings = []

        for sub in submissions:
            cid = sub["client_id"]
            pk = self.client_keys[cid]
            payload = sub["payload"]
            signature = sub["signature"]
            update_bytes = sub["update_bytes"]

            # Decide expected payload based on mode
            if len(payload) == 32:
                expected_payload = hashlib.sha256(update_bytes).digest()
            else:
                expected_payload = update_bytes

            # Step 1: integrity check
            if expected_payload != payload:
                is_valid = False
                verify_ms = 0.0
                status = "REJECTED (PAYLOAD MISMATCH)"
            else:
                # Step 2: Dilithium verification
                is_valid, verify_ms = dilithium_utils.verify(pk, payload, signature)
                status = "VALID" if is_valid else "REJECTED (BAD SIGNATURE)"

            print(f"  [server] verify {cid} : {verify_ms:.3f} ms  [{status}]")

            round_timings.append({
                "round": round_num,
                "client_id": cid,
                "sign_ms": round(sub["sign_ms"], 3),
                "verify_ms": round(verify_ms, 3),
                "valid": is_valid,
            })

            if is_valid:
                arrays = bytes_to_weight_arrays(update_bytes, self.global_model)
                verified_weight_arrays.append(arrays)

        if verified_weight_arrays:
            averaged = [
                np.mean([w[i] for w in verified_weight_arrays], axis=0)
                for i in range(len(verified_weight_arrays[0]))
            ]
            apply_weight_arrays(self.global_model, averaged)
            print(f"  [server] FedAvg over {len(verified_weight_arrays)} verified update(s)")
        else:
            print("  [server] WARNING — no valid updates, global model unchanged")

        self.all_timings.extend(round_timings)
        return round_timings

    def evaluate(self, test_loader: DataLoader) -> float:
        self.global_model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                preds = self.global_model(x).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += len(y)
        return 100.0 * correct / total