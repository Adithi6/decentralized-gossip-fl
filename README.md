#  Secure Decentralized Federated Learning using Gossip Protocol

##  Overview

<<<<<<< HEAD
This project implements a **fully decentralized federated learning (FL) system** using a **gossip-based communication protocol** and **post-quantum cryptographic signatures (Dilithium)** for secure model updates.

Unlike traditional federated learning systems that rely on a **central server**, this implementation eliminates central aggregation and enables **peer-to-peer model exchange and aggregation**.

---

##  Key Features

*  **Decentralized Training**
  No central server is used for aggregation — all nodes collaborate via peer-to-peer gossip.

*  **Gossip Protocol Communication**
  Model updates are propagated using a **push-based randomized gossip protocol**.

*  **Post-Quantum Security (Dilithium)**
  Each client signs its model updates using **Dilithium digital signatures**, ensuring integrity and authenticity.

*  **Local Model Aggregation**
  Each node aggregates received updates independently using decentralized FedAvg-style aggregation.

*  **Efficient Propagation**
  Controlled using:

  * `fanout` (number of peers)
  * `max_hops` (propagation depth)
=======
```
dilithium_fl/
│
├── main.py                      ← Entry point (runs decentralized FL loop)
├── config.yaml                  ← Experiment, gossip, and security settings
│
├── model/
│   └── cnn.py                   ← SmallCNN architecture (PyTorch)
│
├── data/
│   └── loader.py                ← Downloads MNIST, splits per client
│
├── crypto/
│   └── dilithium_utils.py       ← Keygen, sign, verify (Dilithium2)
│
├── gossip/
│   ├── node.py                  ← GossipNode (client + inbox + aggregation)
│   └── protocol.py              ← Peer-to-peer gossip propagation logic
│
├── client/
│   └── fl_client.py             ← Local training + signing
│
├── server/
│   └── fl_server.py             ← Initializes global model + stores public keys
│
└── utils/
    └── weights.py               ← Convert model weights ↔ bytes
```

---

## How it flows

```
main.py
  │
  ├─ config.yaml                 → load experiment + gossip settings
  ├─ data/loader.py              → split MNIST among clients
  ├─ crypto/dilithium_utils.py   → each node generates Dilithium keypair
  │
  └─ for each round:
       ├─ gossip/node.py         → local SGD training on node's data
       ├─ gossip/node.py         → sign(SHA256(weights)) using Dilithium
       ├─ gossip/protocol.py     → propagate updates via gossip
       │     each receiver verifies signature before forwarding
       │     propagation stops at max_hops
       └─ gossip/node.py         → each node aggregates received updates locally
```

---

## Gossip Protocol

### Without Gossip

* Clients send updates directly to a central server (star topology)

### With Gossip (this project)

* Each node forwards its update to `fanout` random peers
* Each receiver:

  * verifies the Dilithium signature
  * forwards only if valid
* Messages propagate up to `max_hops`
* Each node stores verified updates in its inbox
* Each node performs **local aggregation (fully decentralized)**

```
node_0 → node_1, node_2
node_1 → node_3, ...
node_2 → node_3, ...

✔ propagation continues up to max_hops  
✔ duplicate messages ignored  
✔ each node aggregates its own received updates  
```

---

## Configuration (`config.yaml`)

```yaml
experiment:
  n_clients: 4
  n_rounds: 3
  local_epochs: 1
  samples_per_client: 500

gossip:
  fanout: 2
  max_hops: 3

security:
  use_hash: true
```

### Key parameters

* `n_clients` → number of participating nodes
* `n_rounds` → FL rounds
* `local_epochs` → local training per round
* `fanout` → peers each node forwards to
* `max_hops` → gossip depth
* `use_hash` → enable SHA-256 before signing

---

## Install

```bash
pip install torch torchvision dilithium-py numpy pyyaml
```

---

## Hash vs Direct Signing Comparison

### 1. Hash-then-Sign (Recommended)

* Model weights are hashed using SHA-256 (32 bytes)
* The hash is signed using Dilithium
* Verification checks:

  * hash integrity
  * signature validity

### 2. Direct Signing

* Full model update (~800 KB) is directly signed
* Signature verification is performed on raw data

---

## Observations

| Feature            | With Hash                | Without Hash      |
| ------------------ | ------------------------ | ----------------- |
| Input size to sign | 32 bytes                 | ~800 KB           |
| Speed              | Faster                   | Slower            |
| Security           | Integrity + authenticity | Authenticity only |
| Scalability        | High                     | Low               |
>>>>>>> 0d6300d (Updated README and finalized decentralized gossip FL implementation)

---

##  System Architecture

<<<<<<< HEAD
```text
          +-----------+
          | Client 0  |
          +-----------+
                |
                v
      +-------------------+
      | Gossip Protocol   |
      +-------------------+
          /      |      \
         v       v       v
   +---------+ +---------+ +---------+
   | Client1 | | Client2 | | Client3 |
   +---------+ +---------+ +---------+
```

Each client:

* trains locally
* signs updates
* sends updates to random peers
* verifies received updates
* aggregates locally

---

##  Workflow

1. **Initialization**

   * All clients start with a **common initial model**
=======
| Local Epochs | Total Time (s) | Final Accuracy |
| ------------ | -------------- | -------------- |
| 1            | 10.66          | 77.94%         |
| 2            | 12.39          | 89.49%         |
| 3            | 18.12          | 93.16%         |
| 15           | 40.70          | 95.37%         |

### Observation

Increasing local epochs improves model accuracy but also increases execution time.
Beyond a point, accuracy gains become smaller compared to computational cost.

---

## Key Insight

Hashing reduces large model updates into a fixed-size representation, making Dilithium signing efficient while preserving integrity.

This is especially important in Federated Learning, where model updates are large.

---
>>>>>>> 0d6300d (Updated README and finalized decentralized gossip FL implementation)

2. **Local Training**

   * Each client trains on its private dataset

3. **Signing**

   * Model updates are signed using Dilithium

4. **Gossip Propagation**

   * Updates are shared with randomly selected peers

5. **Verification**

   * Each node verifies received updates

6. **Local Aggregation**

   * Each node aggregates valid updates locally

7. **Repeat**

   * The process continues for multiple rounds

---

## ⚙️ Configuration

```python
N_CLIENTS = 4
N_ROUNDS = 3
LOCAL_EPOCHS = 15
SAMPLES_PER_CLIENT = 500
GOSSIP_FANOUT = 2
GOSSIP_MAX_HOPS = 3
```

---

##  How to Run

```bash
# Install dependencies
pip install torch torchvision numpy

# Run the project
py main.py
```

---

<<<<<<< HEAD
##  Sample Output

The program displays:

* local training loss for each client
* Dilithium signing time
* gossip propagation logs
* signature verification time
* round execution time

---

##  Security Aspect

This project uses **Dilithium**, a **post-quantum digital signature scheme**, to secure model updates.

This ensures:

*  authenticity of updates
*  integrity of transmitted model parameters
*  resistance against quantum attacks

---

##  Key Concepts Used

* Federated Learning
* Gossip Protocol
* Decentralized Aggregation
* Peer-to-Peer Communication
* Digital Signatures
* Post-Quantum Cryptography
* FedAvg

---



##  Conclusion

This project demonstrates a **secure, scalable, and fully decentralized federated learning system** using gossip-based communication and post-quantum cryptography.

It removes reliance on a central server while maintaining model integrity and collaborative learning.

---


=======
## Final Note

This implementation is:

* Fully **decentralized (no central aggregation)**
* Uses **gossip-based communication**
* Secured using **post-quantum Dilithium signatures**
* Optimized using **hash-based signing for large model updates**
>>>>>>> 0d6300d (Updated README and finalized decentralized gossip FL implementation)
