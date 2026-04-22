import logging
import time
from dilithium_py.dilithium import Dilithium2


def keygen() -> tuple[bytes, bytes, float]:
    t0 = time.perf_counter()
    pk, sk = Dilithium2.keygen()
    elapsed_ms = (time.perf_counter() - t0) * 1000

    logging.debug(f"Dilithium keygen completed in {elapsed_ms:.2f} ms")

    return pk, sk, elapsed_ms


def sign(sk: bytes, message: bytes) -> tuple[bytes, float]:
    t0 = time.perf_counter()
    signature = Dilithium2.sign(sk, message)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    logging.debug(
        f"Dilithium signing completed in {elapsed_ms:.2f} ms (input={len(message)} bytes)"
    )

    return signature, elapsed_ms


def verify(pk: bytes, message: bytes, signature: bytes) -> tuple[bool, float]:
    t0 = time.perf_counter()
    is_valid = Dilithium2.verify(pk, message, signature)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    logging.debug(
        f"Dilithium verify completed in {elapsed_ms:.2f} ms | valid={is_valid}"
    )

    return is_valid, elapsed_ms