"""HMAC-signed, expiring download URLs for locally stored images."""

import hashlib
import hmac
import uuid


class UrlSigner:
    def __init__(self, secret: str) -> None:
        self._secret = secret.encode()

    def sign(self, job_id: uuid.UUID, index: int, expires: int) -> str:
        message = f"chriek-image-v1\n{job_id}\n{index}\n{expires}".encode()
        return hmac.new(self._secret, message, hashlib.sha256).hexdigest()

    def verify(self, job_id: uuid.UUID, index: int, expires: int, signature: str) -> bool:
        # Bytes, not str: compare_digest raises TypeError on non-ASCII str input.
        return hmac.compare_digest(self.sign(job_id, index, expires).encode(), signature.encode())
