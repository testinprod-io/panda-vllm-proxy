import os
import time
import secrets

from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import ec

from fastapi import Request

from .config import get_settings
from .logger import log


CHALLENGE_HEADER = "Panda-Challenge"
PUBLIC_KEY_HEADER = "Panda-Public-Key"
SIGNATURE_HEADER = "Panda-Signature"
SERVER_RANDOM_HEADER = "Panda-Server-Random"
TS_HEADER = "Panda-Timestamp"
PROOF_PREFIX = "PANDA_PROOF_V0"
AUTH_HEADER = "Authorization"
MAX_RETRIES = 20

private_key = None
hex_public_key = ""

settings = get_settings()
if settings.TLS_CERT_PRIVATE_KEY_PATH is None or settings.TLS_CERT_PATH is None:
    log.warn("No cert path provided. will skip cert signing")
else:
    for attempt in range(MAX_RETRIES):
        if os.path.exists(settings.TLS_CERT_PRIVATE_KEY_PATH) and os.path.exists(settings.TLS_CERT_PATH):
            break
        log.info(f"Cert file not found. Retrying ({attempt + 1}/{MAX_RETRIES})...")
        time.sleep(3)
    else:
        raise Exception("Unable to find cert file")

    with open(settings.TLS_CERT_PRIVATE_KEY_PATH, "rb") as f:
        priv_pem = f.read()
    private_key = serialization.load_pem_private_key(priv_pem, password=None)
    assert isinstance(private_key, ec.EllipticCurvePrivateKey)

    with open(settings.TLS_CERT_PATH, "rb") as f:
        cert_pem = f.read()
    cert = x509.load_pem_x509_certificate(cert_pem, default_backend())
    public_key = cert.public_key()
    assert isinstance(public_key, ec.EllipticCurvePublicKey)
    hex_public_key = public_key.public_bytes(
        encoding=serialization.Encoding.X962,
        format=serialization.PublicFormat.UncompressedPoint
    ).hex()
    log.info(f"Loaded TLS cert. public key: {hex_public_key}")

async def prove_server_identity(request: Request, call_next):
    resp = await call_next(request)
    if private_key is None:
        log.warn("No private key provided. skipping cert signing")
        return resp

    resp.headers[PUBLIC_KEY_HEADER] = hex_public_key

    challenge = request.headers.get(CHALLENGE_HEADER)
    if challenge is None or challenge == "":
        challenge = request.headers.get(AUTH_HEADER)
        if challenge is None or challenge == "":
            log.info("No challenge of auth header. skipping cert signing")
            return resp

    server_random = secrets.token_hex(32)
    ts = str(int(time.time()))
    proof = f"{PROOF_PREFIX}|{ts}|{server_random}|{challenge}"

    resp.headers[SERVER_RANDOM_HEADER] = server_random
    resp.headers[TS_HEADER] = ts
    resp.headers[SIGNATURE_HEADER] = private_key.sign(proof.encode("utf-8"), ec.ECDSA(hashes.SHA256())).hex()
    return resp