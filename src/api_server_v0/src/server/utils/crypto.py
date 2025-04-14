from Crypto.Cipher import AES
from fastapi import HTTPException
from config.logging_config import logger

def decrypt_data(encrypted_data: bytes, key: bytes) -> bytes:
    try:
        nonce, ciphertext = encrypted_data[:12], encrypted_data[12:]
        cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
        plaintext = cipher.decrypt_and_verify(ciphertext[:-16], ciphertext[-16:])
        return plaintext
    except Exception as e:
        logger.error(f"Decryption failed: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid encrypted data")