import os
from cryptography.fernet import Fernet

key = os.getenv("ENCRYPTION_KEY")
fernet = Fernet(key)

def encrypt_token(token: str) -> str:
    return fernet.encrypt(token.encode()).decode()

def decrypt_token(token: str) -> str:
    return fernet.decrypt(token.encode()).decode()