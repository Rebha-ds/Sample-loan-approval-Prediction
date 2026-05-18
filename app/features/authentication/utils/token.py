from datetime import datetime, timedelta
from joserfc import jwt
from joserfc.jwk import OctKey
from app.config.settings import SECRET_KEY

key = OctKey.import_key(SECRET_KEY)


def create_access_token(data: dict, expires_minutes: int = 60):
    payload = data.copy()
    payload["iat"] = int(datetime.utcnow().timestamp())
    payload["exp"] = int(
        (datetime.utcnow() + timedelta(minutes=expires_minutes)).timestamp()
    )

    token = jwt.encode(
        {"alg": "HS256"},
        payload,
        key
    )

    return token.decode() if isinstance(token, bytes) else token

def verify_access_token(token: str):
    try:
        claims = jwt.decode(token, key)
        claims.validate()
        return claims
    except Exception:
        return None