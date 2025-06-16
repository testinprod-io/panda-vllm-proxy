from dataclasses import dataclass
import jwt
from fastapi import HTTPException, Header, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from ...config import get_settings
from ...logger import log

limiter = Limiter(key_func=get_remote_address, storage_uri="memory://")

@dataclass
class AuthInfo:
    user_id: str
    is_api_key: bool = False

@limiter.limit("10/minute")
def verify_authorization_header(request: Request, authorization: str = Header(None)) -> AuthInfo:
    """
    Verify the authorization header.
    Handles both API keys (without Bearer prefix) and JWT tokens (with Bearer prefix).
    
    Args:
        request: The LLMRequest object
        authorization: The raw Authorization header value
        
    Returns:
        AuthInfo: Object containing user_id and authentication method info
        
    Raises:
        HTTPException: If the token is invalid or expired
    """
    settings = get_settings()

    if not authorization or not authorization.startswith("Bearer "):
        log.error("No authorization header provided")
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    token = authorization.split("Bearer ")[1]

    # First check if it's a direct API key (no Bearer prefix)
    if token in settings.API_KEYS:
        return AuthInfo(user_id=f"api_key_{token[:8]}", is_api_key=True)
    
    # Verify as JWT
    jwt_pub_key = settings.JWT_PUB_KEY
    jwt_algorithm = settings.JWT_ALGORITHM
    app_id = settings.APP_ID

    try:
        payload = jwt.decode(
            token,
            jwt_pub_key,
            issuer="privy.io",
            audience=app_id,
            algorithms=[jwt_algorithm],
            options={"verify_exp": True}
        )
        return AuthInfo(user_id=payload["sub"])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError as e:
        log.error(f"Invalid token: {str(e)}")
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")
