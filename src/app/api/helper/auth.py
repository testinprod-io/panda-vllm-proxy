import jwt
from fastapi import HTTPException, Header, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from ...config import get_settings
from ...logger import log

limiter = Limiter(key_func=get_remote_address, storage_uri="memory://")

@limiter.limit("10/minute")
def verify_authorization_header(request: Request, authorization: str = Header(None)) -> str:
    """
    Verify the JWT token in the Authorization header.
    If the token is valid, return the user id.
    
    Args:
        request: The LLMRequest object
        authorization: The raw Authorization header value
        
    Returns:
        str: The decoded token payload if verification is successful
        
    Raises:
        HTTPException: If the token is invalid or expired
    """
    settings = get_settings()

    jwt_pub_key = settings.JWT_PUB_KEY
    jwt_algorithm = settings.JWT_ALGORITHM
    app_id = settings.APP_ID

    if not authorization or not authorization.startswith("Bearer "):
        log.error("No authorization header provided")
        raise HTTPException(status_code=401, detail="Unauthorized")
  
    token = authorization.split("Bearer ")[1]
    try:
        payload = jwt.decode(
            token,
            jwt_pub_key,
            issuer="privy.io",
            audience=app_id,
            algorithms=[jwt_algorithm],
            options={"verify_exp": True}
        )
        return payload["sub"]
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError as e:
        log.error(f"Invalid token: {str(e)}")
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")
