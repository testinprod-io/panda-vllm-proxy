import os
from fastapi import HTTPException, Header, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
import jwt

limiter = Limiter(key_func=get_remote_address, storage_uri="memory://")

JWT_PUB_KEY = os.getenv("JWT_PUB_KEY")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM")
APP_ID = os.getenv("APP_ID")

def verify_authorization_header(request: Request, authorization: str = Header(None)) -> str:
    """
    Verify the JWT token in the Authorization header.
    If the token is valid, return the decoded token payload.
    If there's no token, apply rate limiting with slowapi.
    
    Args:
        request: The FastAPI request object
        authorization: The raw Authorization header value
        
    Returns:
        str: The decoded token payload if verification is successful
        
    Raises:
        HTTPException: If the token is invalid or expired
    """
    if not authorization or not authorization.startswith("Bearer "):
        rate_limit(request)
        return None
  
    token = authorization.split("Bearer ")[1]
    try:
        payload = jwt.decode(
            token,
            JWT_PUB_KEY,
            issuer="privy.io",
            audience=APP_ID,
            algorithms=[JWT_ALGORITHM],
            options={"verify_exp": True}
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")

@limiter.limit("3/minute")
def rate_limit(request: Request):
    """
    Simple rate limiting function to restrict the number of requests per minute 
    for non-authenticated users. Dependency will act as a rate limiter.
    """
    return True