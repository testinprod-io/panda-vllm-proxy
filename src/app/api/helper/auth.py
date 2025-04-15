import os
from fastapi import Header, HTTPException
import jwt

JWT_PUB_KEY = os.getenv("JWT_PUB_KEY")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM")
APP_ID = os.getenv("APP_ID")

def verify_authorization_header(authorization: str = Header("Authorization")) -> str:
    """
    Verify the JWT token in the Authorization header.
    
    Args:
        authorization: The Authorization header value
        
    Returns:
        str: The decoded token payload if verification is successful
        
    Raises:
        HTTPException: If the token is invalid or expired
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401, detail="Invalid or missing Authorization header"
        )
  
    token = authorization.split("Bearer ")[1]
    try:
        # Verify and decode the JWT token
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