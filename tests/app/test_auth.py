import pytest
from datetime import datetime, timedelta, timezone
from fastapi import HTTPException
from tests.app.test_helpers import setup_test_auth, create_test_token, create_test_request

setup_test_auth()

from app.api.helper.auth import verify_authorization_header

def test_valid_token():
    """Test verification of a valid JWT token"""
    token = create_test_token()
    auth_header = f"Bearer {token}"
    request = create_test_request()
    payload = verify_authorization_header(request, auth_header)
    assert payload["sub"] == "test_user"

def test_expired_token():
    """Test handling of expired token"""
    token = create_test_token(expires_in=timedelta(seconds=-1))  # Expired token
    auth_header = f"Bearer {token}"
    request = create_test_request()
    with pytest.raises(HTTPException) as exc_info:
        verify_authorization_header(request, auth_header)
    assert exc_info.value.status_code == 401
    assert "Token has expired" in str(exc_info.value.detail)

def test_invalid_token():
    """Test handling of invalid token"""
    auth_header = "Bearer invalid.token.here"
    request = create_test_request()
    with pytest.raises(HTTPException) as exc_info:
        verify_authorization_header(request, auth_header)
    assert exc_info.value.status_code == 401
    assert "Invalid token" in str(exc_info.value.detail)

def test_custom_payload():
    """Test verification with custom payload"""
    custom_payload = {
        "sub": "test_user",
        "role": "admin",
        "iss": "privy.io",
        "exp": int((datetime.now(timezone.utc) + timedelta(days=1)).timestamp())
    }
    token = create_test_token(payload=custom_payload)
    auth_header = f"Bearer {token}"
    request = create_test_request()
    payload = verify_authorization_header(request, auth_header)
    assert payload["sub"] == "test_user"
    assert payload["role"] == "admin" 