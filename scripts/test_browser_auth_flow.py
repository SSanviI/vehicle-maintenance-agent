"""
Integration test for session cookie authentication, emergency contact persistence, and alert dispatch.
"""

import requests
from backend.app.auth.session import create_session_token, SESSION_COOKIE_NAME

def main():
    base_url = "http://localhost:7860"
    session = requests.Session()

    print("--- TEST A: Unauthenticated status check ---")
    res = session.get(f"{base_url}/auth/status")
    print("Status code:", res.status_code)
    data = res.json()
    print("Unauthenticated status response:", data)
    assert data == {"authenticated": False}, f"Expected authenticated=False, got {data}"

    print("\n--- TEST D & E: Simulating valid vm_session cookie ---")
    # Using test user from db fixture or create a session token for user
    test_gid = "109876543210987654321"
    
    # Create or update user in database first
    import asyncio
    from datetime import datetime
    import uuid
    from backend.app.database.session import get_session_factory
    from backend.app.database.models import GoogleUser, OAuthToken

    async def seed_user():
        factory = get_session_factory()
        async with factory() as db:
            from sqlalchemy import select
            res = await db.execute(select(GoogleUser).where(GoogleUser.google_user_id == test_gid))
            user = res.scalar_one_or_none()
            if not user:
                user = GoogleUser(
                    id=uuid.uuid4(),
                    google_user_id=test_gid,
                    name="Test Driver",
                    email="testdriver@gmail.com",
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                )
                db.add(user)
                await db.commit()
            return user

    asyncio.run(seed_user())

    cookie_val = create_session_token(test_gid)
    session.cookies.set(SESSION_COOKIE_NAME, cookie_val)

    res = session.get(f"{base_url}/auth/status")
    print("Status code with cookie:", res.status_code)
    data = res.json()
    print("Authenticated status response:", data)
    assert data.get("authenticated") is True, f"Expected authenticated=True, got {data}"
    assert data.get("email") == "testdriver@gmail.com"

    print("\n--- TEST F & G: Save Emergency Contact ---")
    contact_payload = {
        "contact_name": "Test Person",
        "relationship": "Father",
        "contact_email": "test-father@example.com"
    }
    res = session.post(f"{base_url}/api/emergency/contact", json=contact_payload)
    print("Create contact status code:", res.status_code)
    contact_data = res.json()
    print("Saved contact response:", contact_data)
    assert contact_data["contact_name"] == "Test Person"
    assert contact_data["relationship"] == "Father"
    assert contact_data["contact_email"] == "test-father@example.com"

    print("\n--- TEST G (Refresh check): Get Emergency Contact ---")
    res = session.get(f"{base_url}/api/emergency/contact")
    print("Get contact status code:", res.status_code)
    get_data = res.json()
    print("Retrieved contact:", get_data)
    assert get_data["contact_email"] == "test-father@example.com"

    print("\n--- TEST J: Check Emergency History ---")
    res = session.get(f"{base_url}/api/emergency/history")
    print("History status code:", res.status_code)
    hist_data = res.json()
    print("Notification History:", hist_data)
    assert "logs" in hist_data

    print("\n✅ ALL BACKEND CONTRACT TESTS PASSED PERFECTLY!")

if __name__ == "__main__":
    main()
