import asyncio
import httpx

async def verify():
    base_url = "http://localhost:7860"
    print("=" * 80)
    print(f"VERIFYING OAUTH AND BACKEND STARTUP ON {base_url}")
    print("=" * 80)

    async with httpx.AsyncClient(follow_redirects=False, timeout=10.0) as client:
        # 1. Root / Docs check
        res_docs = await client.get(f"{base_url}/docs")
        print(f"GET /docs                     -> Status: {res_docs.status_code}")
        assert res_docs.status_code == 200, "/docs failed!"

        # 2. Auth Me check
        res_me = await client.get(f"{base_url}/auth/me")
        print(f"GET /auth/me                  -> Status: {res_me.status_code}, Body: {res_me.json()}")
        assert res_me.status_code == 200

        # 3. Auth Login redirect check
        res_login = await client.get(f"{base_url}/auth/login")
        print(f"GET /auth/login               -> Status: {res_login.status_code}")
        location = res_login.headers.get("location", "")
        print(f"   Redirect Target: {location[:80]}...")
        assert res_login.status_code == 302
        assert "accounts.google.com" in location
        assert "http%3A%2F%2Flocalhost%3A7860%2Fauth%2Fcallback" in location or "http://localhost:7860/auth/callback" in location or "7860" in location

        # 4. Auth Callback check (missing state/code returns redirect to frontend with error parameter)
        res_cb = await client.get(f"{base_url}/auth/callback")
        print(f"GET /auth/callback (no code)  -> Status: {res_cb.status_code}")
        cb_location = res_cb.headers.get("location", "")
        print(f"   Redirect Target: {cb_location}")
        assert res_cb.status_code == 302
        assert "auth_error=" in cb_location

    print("\nALL OAUTH PORT & ROUTE VERIFICATIONS PASSED ON PORT 7860!")

if __name__ == "__main__":
    asyncio.run(verify())
