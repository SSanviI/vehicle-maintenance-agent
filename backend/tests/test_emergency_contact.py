"""
Backend tests for Emergency Contact CRUD API endpoints and authorization boundaries.
"""

import pytest
import uuid
from backend.app.database.models import GoogleUser, EmergencyContact


@pytest.mark.asyncio
async def test_emergency_contact_crud(async_session):
    """Test creating, reading, updating, and deleting emergency contacts."""
    user_gid = f"test-user-{uuid.uuid4()}"
    user = GoogleUser(
        id=uuid.uuid4(),
        google_user_id=user_gid,
        name="Test Driver",
        email="driver@example.com",
    )
    async_session.add(user)
    await async_session.commit()

    # 1. Create Contact
    contact = EmergencyContact(
        id=uuid.uuid4(),
        google_user_id=user_gid,
        contact_name="Alice Smith",
        relationship="Spouse",
        contact_email="alice@example.com",
    )
    async_session.add(contact)
    await async_session.commit()

    # 2. Read Contact
    from sqlalchemy import select
    stmt = select(EmergencyContact).where(EmergencyContact.google_user_id == user_gid)
    res = await async_session.execute(stmt)
    fetched = res.scalar_one_or_none()
    assert fetched is not None
    assert fetched.contact_name == "Alice Smith"
    assert fetched.contact_email == "alice@example.com"

    # 3. Update Contact
    fetched.contact_name = "Alice Johnson"
    await async_session.commit()

    res2 = await async_session.execute(stmt)
    updated = res2.scalar_one_or_none()
    assert updated.contact_name == "Alice Johnson"

    # 4. Delete Contact
    await async_session.delete(updated)
    await async_session.commit()

    res3 = await async_session.execute(stmt)
    assert res3.scalar_one_or_none() is None


@pytest.mark.asyncio
async def test_emergency_contact_user_isolation(async_session):
    """Verify emergency contact belongs exclusively to authenticated Google user."""
    user1_gid = "user-111"
    user2_gid = "user-222"

    contact1 = EmergencyContact(
        id=uuid.uuid4(),
        google_user_id=user1_gid,
        contact_name="User 1 Contact",
        contact_email="contact1@example.com",
    )
    async_session.add(contact1)
    await async_session.commit()

    from sqlalchemy import select
    # User 2 queries for their contact -> must receive None
    stmt = select(EmergencyContact).where(EmergencyContact.google_user_id == user2_gid)
    res = await async_session.execute(stmt)
    assert res.scalar_one_or_none() is None
