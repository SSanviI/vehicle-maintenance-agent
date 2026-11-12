import sys
import asyncio
from pathlib import Path

root = Path(__file__).parent.parent
sys.path.insert(0, str(root))

from backend.app.database.session import get_session_factory, init_db
from backend.app.memory.repository import MemoryRepository
from backend.app.agent.state import AgentState

async def main():
    await init_db()
    factory = get_session_factory()

    # 1. Create a conversation and save state
    async with factory() as session:
        repo = MemoryRepository(session)
        conv = await repo.get_or_create_conversation()
        cid = str(conv.id)
        print(f"Created conversation ID: {cid}")

        state1 = AgentState(conversation_id=cid)
        state1.update_feature("Vehicle_Age", 5)
        print("State1 known features:", state1.get_known_features())

        await repo.save_agent_state(cid, state1)
        print("Saved state1 to DB.")

    # 2. Re-open session and reload conversation from DB
    async with factory() as session:
        repo = MemoryRepository(session)
        conv_reloaded = await repo.get_or_create_conversation(cid)
        print("Reloaded conv.feature_state type:", type(conv_reloaded.feature_state))
        print("Reloaded conv.feature_state content:", conv_reloaded.feature_state)
        print("Reloaded conv.declined_features content:", conv_reloaded.declined_features)

        state2 = AgentState.from_json(
            {
                "feature_state": conv_reloaded.feature_state,
                "asked_features": conv_reloaded.asked_questions,
                "declined_features": conv_reloaded.declined_features,
            },
            conversation_id=cid
        )
        print("State2 known features:", state2.get_known_features())
        print("State2 context dict:", state2.to_context_dict())

if __name__ == "__main__":
    asyncio.run(main())
