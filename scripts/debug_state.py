import sys
from pathlib import Path
root = Path(__file__).parent.parent
sys.path.insert(0, str(root))

from backend.app.agent.state import AgentState
from backend.app.agent.actions import AgentTools

def test():
    state = AgentState(conversation_id="test")
    print("Initial declined:", state.declined_features)
    tools = AgentTools(None)
    tools.mark_feature_declined(state, "Brake_Condition")
    print("After mark_feature_declined:", state.declined_features)
    print("Context dict:", state.to_context_dict())

if __name__ == "__main__":
    test()
