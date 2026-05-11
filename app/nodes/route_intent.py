"""LangGraph node: route user intent to the correct workflow."""

from app.state import AgentState


def route_intent(state: AgentState) -> AgentState:
    """Determine the user's intent and set up routing.

    Intent is expected to be pre-set by the caller (GUI / API).
    Validates and normalizes the intent value.
    """
    intent = state.get("intent", "search")
    valid_intents = {"import", "delete", "search", "summarize"}

    if intent not in valid_intents:
        state["error"] = f"Unknown intent: {intent}. Must be one of {valid_intents}"
        state["intent"] = "search"
    else:
        state["error"] = ""

    return state


def route_by_intent(state: AgentState) -> str:
    """Conditional edge: route to the next node based on intent."""
    intent = state.get("intent", "search")
    return intent
