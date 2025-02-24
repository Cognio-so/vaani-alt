import os
from typing import Dict, List, Tuple, Any, TypedDict, Optional, Annotated
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
# Remove SQLite checkpointing for Langgraph Studio
# from langgraph.checkpoint.sqlite import SqliteSaver
# import sqlite3

from vaani_alt.utils.state import VaaniState
from vaani_alt.utils.nodes import (
    user_input_node,
    check_message_threshold,
    summarize_chat_history,
    orchestrator_node,
    extra_questions_node,
    rag_agent_node,
    image_agent_node,
    websearch_agent_node
)

# Load environment variables
load_dotenv()

# Remove SQLite database setup
# db_path = "vaani_state.db"
# conn = sqlite3.connect(db_path, check_same_thread=False)
# memory = SqliteSaver(conn)

def vaani_graph() -> StateGraph:
    """Create and return the Vaani.pro workflow graph."""
    # Initialize the graph with our state
    workflow = StateGraph(VaaniState)
    
    # Add all nodes to the graph
    workflow.add_node("user_input", user_input_node)
    workflow.add_node("summarize", summarize_chat_history)
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("extra_questions", extra_questions_node)
    workflow.add_node("rag_agent", rag_agent_node)
    workflow.add_node("image_agent", image_agent_node)
    workflow.add_node("websearch_agent", websearch_agent_node)
    
    # Define the edges of the graph
    
    # Start with user input
    workflow.set_entry_point("user_input")
    
    # Check message threshold after user input and route accordingly
    workflow.add_conditional_edges(
        "user_input",
        check_message_threshold,
        {
            "summarize": "summarize",
            "continue": "orchestrator"
        }
    )
    
    # After summarization, proceed to orchestrator
    workflow.add_edge("summarize", "orchestrator")
    
    # Define routing based on agent_name in state
    workflow.add_conditional_edges(
        "orchestrator",
        lambda state: state.agent_name,
        {
            "rag_agent": "rag_agent",
            "image_agent": "image_agent",
            "websearch_agent": "websearch_agent",
            "output": END
        }
    )
    
    # Add edges from specialized agents to END
    workflow.add_conditional_edges(
        "rag_agent",
        lambda state: state.agent_name,
        {
            "output": END
        }
    )
    
    workflow.add_conditional_edges(
        "image_agent",
        lambda state: state.agent_name,
        {
            "output": END
        }
    )
    
    workflow.add_conditional_edges(
        "websearch_agent",
        lambda state: state.agent_name,
        {
            "output": END
        }
    )
    
    # Parallel process for extra questions
    # This runs alongside the main workflow
    workflow.add_edge("user_input", "extra_questions", parallel=True)
    workflow.add_edge("extra_questions", END)
    
    # Compile the graph without checkpointing for Langgraph Studio
    return workflow.compile()

# Entry point for direct execution
if __name__ == "__main__":
    graph = vaani_graph()
    
    # Example usage
    result = graph.invoke({
        "user_input": "Can you tell me about the latest AI developments?",
        "chat_history": []
    })
    
    print("Result:", result)