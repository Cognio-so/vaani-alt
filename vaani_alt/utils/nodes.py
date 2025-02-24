import os
from typing import Dict, List, Tuple, Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_core.prompts import PromptTemplate

from vaani_alt.utils.state import VaaniState
from vaani_alt.utils.tools import (
    web_search, 
    process_document, 
    split_documents, 
    create_qdrant_from_documents,
    query_qdrant,
    generate_image
)

# Initialize LLM models
orchestrator_model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7
)

extra_questions_model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3
)

summary_model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",
    temperature=0.2
)

# System prompts
ORCHESTRATOR_PROMPT = """You are the orchestrator for Vaani.pro, an intelligent assistant that routes user queries to appropriate specialized agents. 
Your role is to analyze the user query and determine which specialized agent should handle it:

1. RAG Agent - for queries related to uploaded documents (PDF, DOC, TXT)
2. Image Generation Agent - for requests to create or generate images
3. Web Search Agent - for queries requiring real-time information from the web

Remember the context of the conversation and previous messages when making decisions.
Respond in a helpful, friendly tone and route the query appropriately."""

SUMMARIZER_PROMPT = """Summarize the conversation history in a concise, informative manner capturing the key points, 
questions asked, and information provided. This summary will be used to maintain context in the conversation."""

IMAGE_AGENT_PROMPT = """You are an image generation specialist. Your job is to create good prompts for image generation
based on user requests and then generate the images. Create detailed, descriptive prompts that will result in
high-quality images that match what the user is asking for."""

WEBSEARCH_AGENT_PROMPT = """You are a web search specialist. Use the provided search tools to find relevant, 
up-to-date information from the web based on the user's query. Provide comprehensive but concise answers, citing sources
when possible."""

RAG_AGENT_PROMPT = """You are a document analysis specialist. Your job is to analyze uploaded documents and answer 
questions based on their content. Be precise and reference specific parts of the document in your answers."""

EXTRA_QUESTIONS_PROMPT = """Based on the user's query, generate a related follow-up question that might help clarify 
their needs or provide additional value. Keep the question concise, relevant, and helpful."""

def user_input_node(state: VaaniState) -> Dict:
    """Process user input and initialize state."""
    # This is intentionally kept minimal as it's mainly a passthrough
    # that maintains the user's input in the state
    return {"agent_name": "orchestrator"}

def check_message_threshold(state: VaaniState) -> str:
    """Check if the message threshold is exceeded and summarization is needed."""
    if len(state.chat_history) > 4:  # Threshold set to 4 (2 Q&A pairs)
        return "summarize"
    return "continue"

def summarize_chat_history(state: VaaniState) -> Dict:
    """Summarize the chat history beyond the recent threshold."""
    # Get all but the most recent 4 messages (2 Q&A pairs)
    messages_to_summarize = state.chat_history[:-4]
    
    if not messages_to_summarize:
        return {"summary": state.summary}  # Keep existing summary if no messages to summarize
    
    # Convert messages to text format for summarization
    conversation_text = "\n".join([
        f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
        for msg in messages_to_summarize
    ])
    
    # Prepare prompt for summarization
    if state.summary:
        # If there's an existing summary, include it
        prompt = f"""
        Previous conversation summary: {state.summary}
        
        New conversation to summarize:
        {conversation_text}
        
        Create an updated summary that integrates both the previous summary and the new conversation:
        """
    else:
        prompt = f"""
        Conversation to summarize:
        {conversation_text}
        
        Create a concise summary of this conversation:
        """
    
    # Get summary from model
    summary_response = summary_model.invoke([
        SystemMessage(content=SUMMARIZER_PROMPT),
        HumanMessage(content=prompt)
    ])
    
    new_summary = summary_response.content
    
    # Remove summarized messages from history to save context window
    updated_chat_history = state.chat_history[-4:]
    
    return {
        "summary": new_summary,
        "chat_history": updated_chat_history
    }

def orchestrator_node(state: VaaniState) -> Dict:
    """Orchestrate the routing of user input to appropriate agents."""
    # Prepare chat history and summary for context
    messages = []
    
    # If there's a summary, include it in a system message
    if state.summary:
        messages.append(SystemMessage(
            content=f"{ORCHESTRATOR_PROMPT}\n\nConversation summary so far: {state.summary}"
        ))
    else:
        messages.append(SystemMessage(content=ORCHESTRATOR_PROMPT))
    
    # Add chat history for context
    messages.extend(state.chat_history)
    
    # Add current user input
    messages.append(HumanMessage(content=state.user_input))
    
    # Determine which agent should handle the query
    response = orchestrator_model.invoke(messages)
    
    # Analyze response to determine next agent
    content = response.content.lower()
    
    # Route to appropriate agent based on content analysis
    if state.file_url and ("document" in content or "pdf" in content or "file" in content or "read" in content):
        next_agent = "rag_agent"
        task_description = "Answer questions based on the uploaded document."
    elif "image" in content or "picture" in content or "draw" in content or "generate" in content:
        next_agent = "image_agent"
        task_description = "Generate images based on the user request."
    elif "search" in content or "find" in content or "look up" in content or "current" in content or "latest" in content:
        next_agent = "websearch_agent"
        task_description = "Search the web for information."
    else:
        # Default to orchestrator answering directly for general queries
        next_agent = "output"
        task_description = "General conversation."
    
    # Add AI response to chat history
    updated_chat_history = state.chat_history + [
        HumanMessage(content=state.user_input),
        AIMessage(content=response.content)
    ]
    
    return {
        "agent_name": next_agent,
        "task_description": task_description,
        "chat_history": updated_chat_history
    }

def extra_questions_node(state: VaaniState) -> Dict:
    """Generate follow-up questions in parallel to the main workflow."""
    # Use a simpler, faster model for generating extra questions
    messages = [
        SystemMessage(content=EXTRA_QUESTIONS_PROMPT),
        HumanMessage(content=state.user_input)
    ]
    
    response = extra_questions_model.invoke(messages)
    
    return {"extra_question": response.content}

def rag_agent_node(state: VaaniState) -> Dict:
    """Process document-based queries using RAG."""
    # Process the document if it's not already processed
    if state.file_url and not state.rag_context:
        documents = process_document(state.file_url, state.file_content)
        
        # Split documents into smaller chunks
        splits = split_documents(documents)
        
        # Create vector store
        collection_name = f"vaani_docs_{hash(state.file_url)}"
        create_qdrant_from_documents(splits, collection_name)
        
        # Generate a brief overview of the document
        doc_overview = "\n".join([doc.page_content[:200] + "..." for doc in documents[:2]])
        rag_context = f"Document Overview: {doc_overview}"
    else:
        collection_name = f"vaani_docs_{hash(state.file_url)}"
        rag_context = state.rag_context or "Document has been processed."
    
    # Query the vector store for relevant context
    query_docs = query_qdrant(state.user_input, collection_name)
    context = "\n\n".join([doc.page_content for doc in query_docs])
    
    # Prepare messages for the RAG agent
    messages = [
        SystemMessage(content=f"{RAG_AGENT_PROMPT}\n\nDocument context: {context}"),
    ]
    
    if state.summary:
        messages.append(SystemMessage(content=f"Conversation summary: {state.summary}"))
    
    messages.extend(state.chat_history[-4:])  # Add recent conversation
    messages.append(HumanMessage(content=state.user_input))
    
    # Get response from model
    response = orchestrator_model.invoke(messages)
    
    # Update chat history
    updated_chat_history = state.chat_history + [
        HumanMessage(content=state.user_input),
        AIMessage(content=response.content)
    ]
    
    return {
        "agent_name": "output",
        "chat_history": updated_chat_history,
        "rag_context": context[:500]  # Store a snippet of the context for future reference
    }

def image_agent_node(state: VaaniState) -> Dict:
    """Generate images based on user requests."""
    # Prepare messages for the image agent
    messages = [
        SystemMessage(content=IMAGE_AGENT_PROMPT),
    ]
    
    if state.summary:
        messages.append(SystemMessage(content=f"Conversation summary: {state.summary}"))
    
    messages.extend(state.chat_history[-4:])  # Add recent conversation
    messages.append(HumanMessage(content=f"Create an image based on this request: {state.user_input}"))
    
    # Get enhanced prompt from model
    response = orchestrator_model.invoke(messages)
    
    # Extract the enhanced prompt
    enhanced_prompt = response.content
    
    # Generate the image
    try:
        image_paths = generate_image(enhanced_prompt)
        
        # Prepare a response that includes information about the generated image
        result_message = (
            f"I've generated the requested image using this prompt: \"{enhanced_prompt}\"\n\n"
            f"The image has been created and is ready for viewing."
        )
        
        # Update chat history
        updated_chat_history = state.chat_history + [
            HumanMessage(content=state.user_input),
            AIMessage(content=result_message)
        ]
        
        return {
            "agent_name": "output",
            "chat_history": updated_chat_history,
            "image_urls": image_paths
        }
    
    except Exception as e:
        error_message = f"I encountered an error while generating the image: {str(e)}"
        
        # Update chat history with error message
        updated_chat_history = state.chat_history + [
            HumanMessage(content=state.user_input),
            AIMessage(content=error_message)
        ]
        
        return {
            "agent_name": "output",
            "chat_history": updated_chat_history
        }

def websearch_agent_node(state: VaaniState) -> Dict:
    """Search the web for information based on user queries."""
    # Use Tavily to get search results
    search_results = web_search.invoke(state.user_input)
    
    # Format search results
    formatted_results = "\n\n".join([
        f"Source: {result.get('url', 'N/A')}\n"
        f"Title: {result.get('title', 'N/A')}\n"
        f"Content: {result.get('content', 'N/A')}"
        for result in search_results
    ])
    
    # Prepare messages for the websearch agent
    messages = [
        SystemMessage(content=f"{WEBSEARCH_AGENT_PROMPT}\n\nSearch Results:\n{formatted_results}"),
    ]
    
    if state.summary:
        messages.append(SystemMessage(content=f"Conversation summary: {state.summary}"))
    
    messages.extend(state.chat_history[-4:])  # Add recent conversation
    messages.append(HumanMessage(content=state.user_input))
    
    # Get response from model
    response = orchestrator_model.invoke(messages)
    
    # Update chat history
    updated_chat_history = state.chat_history + [
        HumanMessage(content=state.user_input),
        AIMessage(content=response.content)
    ]
    
    return {
        "agent_name": "output",
        "chat_history": updated_chat_history,
        "search_results": search_results
    }