import os
import sys
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient

from agent.state import ChatState
from agent.configuration import Configuration
from agent.prompts import get_current_date, chat_instructions

load_dotenv()

if os.getenv("OPENAI_API_KEY") is None:
    raise ValueError("OPENAI_API_KEY is not set")

# Global MCP client instance
_mcp_client = None

def get_mcp_client(configurable: Configuration) -> MultiServerMCPClient:
    """Get or create the MCP client instance."""
    global _mcp_client
    
    if _mcp_client is None and configurable.enable_mcp:
        # Set up MCP server configurations
        server_configs = {}
        
        # Get the current working directory for the MCP server
        current_dir = Path(__file__).parent.parent  # backend/src
        
        for server_name, server_config in configurable.mcp_servers.items():
            # Set the working directory to the backend/src folder
            config = server_config.copy()
            if config["cwd"] is None:
                config["cwd"] = str(current_dir)
            
            server_configs[server_name] = config
            
        print(f"🔧 Initializing MCP client with servers: {list(server_configs.keys())}")
        _mcp_client = MultiServerMCPClient(server_configs)
        
    return _mcp_client


async def get_mcp_tools(configurable: Configuration):
    """Get MCP tools asynchronously."""
    if not configurable.enable_mcp:
        return []
    
    try:
        mcp_client = get_mcp_client(configurable)
        if mcp_client:
            tools = await mcp_client.get_tools()
            print(f"🔧 Available MCP tools: {[tool.name for tool in tools]}")
            return tools
    except Exception as e:
        print(f"⚠️  MCP client initialization failed: {e}")
        return []
    
    return []


async def execute_tool_async(tool, tool_call):
    """Execute a tool call asynchronously."""
    try:
        if hasattr(tool, 'ainvoke'):
            result = await tool.ainvoke(tool_call.get("args", {}))
        else:
            # Fallback to sync invocation in thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                result = await asyncio.get_event_loop().run_in_executor(
                    executor, tool.invoke, tool_call.get("args", {})
                )
        return result
    except Exception as e:
        raise e


def chat_node(state: ChatState, config: RunnableConfig) -> ChatState:
    """Enhanced chat node with MCP support for tool calling.

    This node takes the user's message and generates a helpful response using
    the configured language model, with optional MCP tool integration.

    Args:
        state: Current graph state containing the conversation messages
        config: Configuration for the runnable, including LLM and MCP settings

    Returns:
        Dictionary with state update, including the AI's response message
    """
    configurable = Configuration.from_runnable_config(config)
    
    # Initialize the chat model - Use GPT-4o for better reasoning
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=configurable.temperature,
        max_retries=2,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    
    # Get MCP tools if enabled (run async operation in sync context)
    tools = []
    if configurable.enable_mcp:
        # Create a new event loop if one doesn't exist
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Run the async function
        if loop.is_running():
            # If loop is already running, we need to use different approach
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, get_mcp_tools(configurable))
                tools = future.result()
        else:
            tools = loop.run_until_complete(get_mcp_tools(configurable))
    
    # Bind tools to the model if available
    if tools:
        llm = llm.bind_tools(tools)
        print(f"✅ LLM bound with {len(tools)} MCP tools")
    
    # Format the prompt with current date
    current_date = get_current_date()
    
    # Enhanced prompt for MCP-enabled agent
    if tools:
        enhanced_prompt = f"""{chat_instructions.format(current_date=current_date)}

You have access to specialized tools for power system analysis via pandapower. When users ask about:
- Power flow analysis
- Network loading
- Electrical calculations  
- Pandapower operations
- Loading networks from files

Use the available tools to help them. For pandapower tasks, you can:
1. Load networks from files (like JSON files) using the load_network tool
2. Run power flow analysis using the run_power_flow tool
3. Perform contingency analysis using the run_contingency_analysis tool
4. Get network information using the get_network_info tool

CRITICAL: When a user asks you to "solve the power flow" or perform power analysis:
1. First use load_network tool to load the network file
2. Then immediately use run_power_flow tool to perform the analysis
3. Explain the results clearly

Don't just say you will do something - actually complete the full task by calling the necessary tools in sequence."""
    else:
        enhanced_prompt = chat_instructions.format(current_date=current_date)
    
    # Get the conversation messages and add the system prompt
    messages = [{"role": "system", "content": enhanced_prompt}] + state["messages"]
    
    # Generate response
    response = llm.invoke(messages)
    
    # Handle tool calls if present
    if hasattr(response, 'tool_calls') and response.tool_calls:
        print(f"🔧 Executing {len(response.tool_calls)} tool call(s)...")
        
        # Execute tool calls asynchronously
        async def execute_all_tools():
            tool_results = []
            for tool_call in response.tool_calls:
                try:
                    # Find the matching tool
                    matching_tool = None
                    for tool in tools:
                        if tool.name == tool_call["name"]:
                            matching_tool = tool
                            break
                    
                    if matching_tool:
                        result = await execute_tool_async(matching_tool, tool_call)
                        tool_results.append({
                            "tool_call_id": tool_call.get("id", f"call_{tool_call['name']}"),
                            "name": tool_call["name"],
                            "content": str(result)
                        })
                        print(f"✅ Tool '{tool_call['name']}' executed successfully")
                    else:
                        error_msg = f"Tool '{tool_call['name']}' not found"
                        tool_results.append({
                            "tool_call_id": tool_call.get("id", f"call_{tool_call['name']}"),
                            "name": tool_call["name"],
                            "content": error_msg
                        })
                        print(f"❌ {error_msg}")
                        
                except Exception as e:
                    error_msg = f"Tool '{tool_call['name']}' failed: {str(e)}"
                    tool_results.append({
                        "tool_call_id": tool_call.get("id", f"call_{tool_call['name']}"),
                        "name": tool_call["name"],
                        "content": error_msg
                    })
                    print(f"❌ {error_msg}")
            
            return tool_results
        
        # Execute tools
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, execute_all_tools())
                tool_results = future.result()
        else:
            tool_results = loop.run_until_complete(execute_all_tools())
        
        # Create tool messages for the conversation
        tool_messages = []
        for result in tool_results:
            tool_messages.append(ToolMessage(
                content=result["content"],
                tool_call_id=result["tool_call_id"]
            ))
        
        # Create a follow-up conversation with tool results
        follow_up_messages = messages + [response] + tool_messages
        
        # Get final response with tool results incorporated
        final_response = llm.invoke(follow_up_messages)
        return {"messages": [AIMessage(content=final_response.content)]}
    
    return {"messages": [AIMessage(content=response.content)]}


# Create the MCP-enhanced chat graph
builder = StateGraph(ChatState, config_schema=Configuration)

# Add the enhanced chat node
builder.add_node("chat", chat_node)

# Set up the simple flow: START -> chat -> END
builder.add_edge(START, "chat")
builder.add_edge("chat", END)

# Compile the graph
graph = builder.compile(name="mcp-chat-agent")
