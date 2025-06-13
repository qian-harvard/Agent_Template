#!/usr/bin/env python3
"""
Test script for the MCP-enabled chat agent with pandapower integration.
"""

import os
import sys
from pathlib import Path

# Get the absolute path to the src directory
current_dir = Path(__file__).parent
src_dir = current_dir / "src"

# Remove any conflicting 'agent' modules from cache
modules_to_remove = [key for key in sys.modules.keys() if key.startswith('agent')]
for module in modules_to_remove:
    del sys.modules[module]

# Add our src directory to the front of Python path
sys.path.insert(0, str(src_dir))

print("🔧 Setting up MCP-enabled chat agent...")
print(f"Using source directory: {src_dir}")

try:
    # Import our enhanced modules
    from agent.graph import graph
    from agent.state import ChatState
    
    print("✅ Successfully imported MCP-enabled agent modules!")
    print(f"Graph nodes: {list(graph.nodes.keys())}")
    
    # Test the MCP-enhanced agent with pandapower request
    print("\n🤖 Testing MCP-enabled chat agent with pandapower request...")
    
    # Use the exact prompt requested by the user
    test_prompt = "could you use pandapower to solve the power flow of /Users/qianzhang/Desktop/Agent/Agent_Template/backend/src/mcp/test_case.json"
    
    state = {"messages": [{"role": "user", "content": test_prompt}]}
    
    print(f"👤 User: {test_prompt}")
    print("🔄 Processing request...")
    
    result = graph.invoke(state)
    
    print("✅ SUCCESS! The MCP-enabled chat agent responded!")
    print("🤖 Assistant:")
    print("=" * 80)
    print(result['messages'][-1].content)
    print("=" * 80)
    
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    print("\nTroubleshooting:")
    print("1. Make sure OPENAI_API_KEY is set in your .env file")
    print("2. Ensure MCP server dependencies are installed")
    print("3. Check that the test_case.json file exists")
    print("4. Verify that pandapower is installed in the environment")
    print("5. If the power flow fails, it may be due to MCP server state not being preserved between tool calls") 