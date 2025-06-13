#!/usr/bin/env python3
"""
Standalone test script for the simplified chat agent.
This avoids import conflicts by explicitly using local modules.
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

print("🔧 Setting up imports...")
print(f"Using source directory: {src_dir}")

try:
    # Import our local modules
    from agent.graph import graph
    from agent.state import ChatState
    
    print("✅ Successfully imported simplified agent modules!")
    print(f"Graph nodes: {list(graph.nodes.keys())}")
    
    # Test the simplified agent
    print("\n🤖 Testing simplified chat agent...")
    
    state = {"messages": [{"role": "user", "content": "Hello! What is LangGraph?"}]}
    
    result = graph.invoke(state)
    
    print("✅ SUCCESS! The simplified chat agent works!")
    print(f"👤 User: {state['messages'][0]['content']}")
    print(f"🤖 Assistant: {result['messages'][-1].content}")
    
    # Test with a follow-up message
    print("\n🔄 Testing conversation continuity...")
    state["messages"] = result["messages"] + [{"role": "user", "content": "Can you explain it simply?"}]
    
    result2 = graph.invoke(state)
    print(f"👤 User: Can you explain it simply?")
    print(f"🤖 Assistant: {result2['messages'][-1].content}")
    
    print("\n✨ Simplified agent is working perfectly!")
    print("You now have a basic LangGraph chat agent with:")
    print("  ✅ Single chat node")
    print("  ✅ Simple state management")
    print("  ✅ Basic configuration")
    print("  ✅ Easy to understand and extend")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    print("\nTroubleshooting:")
    print("1. Make sure GEMINI_API_KEY is set in your .env file")
    print("2. Check that all files were properly simplified")
    print("3. Restart your Python kernel if running in a notebook") 