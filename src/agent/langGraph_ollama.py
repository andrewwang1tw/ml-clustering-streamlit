from dotenv import load_dotenv
import os
import json

from langchain_tavily import TavilySearch
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage

from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END

from typing import Annotated
from typing_extensions import TypedDict

load_dotenv()    
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
langchain_tracing_v2 = os.getenv("LANGCHAIN_TRACING_V2")
langchain_project = os.getenv("LANGCHAIN_PROJECT")

print(f"LANGCHAIN_API_KEY: {langchain_api_key}")
print(f"LANGCHAIN_TRACING_V2: {langchain_tracing_v2}")
print(f"LANGCHAIN_PROJECT: {langchain_project}")
print(f"LANGCHAIN_PROJECT: {OLLAMA_MODEL}")

OLLAMA_MODEL = 'gpt-oss:120b-cloud'

#--------------------------------------------------------------------
# Tool and llama3.1
#--------------------------------------------------------------------
tool = TavilySearch(max_results=5)
tools = [tool]
#response = tool.invoke("6202 盛群 2025 營運與股價 新聞 ?")
#print(json.dumps(response, indent=2))

# LLM (llama3.1) 
llm = ChatOllama(
    model=OLLAMA_MODEL, 
    temperature=0, 
    base_url="http://localhost:11434"
)

# Tell LLM(llama3.1) which tools it can call
llm_with_tools = llm.bind_tools(tools)

#--------------------------------------------------------------------
# 
#--------------------------------------------------------------------
class State(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot(state: State):     
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Prebuild ToolNode
tool_node = ToolNode(tools=tools)  

#---------------------------------------
# Build the state graph
#---------------------------------------
graph_builder = StateGraph(State)

# Add nodes 
graph_builder.add_node("chatbot", chatbot)  
graph_builder.add_node("tools", tool_node)

# Add edges
graph_builder.add_edge(START, "chatbot")

# Add Conditional edge: if the last message has tool calls, go to the tool_node
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

# After the tools node, go back to the chatbot for further processing
graph_builder.add_edge("tools", "chatbot")
graph = graph_builder.compile()


#-----------------------------------------------------------------------------------------
# stream_graph_updates
#-----------------------------------------------------------------------------------------
def stream_graph_updates(user_input: str):        
    last_msg = None 
    all_messages = [] 
    
    response = graph.stream({"messages": [{"role": "user", "content": user_input}]}) 
    for res in response:                
        for value in res.values():
            value["messages"][-1].pretty_print()  
            last_msg = value["messages"][-1].content   
            if last_msg:       
                all_messages.append(last_msg)
    
    return all_messages, last_msg        
                    
  
#-----------------------------------------------------------------------------------------
# Only ollama
#-----------------------------------------------------------------------------------------
def stream_ollama_updates(user_input: str):  
    user_input_message = HumanMessage(content=user_input)
    response = llm.invoke([user_input_message])
    return response.content
      