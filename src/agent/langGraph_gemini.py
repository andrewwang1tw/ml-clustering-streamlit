from dotenv import load_dotenv
import os
import streamlit as st
import time
from typing import Annotated
from typing_extensions import TypedDict

from langchain_tavily import TavilySearch
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END

import google.generativeai as genai


load_dotenv()    
MODEL_LLM = os.getenv("GEMINI_MODEL")
MODEL_KEY = os.getenv("GOOGLE_API_KEY")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
langchain_tracing_v2 = os.getenv("LANGCHAIN_TRACING_V2")
langchain_project = os.getenv("LANGCHAIN_PROJECT")

# Read secrets from Streamlit Cloud's secrets manager
# OLLAMA_MODEL = st.secrets.get("OLLAMA_MODEL")
# TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY")
# LANGCHAIN_API_KEY = st.secrets.get("LANGCHAIN_API_KEY")
# LANGCHAIN_TRACING_V2 = st.secrets.get("LANGCHAIN_TRACING_V2")
# LANGCHAIN_PROJECT = st.secrets.get("LANGCHAIN_PROJECT")

print(f"LANGCHAIN_API_KEY: {langchain_api_key}")
print(f"LANGCHAIN_TRACING_V2: {langchain_tracing_v2}")
print(f"LANGCHAIN_PROJECT: {langchain_project}")
print(f"\nGEMINI_MODEL: {MODEL_LLM}")
print(f"GOOGLE_API_KEY: {MODEL_KEY}")


# # 確保你的 GOOGLE_API_KEY 環境變數已設定 
# genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
# print("可用且支援 generateContent 的模型：")
# try:
#     for m in genai.list_models():
#         if "generateContent" in m.supported_generation_methods:
#             print(f"- {m.name}")
# except Exception as e:
#     print(f"列出模型時發生錯誤: {e}")

#--------------------------------------------------------------------
# Tool and llama3.1
#--------------------------------------------------------------------
tool = TavilySearch(max_results=10)
tools = [tool]
#response = tool.invoke("6202 盛群 2025 營運與股價 新聞 ?")
#print(json.dumps(response, indent=2))

# LLM (llama3.1) 
llm = ChatGoogleGenerativeAI(
    model=MODEL_LLM, 
    temperature=0, 
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
# stream_graph_updates2
#-----------------------------------------------------------------------------------------              
def stream_graph_updates2(user_input: str):        
    last_msg_content = None     
    response = graph.stream({"messages": [{"role": "user", "content": user_input}]}) 
    for res in response:                
        for value in res.values():
            value["messages"][-1].pretty_print()  
            
            last_msg_content = value["messages"][-1].content               
            if last_msg_content:    
                if len(last_msg_content) > 200:
                    
                    # 將大區塊內容分成較小的片段
                    chunk_size = 10  # 每次模擬輸出 10 個字元                    
                    for i in range(0, len(last_msg_content), chunk_size):
                        chunk = last_msg_content[i:i + chunk_size]
                        
                        # 立即產出這一小塊內容
                        yield chunk
                        
                        # 模擬延遲：讓 Streamlit 有時間顯示，並給用戶「正在思考」的感覺 
                        # # 調整這個數值來控制速度 (0.01 秒通常不錯)
                        time.sleep(0.01) 
                        
                else:
                    # 對於短小的一般內容（例如 AI 的簡短判斷），直接產出
                    yield last_msg_content                    
                    time.sleep(0.5)
       
  
#-----------------------------------------------------------------------------------------
# Only LLM
#-----------------------------------------------------------------------------------------
def stream_llm_updates(user_input: str):  
    user_input_message = HumanMessage(content=user_input)
    response = llm.invoke([user_input_message])
    return response.content
      