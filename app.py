import streamlit as st
import os

from langchain_groq import ChatGroq
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Tools
api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

search = DuckDuckGoSearchRun(name="Search")


# App
st.title("🔎 LangChain - Chat with search")
"""
In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions of an agent in an interactive Streamlit app.
Try more LangChain 🤝 Streamlit Agent examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
"""

# Side bar settings
st.sidebar.title('Settings')
groq_api_key = st.sidebar.text_input("Enter your Groq API Key: ", type="password")
#st.secrets['GROQ_API_KEY']

if not groq_api_key:
    st.info("Please enter the Groq API Key to continue.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role":"assistant",
         "content":"Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt:=st.chat_input(placeholder="What is machine learning?"):
    st.session_state.messages.append({"role":"user", "content":prompt})
    st.chat_message("user").write(prompt)

    llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192", streaming=True)
    tools = [search, arxiv, wiki]

    search_engine = initialize_agent(tools, llm, agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION, handling_parsing_errors=True)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_engine.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({"role":"assistant", "content":response})
        st.write(response)

    
