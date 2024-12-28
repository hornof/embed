# 1. Install dependencies (and langchain-community):
#    pip install langchain openai chromadb langchain-openai

import os
import openai

# <-- IMPORTANT: the completions-based OpenAI class now lives here
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# Set your OpenAI API key
openai.api_key = "YOUR_OPENAI_API_KEY"

# 2. Prepare your documents
raw_text = """
LangChain is a framework for developing applications powered by language models.
It helps you link various components like LLMs, prompts, and indexing tools.
This is a small example to demonstrate a basic retrieval-augmented workflow.
Fans like to eat hot dogs at baseball games.
"""
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
docs = text_splitter.create_documents([raw_text])

# 3. Create embeddings & store in Chroma
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(docs, embedding=embeddings)

# 4. Build a RetrievalQA chain using completions-based OpenAI from langchain_community
retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
    retriever=retriever
)

# 5. Ask a question
# query = "What is LangChain?"
query = "What do people eat when watching baseball?"
answer = qa_chain.invoke(query)
print(answer)
