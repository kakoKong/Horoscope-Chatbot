from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import HuggingFaceHub
from langchain.chains import LLMChain
import pinecone
from dotenv import load_dotenv
import os

load_dotenv()

loader = TextLoader('./horoscope.txt')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=4)
docs = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings()

pinecone.init(
    api_key= os.getenv('PINECONE_API_KEY'),
    environment='gcp-starter'
)

index_name = "mytestindex"

if index_name not in pinecone.list_indexes():
  pinecone.create_index(name=index_name, metric="cosine", dimension=768)

docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)

repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
llm = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"temperature": 0.7}, huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY')
)

from langchain import PromptTemplate

template = """
You are a seer. These Human will ask you a questions about their life. Using the following context, answer them in less than 2 sentences. Add emojis at the end of every responses
Answer your answer without mentioning any stars. 
Always start your message with: this month for you as a <star sign>...

{context}

{question}
"""

prompt = PromptTemplate(template=template, input_variables=["context", "question"])

from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={"k": 2}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt},
)

import streamlit as st
    
st.set_page_config(page_title="🤗💬 HugChat")
with st.sidebar:
    st.title('🤗💬 HugChat')
    st.markdown('📖 Learn how to build this app in this [blog](https://blog.streamlit.io/how-to-build-an-llm-powered-chatbot-with-streamlit/)!')

# Function for generating LLM response
def generate_response(input):
    result = qa_chain(input)
    res = result["result"]
    return res

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    print('hello')
    st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User-provided prompt
if input := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": input})
    with st.chat_message("user"):
        st.write(input)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(input) 
            st.write(response) 
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)