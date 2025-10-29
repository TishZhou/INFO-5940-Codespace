import streamlit as st
import os
from openai import OpenAI
from os import environ

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from pypdf import PdfReader

#Functions-------------------
@st.cache_resource # ensure only run once
def get_vectorstore_from_file(uploaded_file):
    """
    This function handles:
    1. Reading the file
    2. Chunking the text (Req 3.1)
    3. Embedding the chunks
    4. Storing them in a vector database (Chroma)
    """

    if uploaded_file is None:
        return None
    
    st.info("You File has been uploaded")

    #Read the file
    file_content = uploaded_file.read().decode("utf-8")
    print(file_content)
    #chunck
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=70,
        length_function=len
    )
    chunks = text_splitter.create_documents([file_content])

    if not chunks:
        st.error("File is empty or could not be split.")
        return None

    #try:
    embeddings_client = OpenAIEmbeddings(
        model="openai.text-embedding-3-large",
        api_key=os.environ["API_KEY"],
        # IMPORTANT: The embedding API URL might be different!
        # It often needs '/v1' at the end.
            base_url="https://api.ai.it.cornell.edu",
    )
    
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings_client
    )
    st.success("File successfully indexed!")
    return vectorstore

    # except Exception as e:
    #     st.error(f"Error creating vector store: {e}")
    #     st.error("Please check your API key and base_url. The base_url for embeddings might be different from chat.")
    #     return None

def format_docs(docs):
    """join docs to let llm read"""
    return "\n\n---\n\n".join(d.page_content for d in docs)

#------------------------------------------------------------------





#Original AI setup
client = OpenAI(
	api_key=os.environ["API_KEY"],
	base_url="https://api.ai.it.cornell.edu",
)

#the interface
st.title("📝 File Q&A with OpenAI")
uploaded_file = st.file_uploader("Upload an article", type=("txt", "md"))

# Read the file and make a index, get the vectorstore of knowledge base

vectorstore = None
if uploaded_file:
    vectorstore = get_vectorstore_from_file(uploaded_file)
    print(vectorstore)

question = st.chat_input(
    "Your knowledgebase is ready. Ask something about the article",
    disabled=(vectorstore is None),
)
# the message history / stored data
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Ask something about the article"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])




# Main logic
if question and vectorstore:
    # Append the user's question to the messages
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").write(question)

    with st.chat_message("assistant"):
        # Retrieve (This is the 'R' in RAG)
        docs = vectorstore.similarity_search(question, k=5) # Get top 5 chunks

        #Augmentation
        context = format_docs(docs)
        system_instructions = (
            "You are a helpful assistant for question answering.\n"
            "Use ONLY the provided context to answer concisely (<=3 sentences).\n"
            "If the answer isn't in the context, say you don't know.\n\n"
            f"Context:\n{context}"
        )


        
        messages_for_llm = [
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": question}
        ]

        stream = client.chat.completions.create(
            model="gpt-4o",
            messages=messages_for_llm, # Send the RAG prompt
            stream=True
        )
        response = st.write_stream(stream)

    # Append the assistant's response to the messages
    st.session_state.messages.append({"role": "assistant", "content": response})