import streamlit as st
import os
from openai import OpenAI
from os import environ

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from pypdf import PdfReader
from langchain_core.documents import Document

#Functions-------------------
@st.cache_resource # ensure only run once
def read(uploaded_file):
    if uploaded_file.type == "application/pdf":
        try:
            reader = PdfReader(uploaded_file)
            file_content = ""
            for page in reader.pages:
                file_content += page.extract_text()
        except Exception as e: 
            print("failed to read pdf")
            print(f"error type is {e}")
            
    elif uploaded_file.type == "text/plain" or uploaded_file.type == "text/markdown":
        try:
            file_content = uploaded_file.read().decode("utf-8")
        except Exception as e:
            print("failed to read text")
            print(f"error type is {e}")
    else:
        print('file type not supported')
        return None
    return file_content

def get_vectorstore_from_file(uploaded_files):
    """
    This function handles:
    1. Reading the file
    2. Chunking the text (Req 3.1)
    3. Embedding the chunks
    4. Storing them in a vector database (Chroma)
    """

    if uploaded_files is None:
        return None
    
    st.info("You File has been uploaded")
    all_chunks = []

    #chunck
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )

    #Read the file
    for file in uploaded_files:
        file_content = read(file)
        chunks = text_splitter.split_text(file_content)
        for chunk in chunks:
            all_chunks.append(
                Document(
                    page_content=chunk, 
                    metadata={"source": file.name} # This links the chunk to its file, matadata - docments in langchain
                )
            )


    if not all_chunks:
        st.error("File is empty or could not be split.")
        return None
    
    embeddings_client = OpenAIEmbeddings(
        model="openai.text-embedding-3-large",
        api_key=os.environ["API_KEY"],
        base_url="https://api.ai.it.cornell.edu",
    )
    
    vectorstore = Chroma.from_documents(
        documents=all_chunks, 
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
uploaded_files = st.file_uploader(
    "Upload an article",
    type=("txt", "md","pdf"),
    accept_multiple_files=True
)

# Read the file and make a index, get the vectorstore of knowledge base

vectorstore = None
if uploaded_files:
    vectorstore = get_vectorstore_from_file(uploaded_files)
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
        with st.spinner("Thinking..."):
            # Retrieve 
            docs = vectorstore.similarity_search(question, k=5) # Get top 5 chunks

            #Augmentation
            context = format_docs(docs)
            system_instructions = (
                "You are a helpful assistant for question answering.\n"
                "Use ONLY the provided context to answer concisely (<=3 sentences).\n"
                "Tell where is the answer come from according to the source.\n"
                "If the answer isn't in the context, say you don't know.\n\n"
                f"Context:\n{context}"
            )


            
            messages_for_llm = [
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": question}
            ]

            stream = client.chat.completions.create(
                model="openai.gpt-4o",
                messages=messages_for_llm, # Send the RAG prompt
                stream=True
            )
            response = st.write_stream(stream)
        #give a ref to sources
        st.markdown("**Sources (Relevant Chunks):**")
        sources = set()
        for d in docs: 
            sources.add(d.metadata['source'])
        
        if sources:
            st.markdown(", ".join(sources))
    # Append the assistant's response to the messages
    st.session_state.messages.append({"role": "assistant", "content": response})                                      