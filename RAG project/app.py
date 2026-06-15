import os
import tempfile
import streamlit as st

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables (Make sure MISTRAL_API_KEY is configured in your .env)
load_dotenv()

st.set_page_config(page_title="PDF Chat Assistant", page_icon="📄", layout="wide")
st.title("📄 PDF Query Assistant using Mistral AI & RAG")

# Initialize session state variables for chat history, retriever, and file tracking
if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "current_file" not in st.session_state:
    st.session_state.current_file = None

# Sidebar for PDF upload and processing configuration
with st.sidebar:
    st.header("1. Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    
    st.markdown("---")
    st.header("Advanced Settings")
    chunk_size = st.slider("Chunk Size", min_value=500, max_value=2000, value=1000, step=100)
    chunk_overlap = st.slider("Chunk Overlap", min_value=50, max_value=500, value=100, step=50)

# Process PDF if a NEW file is uploaded
if uploaded_file is not None:
    # Check if this is a different file than the one currently in memory
    if st.session_state.current_file != uploaded_file.name:
        
        # Reset memory for the new file
        st.session_state.retriever = None
        st.session_state.messages = []
        
        with st.spinner(f"Processing '{uploaded_file.name}' and building database..."):
            try:
                # 1. Save uploaded file to a temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name

                # 2. Load the PDF
                loader = PyPDFLoader(tmp_file_path)
                docs = loader.load()

                # 3. Create chunks
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                chunks = splitter.split_documents(docs)

                # 4. Initialize embedding model
                embedding_model = MistralAIEmbeddings()

                # 5. Store chunks into ephemeral (in-memory) Chroma DB
                vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=embedding_model
                )

                # 6. Configure retriever
                st.session_state.retriever = vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={
                        "k": 4,
                        "fetch_k": 10,
                        "lambda_mult": 0.5
                    }
                )
                
                # Update current file tracker
                st.session_state.current_file = uploaded_file.name
                
                # Clean up the temporary file
                os.remove(tmp_file_path)
                st.success("Database successfully created! You can now start chatting.")
                
            except Exception as e:
                st.error(f"An error occurred during file processing: {e}")
                st.session_state.current_file = None # Reset on failure

else:
    # No file is uploaded. Clear everything.
    st.session_state.retriever = None
    st.session_state.messages = []
    st.session_state.current_file = None
    st.info("Please upload a PDF document in the sidebar to begin.")


# Main chat interface implementation
if st.session_state.retriever is not None:
    
    # Render existing conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle incoming user query
    if query := st.chat_input("Ask a question about your document:"):
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})

        # Generate response using RAG structure
        with st.chat_message("assistant"):
            with st.spinner("Searching document context..."):
                try:
                    # Retrieve context documents
                    docs = st.session_state.retriever.invoke(query)
                    context = "\n\n".join([doc.page_content for doc in docs])

                    # Set up structural prompt template
                    prompt = ChatPromptTemplate.from_messages([
                        (
                            "system",
                            """You are a helpful AI assistant.
                            
                            Use ONLY the provided context to answer the question.
                            
                            If the answer is not present in the context,
                            say: "I could not find the answer in the document." """
                        ),
                        (
                            "human",
                            """Context:
                            {context}
                            
                            Question:
                            {question}"""
                        )
                    ])

                    # Format prompt with retrieved data
                    final_prompt = prompt.invoke({
                        "context": context,
                        "question": query
                    })

                    # Run inference via Mistral LLM
                    llm = ChatMistralAI(model="mistral-medium-3-5")
                    response = llm.invoke(final_prompt)
                    
                    # Display response
                    st.markdown(response.content)
                    st.session_state.messages.append({"role": "assistant", "content": response.content})
                    
                except Exception as e:
                    st.error(f"Error producing response: {e}")