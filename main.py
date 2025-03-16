# main.py
import os
import pandas as pd
import time
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import streamlit as st

# --- Configuration and Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("vector_embedding_project.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file with explicit path
load_dotenv(dotenv_path=r"C:\Users\Manan\OneDrive\Documents\vector_embedding_project\.env")

# Configuration
PINECONE_INDEX_NAME = "product-raw-material-index"
PINECONE_DIMENSION = 1536
PINECONE_METRIC = "cosine"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"
EMBEDDING_MODEL = "text-embedding-ada-002"
LLM_MODEL = "gpt-3.5-turbo"
BATCH_SIZE = 100
RETRIEVAL_K = 3

# Validate environment variables
required_env_vars = ["OPENAI_API_KEY", "PINECONE_API_KEY"]
for var in required_env_vars:
    if not os.getenv(var):
        logger.error(f"Environment variable {var} is not set.")
        raise ValueError(f"Environment variable {var} is not set.")

# --- Helper Functions ---
def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    """Load and preprocess the CSV data."""
    try:
        logger.info(f"Loading data from {file_path}")
        data = pd.read_csv(file_path)
        logger.info(f"Loaded {len(data)} rows from {file_path}")
        
        data = data.dropna(subset=["product", "color"])
        data["product"] = data["product"].str.lower().str.strip()
        data["color"] = data["color"].str.lower().str.strip()
        
        logger.info("Data preprocessing completed")
        return data
    except FileNotFoundError:
        logger.error(f"File {file_path} not found. Please run generate_data.py first.")
        raise
    except Exception as e:
        logger.error(f"Error loading/preprocessing data: {e}")
        raise

def generate_embeddings(data: pd.DataFrame, embeddings: OpenAIEmbeddings) -> pd.DataFrame:
    """Generate embeddings for product and color texts in batches."""
    try:
        product_texts = data["product"].tolist()
        color_texts = data["color"].tolist()
        
        logger.info(f"Generating embeddings for {len(product_texts)} products")
        
        product_embeddings = []
        color_embeddings = []
        for i in range(0, len(product_texts), BATCH_SIZE):
            batch_products = product_texts[i:i + BATCH_SIZE]
            batch_colors = color_texts[i:i + BATCH_SIZE]
            product_batch = embeddings.embed_documents(batch_products)
            color_batch = embeddings.embed_documents(batch_colors)
            product_embeddings.extend(product_batch)
            color_embeddings.extend(color_batch)
            logger.info(f"Processed embeddings for batch {i // BATCH_SIZE + 1}")
        
        data["product_embedding"] = product_embeddings
        data["color_embedding"] = color_embeddings
        logger.info("Embeddings generated successfully")
        return data
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise

def setup_pinecone_index(api_key: str) -> Pinecone:
    """Initialize Pinecone client and ensure the index exists."""
    try:
        pinecone = Pinecone(api_key=api_key)
        existing_indexes = [index["name"] for index in pinecone.list_indexes()]
        
        if PINECONE_INDEX_NAME not in existing_indexes:
            logger.info(f"Creating Pinecone index '{PINECONE_INDEX_NAME}'")
            pinecone.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=PINECONE_DIMENSION,
                metric=PINECONE_METRIC,
                spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION)
            )
        
        while not pinecone.describe_index(PINECONE_INDEX_NAME).status["ready"]:
            logger.info("Waiting for Pinecone index to be ready...")
            time.sleep(1)
        
        logger.info(f"Pinecone index '{PINECONE_INDEX_NAME}' is ready")
        return pinecone
    except Exception as e:
        logger.error(f"Error setting up Pinecone index: {e}")
        raise

def store_embeddings_in_pinecone(data: pd.DataFrame, embeddings: OpenAIEmbeddings) -> PineconeVectorStore:
    """Store embeddings in Pinecone."""
    try:
        documents = data["product"].tolist()
        metadatas = [{"product": prod, "color": col} for prod, col in zip(data["product"], data["color"])]
        
        logger.info(f"Storing {len(documents)} embeddings in Pinecone")
        vector_store = PineconeVectorStore.from_texts(
            texts=documents,
            embedding=embeddings,
            metadatas=metadatas,
            index_name=PINECONE_INDEX_NAME
        )
        logger.info("Embeddings stored in Pinecone successfully")
        return vector_store
    except Exception as e:
        logger.error(f"Error storing embeddings in Pinecone: {e}")
        raise

def setup_retrieval_qa(vector_store: PineconeVectorStore) -> RetrievalQA:
    """Set up the RetrievalQA chain."""
    try:
        llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
        retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVAL_K})
        
        prompt_template = """
        You are a helpful assistant. The user asks about the raw material (color) associated with a product (fabric). Use the provided context to answer.
        
        Context: {context}
        
        User Prompt: {question}
        
        Based on the context, extract the product and its associated colors, then answer in a natural sentence:
        "The raw materials (colors) associated with [product] are [colors]."
        If no raw materials are found, respond: "I couldn't find the raw material for that product."
        """
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        logger.info("RetrievalQA chain set up successfully")
        return qa_chain
    except Exception as e:
        logger.error(f"Error setting up RetrievalQA chain: {e}")
        raise

def retrieve_raw_material(qa_chain: RetrievalQA, prompt: str) -> str:
    """Retrieve raw material (color) for a given prompt."""
    try:
        result = qa_chain({"query": prompt})
        return result["result"]
    except Exception as e:
        logger.error(f"Error processing prompt '{prompt}': {e}")
        return f"Error processing prompt: {e}"

# --- Streamlit UI ---
def initialize_pipeline():
    """Initialize the pipeline and return the QA chain."""
    try:
        # Load and preprocess data
        data = load_and_preprocess_data("fabric_data.csv")
        
        # Generate embeddings
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        data = generate_embeddings(data, embeddings)
        
        # Set up Pinecone and store embeddings
        pinecone = setup_pinecone_index(os.getenv("PINECONE_API_KEY"))
        vector_store = store_embeddings_in_pinecone(data, embeddings)
        
        # Set up RetrievalQA chain
        qa_chain = setup_retrieval_qa(vector_store)
        
        return qa_chain
    except Exception as e:
        logger.error(f"Pipeline initialization failed: {e}")
        st.error(f"Failed to initialize the pipeline: {e}")
        return None

def main():
    """Main function to run the Streamlit UI."""
    st.set_page_config(page_title="Fabric Raw Material Finder", page_icon="🧵", layout="wide")
    
    # Custom CSS for styling
    st.markdown("""
        <style>
        .main { background-color: #f5f5f5; }
        .stTextInput > div > div > input { border-radius: 8px; padding: 10px; }
        .stButton > button { background-color: #4CAF50; color: white; border-radius: 8px; }
        .response-box { background-color: #ffffff; padding: 15px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        </style>
    """, unsafe_allow_html=True)
    
    # Title and description
    st.title("🧵 Fabric Raw Material Finder")
    st.markdown("Enter a question about a fabric (e.g., 'What raw material is used for cashmere?') to find the associated colors.")
    
    # Initialize the pipeline once and cache it
    if "qa_chain" not in st.session_state:
        with st.spinner("Initializing the pipeline... This may take a moment."):
            st.session_state.qa_chain = initialize_pipeline()
    
    # Input form
    with st.form(key="query_form", clear_on_submit=True):
        prompt = st.text_input("Ask about a fabric:", placeholder="e.g., What color is used for bamboo fabric?")
        submit_button = st.form_submit_button(label="Get Answer")
    
    # Process the prompt and display the response
    if submit_button and prompt:
        if st.session_state.qa_chain:
            with st.spinner("Fetching the answer..."):
                response = retrieve_raw_material(st.session_state.qa_chain, prompt)
                st.session_state.last_response = response  # Store the last response
        else:
            st.error("Pipeline not initialized. Please check the logs for errors.")
    
    # Display the response if available
    if "last_response" in st.session_state:
        st.markdown("### Answer")
        st.markdown(f"<div class='response-box'>{st.session_state.last_response}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()