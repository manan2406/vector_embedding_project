# Vector Embedding Project

Welcome to the **Vector Embedding Project**, an innovative tool designed to explore the raw materials (represented as colors) associated with various fabrics through natural language queries. This project combines cutting-edge AI technologies—OpenAI embeddings, Pinecone vector storage, and LangChain’s RetrievalQA chain—with a user-friendly Streamlit interface to deliver an interactive experience for fabric enthusiasts, researchers, or anyone curious about material properties.

---

## Project Overview
The **Vector Embedding Project** is a Python-based application that answers questions about fabric raw materials using advanced natural language processing (NLP) and vector search techniques. Imagine asking, "What raw material is used for cashmere?" and receiving a clear, concise answer like, "The raw materials (colors) associated with cashmere are beige and gray." This project achieves that by:
- Processing a dataset of fabric names and their associated colors.
- Converting this data into vector embeddings using OpenAI’s `text-embedding-ada-002` model.
- Storing these embeddings in Pinecone for fast, scalable retrieval.
- Using LangChain’s RetrievalQA chain with `gpt-3.5-turbo` to generate natural language responses.

The result is a web-based tool, powered by Streamlit, where users can input queries and get instant answers, all wrapped in a clean, intuitive interface.

---

## Key Features
- **Interactive Query System:** Ask questions about fabrics in plain English and get meaningful responses.
- **Semantic Search:** Leverages vector embeddings for accurate, context-aware answers.
- **Scalable Storage:** Uses Pinecone to store and retrieve embeddings efficiently, even for large datasets.
- **Elegant UI:** Built with Streamlit, featuring a light-themed design with a responsive input form and styled response display.
- **Robust Logging:** Tracks operations and errors in a log file for easy debugging.
- **Secure Configuration:** Stores API keys in a `.env` file, excluded from version control for safety.

---

## How It Works
1. **Data Preparation:** A script (`generate_data.py`) creates `fabric_data.csv`, containing fabric names and their raw material colors.
2. **Embedding Generation:** OpenAI’s embedding model converts fabric names into numerical vectors, capturing their semantic meaning.
3. **Vector Storage:** These vectors, along with metadata (fabric and color), are stored in a Pinecone index.
4. **Query Processing:** User inputs are processed by a RetrievalQA chain, which retrieves relevant embeddings from Pinecone and generates answers using GPT-3.5-turbo.
5. **UI Interaction:** Streamlit provides a front-end where users type queries and view responses.

---

## Prerequisites
Before setting up the project, ensure you have:
- **Python 3.8 or higher:** Download from [python.org](https://www.python.org/downloads/).
- **Git:** Install from [git-scm.com](https://git-scm.com/) for version control.
- **API Keys:**
  - **OpenAI API Key:** Sign up at [platform.openai.com](https://platform.openai.com/) and generate a key.
  - **Pinecone API Key:** Register at [pinecone.io](https://www.pinecone.io/) and obtain a key.
- **Internet Connection:** Required for API calls and dependency installation.

---