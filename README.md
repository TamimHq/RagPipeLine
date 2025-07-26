# RagPipeLine
## Setup Guide

Follow these steps to set up and run the project locally or in Google Colab.

### Prerequisites

* **Python 3.9+**
* **Git** (for cloning the repository)
* **Google Gemini API Key:** Obtain one from [Google AI Studio](https://aistudio.google.com/app/apikey).
* **MongoDB Atlas Cluster (Optional, if switching from ChromaDB):** If you decide to use MongoDB Atlas Vector Search, refer to previous instructions for setup, database/collection creation, and **Vector Search Index creation (crucially, matching embedding dimensions, e.g., 768 for LaBSE)**. For this README, ChromaDB is the default.
* **PDF Document:** Have your PDF file (e.g., `HSC26-Bangla1st-Paper.pdf`) ready.

### Local Setup Steps

1.  **Clone the Repository:**
    ```bash
    git clone <https://github.com/TamimHq/RagPipeLine.git>
    cd <RagPipeLine>
    ```
2.  **Create and Activate a Virtual Environment:**
    ```bash
    python3 -m venv venv
    # On macOS/Linux:
    source venv/bin/activate
    # On Windows:
    # venv\Scripts\activate.bat
    ```
3.  **Install Dependencies:**
    ```bash
    pip install PyPDF2 sentence-transformers langchain chromadb python-dotenv google-generativeai rank_bm25 pytesseract pdf2image
    # Install system-level OCR tools (for Linux/WSL):
    sudo apt-get install poppler-utils tesseract-ocr tesseract-ocr-ben
    ```
4.  **Create `.env` File:**
    In the root directory of your project, create a file named `.env` and add your API key:
    ```env
    GEMINI_API_KEY="YOUR_GEMINI_API_KEY_HERE"
    ```
5.  **Place Your PDF:** Put your PDF file (e.g., `HSC26-Bangla1st-Paper.pdf`) in the same directory.
### Running the RAG System

The main interaction is via a Command Line Interface (CLI).

1.  **Run the Script:**
    ```bash
    python rag_system.py
    ```
2.  **Follow Prompts:**
    * The system will prompt you to **upload a PDF**. Provide the path (e.g., `HSC26-Bangla1st-Paper.pdf`).
    * If the knowledge base is not empty, it will ask if you want to **add to existing data or clear and load new data**. Choose `2` to clear and rebuild for a fresh start with the chosen PDF.
    * The PDF will be processed (OCR, chunking, embedding, storage in `chroma_db` folder).
    * Once "Knowledge Base is ready" is displayed, you can **type your questions** in English or Bengali.
    * Type `exit` to quit the Q&A session.

---

## Used Tools, Libraries, Packages

* **`Python 3.11+`**: Programming language.
* **`PyPDF2`**: For initial PDF text extraction (fallback).
* **`pdf2image`**: Converts PDF pages into images for OCR.
* **`pytesseract`**: Python wrapper for Tesseract OCR engine, crucial for extracting text from image-based PDFs.
* **`Pillow (PIL)`**: Image processing library, used by `pdf2image` and `pytesseract`.
* **`tesseract-ocr` (system package)**: The OCR engine itself.
* **`tesseract-ocr-ben` (system package)**: Bengali language pack for Tesseract.
* **`poppler-utils` (system package)**: Command-line utilities for PDFs, used by `pdf2image`.
* **`sentence-transformers`**: Generates high-quality text embeddings. Specifically, `LaBSE` (Language-agnostic BERT Sentence Embeddings) is used for its multilingual capabilities, outputting 768-dimensional vectors.
* **`langchain` (`RecursiveCharacterTextSplitter`)**: Provides robust text chunking functionalities.
* **`chromadb`**: The vector database used for storing and efficiently searching text embeddings.
* **`rank_bm25`**: Implements the BM25 algorithm for lexical (keyword-based) search, part of the Hybrid Retrieval.
* **`google-generativeai`**: Official Python client library for interacting with the Google Gemini API.
* **`python-dotenv`**: Manages environment variables (like API keys) securely from a `.env` file.
* **`nltk`**: Natural Language Toolkit, specifically `nltk.tokenize.word_tokenize` for text tokenization (for BM25).

