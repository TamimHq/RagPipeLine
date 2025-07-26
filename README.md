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
    git clone <repository_url_here>
    cd <repository_name>
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
