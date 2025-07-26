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
## Answers to Specific Questions

### 1. What method or library did you use to extract the text, and why? Did you face any formatting challenges with the PDF content?

* **Method/Library:** Initially, `PyPDF2` was used for direct text extraction. However, it failed to extract usable text from the `HSC26-Bangla1st-Paper.pdf` file, resulting in heavily garbled characters. Therefore, the method was switched to use **`pdf2image`** (to convert PDF pages into images) combined with **`pytesseract`** (to perform Optical Character Recognition or OCR on these images). This relies on the external `tesseract-ocr` and `poppler-utils` system packages.
* **Why OCR:** OCR was chosen because the target PDF, despite appearing text-based, likely contained embedded fonts or formatting that `PyPDF2` couldn't correctly interpret, or was partially scanned. OCR bypasses these issues by "reading" the text from the visual representation of the page.
* **Formatting Challenges:** Yes, we faced **significant formatting challenges**.
    * **Garbled Text:** The initial `PyPDF2` extraction produced unreadable, corrupted Bengali characters.
    * **OCR Artifacts:** Even after switching to OCR, the raw extracted text still contained a lot of noise: page numbers, headers/footers (`10 MINUTE SCHOOL`, `HSC 26 অনলাইন ব্যাচ`), board exam tags (`[ঢা. বো. '২২]`), question numbers (`১।`, `(ক)`), answer prefixes (`উত্তর:`), explanation headers (`ব্যাখ্যা:`), and various symbols or fragmented words from OCR errors.
    * **Solution:** This necessitated developing an **extremely aggressive `clean_text` function** with numerous regular expressions to systematically strip out these artifacts and normalize whitespace. This was an iterative process, as new noise patterns emerged.

### 2. What chunking strategy did you choose? Why do you think it works well for semantic retrieval?

* **Strategy:** We used a **character-limit based chunking strategy with generous overlap** implemented by `langchain`'s **`RecursiveCharacterTextSplitter`**.
    * `CHUNK_SIZE = 150` characters.
    * `CHUNK_OVERLAP = 100` characters.
* **Why it works well for semantic retrieval:**
    * **Semantic Coherence:** `RecursiveCharacterTextSplitter` attempts to split text at natural language boundaries (like `\n\n`, `\n`, `।`, `.`, `?`, `!`) before resorting to brute-force character limits. This helps ensure that individual chunks represent more complete thoughts or sentences, which are better units for semantic understanding.
    * **Context Preservation (Overlap):** The generous overlap (100 characters for a 150-character chunk) is crucial. It ensures that if a key piece of information or a sentence spans a chunk boundary, it's likely to be fully contained within multiple adjacent chunks. This reduces the risk of splitting critical context and makes it more likely that the embedding model can correctly represent its meaning.
    * **Manageable Size:** Smaller chunks (like 150 chars) are less prone to "context dilution" where a relevant sentence's meaning gets lost among too much irrelevant text within a large chunk. This makes the embedding more focused.
* **Challenges:** Initially, even with aggressive cleaning, the filter to skip "question-like" chunks was too broad, leading to very few chunks. Fine-tuning the balance between `CHUNK_SIZE`, `CHUNK_OVERLAP`, and the filtering heuristic was a significant challenge.

### 3. What embedding model did you use? Why did you choose it? How does it capture the meaning of the text?

* **Model Used:** **`sentence-transformers/LaBSE`** (Language-agnostic BERT Sentence Embeddings).
* **Why it was chosen:**
    * **Multilingual Capability:** The project requires understanding both English and Bengali. `LaBSE` is specifically designed to produce embeddings where semantically similar sentences from *different languages* are mapped close together in the vector space. This is vital for comparing English queries to Bengali documents (and vice-versa).
    * **Performance:** It's a robust and well-regarded model for multilingual semantic similarity tasks.
    * **Dimensionality:** It outputs 768-dimensional embeddings, which is a common and effective dimensionality for capturing rich semantic meaning.
* **How it captures meaning:** `LaBSE` is a transformer-based model (like BERT) that has been pre-trained on a massive amount of text data across many languages. It's often fine-tuned using tasks like Natural Language Inference (NLI) or Semantic Textual Similarity (STS). During this training, it learns to understand the contextual relationships between words and sentences. It produces dense numerical vectors where sentences with similar meanings (regardless of specific wording or language) are geometrically "close" in the high-dimensional vector space. For example, a Bengali sentence and its English translation would have very similar `LaBSE` embeddings.

### 4. How are you comparing the query with your stored chunks? Why did you choose this similarity method and storage setup?

* **Comparison Method:** We use **Hybrid Search**, combining two primary methods:
    1.  **Vector Search (Semantic Similarity):** Uses **Cosine Similarity** on the embeddings.
    2.  **Lexical Search (Keyword Matching):** Uses the **BM25Okapi algorithm**.
* **Why Hybrid Search:** Pure semantic search (cosine similarity on embeddings) alone often struggled with very specific factual recall. It might retrieve chunks that are "about" the topic but lack the exact answer, or it might retrieve questions from a question bank that share keywords with the query but don't contain the answer. Hybrid search addresses these limitations:
    * **Vector Search (Cosine Similarity):** Excellent for understanding the *meaning* and *intent* of the query, finding synonyms, and semantically related concepts.
    * **BM25 (Lexical Search):** Excellent for finding exact keyword matches. This is crucial for retrieving precise facts, names, or numbers that might not have strong semantic uniqueness otherwise.
    * **Blending:** By combining results and scores (with a custom weighting, giving a significant boost to BM25 hits), the system balances conceptual understanding with exact recall, leading to much more accurate retrieval for diverse query types.
* **Storage Setup:** **ChromaDB** (a persistent local vector database).
    * **Choice Reason:** ChromaDB was chosen for its ease of setup, lightweight nature, and simplicity of use in a Google Colab or local development environment. It manages the indexing and storage of vector embeddings efficiently without requiring complex external server configurations or managed cloud services. It abstracts away the low-level details of vector indexing (like HNSW).

### 5. How do you ensure that the question and the document chunks are compared meaningfully? What would happen if the query is vague or missing context?

* **Ensuring Meaningful Comparison:**
    * **Consistent Embedding Model:** The *same* `LaBSE` embedding model is used to convert both the document chunks (during indexing) and the user queries (during retrieval) into vectors. This ensures they exist in the same high-dimensional semantic space, making their comparison meaningful.
    * **Aggressive Text Cleaning:** The `clean_text` function removes noise and artifacts from both the PDF content (before chunking) and implicitly from the query (as `word_tokenize` for BM25 works on clean input). This reduces irrelevant signals in the embeddings.
    * **Effective Chunking:** The `RecursiveCharacterTextSplitter` creates semantically coherent chunks, making it more likely that a chunk's embedding truly represents a single idea.
    * **Hybrid Search:** By combining semantic (vector) and lexical (BM25) comparison, the system ensures queries are compared meaningfully both by their underlying meaning and by direct keyword overlap.
* **If the query is vague or missing context:**
    * **Retrieval:** A vague query will result in an embedding that is not highly specific. The vector search might retrieve a broader, less precise set of chunks (e.g., general topics from the document). BM25 might still pick up common keywords, but the combined score might not pinpoint a specific answer.
    * **LLM Response:** The RAG system is explicitly instructed to answer "ONLY on the provided context." If the retrieved chunks for a vague query do not contain the specific answer, the LLM will respond with "I don't know" or "cannot find the answer based on the provided information." This is crucial to prevent hallucinations.
    * **Short-Term Memory:** The system maintains short-term conversational history. If a follow-up query is vague but refers to a previous turn (e.g., "What about her age?"), the `Conversation History` included in the LLM's prompt helps the LLM understand the current query's context. However, this history doesn't inherently improve the *retrieval* of new, long-term memory chunks unless the history is also used to modify the retrieval query itself (which is a more advanced technique not fully implemented here).

### 6. Do the results seem relevant? If not, what might improve them?

* **Relevance of Results:** The relevance of results was **initially very poor** for specific factual queries (like Kalyani's age) due to severe PDF extraction issues and later, aggressive filtering and limitations of purely semantic search. For some queries like "কল্যানীর পিতা কে?", the results became highly relevant after improvements.
* **Improvements Made Throughout the Project:**
    * **OCR Implementation:** Switching from `PyPDF2` to `pdf2image` + `pytesseract` dramatically improved raw text extraction quality from the problematic PDF.
    * **Aggressive `clean_text`:** Iterative refinement of regex patterns was crucial to remove persistent OCR artifacts, headers, footers, and exam-related noise.
    * **Refined Chunking Parameters:** Experimenting with `CHUNK_SIZE` (tuned to 150) and `CHUNK_OVERLAP` (tuned to 100) improved the integrity of semantic units.
    * **Heuristic Filtering of Question Chunks:** This initially caused low chunk counts but, when properly tuned (or temporarily disabled for full data indexing), helped reduce noisy question-like chunks from entering the database.
    * **`LaBSE` Embedding Model:** Switching from `MiniLM` to `LaBSE` enhanced multilingual semantic understanding.
    * **Hybrid Search (Vector + BM25):** Implementing BM25 alongside vector search, with a significant weight boost for BM25 hits, was the most impactful improvement for precise factual recall, balancing semantic meaning with keyword matching.
* **Further Potential Improvements (Beyond Current Scope):**
    * **Sophisticated Re-ranking:** After retrieving N chunks via hybrid search, use a more powerful model (like a cross-encoder from `sentence-transformers`) to re-rank these N chunks based on their direct relevance to the query.
    * **Query Expansion/Rewriting:** Use an LLM to rephrase or generate synonyms for the user's query before performing retrieval, increasing the chances of finding relevant matches.
    * **Context-Aware Retrieval:** Modify the retrieval query itself by incorporating parts of the chat history, not just the current question.
    * **Domain-Specific Fine-Tuning:** Fine-tune the `LaBSE` model on a large corpus of Bengali academic text or similar content to make its embeddings even more specialized for the document's domain.
    * **Advanced PDF Parsing:** For PDFs with very complex layouts (multi-column, tables, figures mixed with text), dedicated PDF parsing libraries (beyond basic text extraction or OCR) that understand document structure could improve chunk quality significantly.
    * **Larger/Diverse Document Corpus:** For broader questions, a larger and more diverse knowledge base would naturally lead to more relevant answers.
