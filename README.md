🔍 Document-Grounded RAG System (LLM + Vector Search)

A production-quality Retrieval-Augmented Generation (RAG) system that answers questions strictly from a provided document, with explicit hallucination prevention and source citations.

Built to demonstrate real-world LLM system design, not demos.

⭐ Why This Project Matters

Prevents LLM hallucinations using retrieval gating + strict prompting

Implements a complete RAG pipeline end-to-end

Produces auditable answers with page-level citations

Designed with reproducibility and correctness in mind

Uses modern LangChain architecture (no deprecated APIs)

🧠 What It Does (High Level)

Ingests a PDF document

Converts it into semantic embeddings

Stores embeddings in a persistent vector database

Retrieves only relevant context for each question

Generates answers only from retrieved content

Refuses unsupported questions with a clear response

⚙️ Tech Stack

Language: Python 3.12

LLM: Google Gemini

Embeddings: Jina AI (jina-embeddings-v3)

Framework: LangChain (Runnable-based)

Vector DB: ChromaDB

Document Parsing: PyPDF

Environment Management: python-dotenv

🧩 System Architecture
PDF → Chunking → Embeddings → Vector Store
                         ↓
                    Semantic Search
                         ↓
                   Context Filtering
                         ↓
                    LLM Generation
                         ↓
                 Answer + Citations

🛡️ Hallucination Control (Key Design Focus)

This system does not guess.

Hallucination prevention is enforced using:

Similarity score thresholding (weak matches are discarded)

Context-only prompting (no external knowledge allowed)

Exact-response enforcement for unsupported questions

Single-source retrieval (no mixed context)

If the answer is not found, the system responds:

Not found in the document.

💬 Example Interaction
Question> What is Crosswalk guards?

Answer:
Crosswalk guards direct the movement of children along or across highways going to or from school.

Sources:
[1] page=5 | Crosswalk guards direct the movement of children along or across highways...


Unsupported question:

Question> What is the capital of France?

Answer:
Not found in the document.

📁 Project Structure
.
├── data/           # Input PDF document
├── chroma_db/      # Persistent vector database
├── output/         # Saved evaluation results
├── rag_cli.py      # Main application
├── requirements.txt
└── README.md

🚀 How to Run
python rag_cli.py


The system:

Builds or loads the vector database

Runs predefined evaluation queries

Saves results to output/results.txt

Launches an interactive CLI

🔍 Engineering Highlights

Relevance-gated retrieval using similarity scores

Persistent embeddings with automatic rebuild on document change

Deterministic generation (temperature = 0)

Single-pass retrieval (no redundant queries)

Clean separation of ingestion, retrieval, and generation logic

📌 Use Cases

Internal knowledge assistants

Policy / compliance Q&A

Technical documentation search

Regulated or high-trust LLM systems

Enterprise RAG prototypes

📄 License

Released for educational and portfolio use.

👋 Recruiter Note

This project focuses on how LLMs should behave in real systems:
predictable, grounded, and explainable — not just “cool outputs”.