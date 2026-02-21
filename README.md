# 🔍 Document-Grounded RAG System (LLM + Vector Search)

A **production-quality Retrieval-Augmented Generation (RAG)** system that answers questions **strictly from a provided document**, with explicit hallucination prevention and source citations.

Built to demonstrate **real-world LLM system design**, not demos.

---

## ⭐ Why This Project Matters

- Prevents **LLM hallucinations** using retrieval gating and strict prompting  
- Implements a **complete RAG pipeline** end-to-end  
- Produces **auditable answers** with page-level citations  
# Document-Grounded RAG CLI

A concise, production-minded Retrieval-Augmented Generation (RAG) CLI that answers questions strictly from a provided PDF and refuses unsupported or unsafe requests. It is built for predictable, grounded responses with clear guardrails.

## Features
- PDF ingestion, chunking, embedding, and retrieval with a persistent vector store
- Deterministic answers (temperature 0) using retrieved context only
- Guardrails for off-topic requests, prompt injection attempts, and PII detection
- Retrieval gating with a relevance threshold and explicit refusal behavior
- Faithfulness evaluation (YES/NO/N/A) and structured logging

## Tech Stack
- Python 3.12
- Google Gemini (ChatGoogleGenerativeAI)
- Jina Embeddings (jina-embeddings-v3)
- LangChain + Chroma
- PyPDF + python-dotenv

## How to Run
1. Add a PDF to `data/`
2. Set `JINA_API_KEY` and `GOOGLE_API_KEY` in `.env`
3. Run:

```bash
python rag_cli.py
```

Results can be saved to `output/results.txt` in a required structured format when logging is enabled in the CLI.

## Project Structure
```
.
├── data/           # Input PDF document
├── chroma_db/      # Persistent vector database
├── output/         # Logged results
├── rag_cli.py      # Main application
├── requirements.txt
└── README.md
```

## Security and Robustness
- Domain-limited answers (Nova Scotia driving/road rules only)
- Retrieval gating with a similarity threshold
- Prompt hardening with strict context-only instructions

## Prompt Injection Defenses (Required)
- Regex/pattern-based blocking of prompt injection attempts (e.g., "ignore previous instructions", "system prompt", "### SYSTEM"); blocked requests return `POLICY_BLOCK`.
- Instruction/data separation using `<retrieved_context>...</retrieved_context>` and an explicit "use only context" rule.
- Strict retrieval gating + refusal: the model answers only when relevant context is retrieved; otherwise it refuses.

## Evaluation Metric
- **Faithfulness (YES/NO/N/A):** checks whether the answer is supported by retrieved context.
- **Why it matters:** it directly measures grounding and hallucination risk and is simple, auditable, and appropriate for RAG systems.

## Findings from Security and Evaluation Test Queries
- Injection attempts are blocked with `POLICY_BLOCK`.
- Off-topic questions are refused with a domain-limited message.
- PII is detected and stripped; the query is still processed (non-blocking).
- Empty queries are handled with a clear error and logged when logging is enabled.
- Borderline questions may return "Not found in the document" depending on the retrieval threshold.
Enterprise RAG prototypes

# 📄 License

Released for educational and portfolio use.

# 👋 Recruiter Note

This project focuses on how LLM systems should behave in production:

predictable, grounded, and explainable — not just “cool outputs”.
