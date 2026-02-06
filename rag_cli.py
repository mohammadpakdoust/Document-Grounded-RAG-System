import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import JinaEmbeddings

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
CHROMA_DIR = BASE_DIR / "chroma_db"

# Assignment required test queries
TEST_QUERIES = [
    "What is Crosswalk guards?",
    "What to do if moving through an intersection with a green signal?",
    "What to do when approached by an emergency vehicle?",
]


def die(msg: str, code: int = 1):
    print(msg)
    raise SystemExit(code)


def find_pdf(data_dir: Path) -> Path:
    """Return the first PDF in data_dir. Exit cleanly if missing."""
    if not data_dir.exists():
        die("❌ data/ folder not found. Please create data/ and add the PDF.")

    pdfs = sorted(data_dir.glob("*.pdf"))
    if not pdfs:
        die("❌ No PDF found in data/. Please add DH-Chapter2.pdf (or any .pdf) to data/.")

    return pdfs[0]


def ensure_env():
    load_dotenv()
    if not os.getenv("JINA_API_KEY"):
        die("❌ Missing JINA_API_KEY in .env")
    if not os.getenv("GOOGLE_API_KEY"):
        die("❌ Missing GOOGLE_API_KEY in .env")


def build_or_load_vectorstore(pdf_path: Path) -> Chroma:
    """Build Chroma DB if empty; otherwise load existing DB."""
    ensure_env()

    embeddings = JinaEmbeddings(
        jina_api_key=os.getenv("JINA_API_KEY"),
        model_name="jina-embeddings-v3",
    )

    # If Chroma DB exists and has content, load it
    if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()):
        vs = Chroma(persist_directory=str(CHROMA_DIR), embedding_function=embeddings)
        # If somehow empty, rebuild
        try:
            if vs._collection.count() > 0:
                return vs
        except Exception:
            pass

    # Build from scratch
    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()
    if not docs:
        die(f"❌ Loaded 0 pages from PDF: {pdf_path}. Is the file valid?")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    if not chunks:
        die("❌ Chunking produced 0 chunks. Cannot build vector store.")

    vs = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name="dh_chapter2",
    )
    return vs


def make_qa_chain(vectorstore: Chroma):
    ensure_env()

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

    template = """Use ONLY the following context to answer the question.
If the answer is not explicitly in the context, reply exactly:
Not found in the document.

Context:
{context}

Question: {question}

Answer:"""
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    RELEVANCE_THRESHOLD = 0.30  # tune 0.25–0.40 if needed

    class QAChain:
        def __init__(self, vectorstore, llm, prompt):
            self.vectorstore = vectorstore
            self.llm = llm
            self.prompt = prompt

        def invoke(self, inputs):
            question = inputs.get("query", "").strip()
            if not question:
                return {"result": "Not found in the document.", "source_documents": []}

            # Retrieve ONCE with scores
            docs_scores = self.vectorstore.similarity_search_with_relevance_scores(question, k=4)

            # Keep only relevant docs
            source_docs = [d for d, s in docs_scores if s >= RELEVANCE_THRESHOLD]

            if not source_docs:
                return {"result": "Not found in the document.", "source_documents": []}

            context = format_docs(source_docs)
            msg = self.prompt.format(context=context, question=question)
            answer = self.llm.invoke(msg).content.strip()

            # Hard guard: if model ignores instruction, enforce exact match
            normalized = answer.strip().lower()
            if normalized in {"not found in the document.", "not found in the document"}:
                return {"result": "Not found in the document.", "source_documents": []}

            return {"result": answer, "source_documents": source_docs}

    return QAChain(vectorstore, llm, prompt)


def format_sources(source_docs) -> str:
    lines = []
    for i, d in enumerate(source_docs, start=1):
        page = d.metadata.get("page", "unknown")
        snippet = d.page_content.replace("\n", " ").strip()
        snippet = (snippet[:220] + "...") if len(snippet) > 220 else snippet
        lines.append(f"[{i}] page={page} | {snippet}")
    return "\n".join(lines)


def answer_question(qa, question: str):
    result = qa.invoke({"query": question})
    answer = (result.get("result") or "").strip()
    sources = result.get("source_documents") or []

    if not sources or not answer:
        return "Not found in the document.", ""

    return answer, format_sources(sources)


def write_required_results(qa):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "results.txt"

    lines = []
    for q in TEST_QUERIES:
        ans, src = answer_question(qa, q)
        lines.append(f"Q: {q}\nA: {ans}\n")
        if src:
            lines.append(f"SOURCES:\n{src}\n")
        lines.append("-" * 60 + "\n")

    out_path.write_text("".join(lines), encoding="utf-8")
    print(f"Saved: {out_path}")


def cli_loop(qa):
    print("\nRAG CLI ready. Type a question, or type 'exit' to quit.\n")
    while True:
        q = input("Question> ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break

        ans, src = answer_question(qa, q)
        print("\nAnswer:\n", ans)
        if src:
            print("\nSources:\n", src)
        print()


def main():
    pdf_path = find_pdf(DATA_DIR)
    
    # Safety: ensure embeddings match the current PDF
    import shutil
    sig_path = CHROMA_DIR / "pdf_signature.txt"
    current_sig = f"{pdf_path.name}|{pdf_path.stat().st_size}"

    if sig_path.exists():
        old_sig = sig_path.read_text(encoding="utf-8").strip()
    else:
        old_sig = ""

    if old_sig != current_sig:
        # PDF changed → rebuild DB
        if CHROMA_DIR.exists():
            for item in CHROMA_DIR.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
    
    vs = build_or_load_vectorstore(pdf_path)
    
    # Write signature to mark this DB as built from current PDF
    CHROMA_DIR.mkdir(exist_ok=True)
    sig_path.write_text(current_sig, encoding="utf-8")
    
    qa = make_qa_chain(vs)

    # Assignment requirement: save answers for required test queries
    write_required_results(qa)

    # Interactive Q&A
    cli_loop(qa)


if __name__ == "__main__":
    main()
