# FinMetrics RAG: Advanced Comparative Financial Intelligence

**FinMetrics RAG** is an industry-grade Retrieval-Augmented Generation (RAG) pipeline engineered to automate the analysis and comparison of complex financial documents. Moving beyond basic vector search, this system implements a multi-stage architecture to deliver high-precision metrics and visual analytics from 10-K filings, annual reports, and equity research.

## 🎯 Project Objective
The primary goal is to solve the "Hallucination and Precision" problem in financial AI. By utilizing **Semantic Chunking** and **Hybrid Ensemble Retrieval**, the system ensures that critical numerical data is captured in its full context, allowing for accurate side-by-side comparisons of different corporate entities.

## 🔭 Project Scope
* **Domain:** Financial Technology (FinTech) & Automated Research.
* **Data Input:** Multi-PDF ingestion (unstructured financial reports).
* **Analytic Focus:** Comparison of Revenue, Net Income, and Total Liabilities.
* **Visualization:** Automated conversion of unstructured text into structured DataFrames and Matplotlib charts.
* **Technology:** Open-source embedding models and state-of-the-art LLMs (Llama 3.3 70B).

---

## 🏗️ Technical Architecture

### 1. Ingestion Layer (Semantic Intelligence)
Unlike standard RAG which splits text at arbitrary character limits, this pipeline uses **Semantic Chunking**. It monitors "breakpoints" in embedding meaning to ensure that financial tables and related disclosure paragraphs stay together.

### 2. Retrieval Layer (Hybrid Ensemble)
To ensure 100% recall, the system implements an **Ensemble Retriever**:
* **Keyword (BM25):** Ensures specific financial terms and exact numbers are indexed.
* **Vector (ChromaDB):** Captures the semantic intent of the query.
* **Weighting:** A 40/60 weighted split optimizes for both factual accuracy and conceptual understanding.

### 3. Refinement Layer (Cross-Encoder Reranking)
The system employs a **Cross-Encoder (`ms-marco-MiniLM`)** pass. This "Expert" model re-evaluates the top candidate chunks, re-ranking them to ensure only the most contextually relevant data is passed to the LLM, effectively eliminating noise.

### 4. Generation & Visualization Layer
The LLM (Llama 3.3 70B via Groq) is prompted to return both a professional analysis and a **structured DATA_JSON block**. This allows the system to bridge the gap between "Generative AI" and "Data Visualization" automatically.

---

## 🛠️ Tech Stack
| Component | Technology |
| :--- | :--- |
| **Orchestration** | LangChain / LangChain-Classic |
| **LLM** | Llama 3.3 70B (Groq) |
| **Embedding Model** | BAAI/bge-small-en-v1.5 |
| **Vector Database** | ChromaDB |
| **Reranker** | Cross-Encoder (Sentence-Transformers) |
| **Visualization** | Pandas, Matplotlib |

---

## 🚀 Getting Started
Quick Start:

1. Set your GROQ_API_KEY in your environment variables.
2. Initialize the ComparativeFinancialRAG class.
3. Upload two financial PDFs.
4. Run the compare() method to generate a full report and comparison chart.

## 📈 Expected Output
a) Professional Summary: A detailed breakdown of growth trends, risk factors, and liability standing.
b) Audit Trail: Verification of the exact document snippets used for the calculation.
c) Side-by-Side Chart: A generated bar chart comparing the primary financial metrics of both companies.


### Installation
```bash
pip install -r requirements.txt
