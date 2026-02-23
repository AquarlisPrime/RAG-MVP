!pip install -qU langchain langchain-community langchain-groq langchain-experimental \
                sentence-transformers chromadb rank_bm25 langchain-huggingface pypdf
!pip install -qU langchain-classic

import os
import json
import re
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from google.colab import files

# Imports for stability
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

# Import for EnsembleRetriever
try:
    from langchain_classic.retrievers import EnsembleRetriever
except ImportError:
    try:
        from langchain.retrievers.ensemble import EnsembleRetriever
    except ImportError:
        from langchain_community.retrievers import EnsembleRetriever

warnings.filterwarnings("ignore")

# --- CONFIG ---
# Groq API Key
os.environ["GROQ_API_KEY"] = "YOUR_GORQ_API_HERE"

class ComparativeFinancialRAG:
    def __init__(self):
        print("--- Initializing Advanced Financial AI ---")
        # Local Embedding Model
        self.embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        
        # Cross-Encoder for precision re-ranking 
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # Groq Llama 3.3 70B 
        self.llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
        
        self.all_chunks = []
        self.vectorstore = None
        self.retriever = None

    def add_document(self, pdf_path: str, label: str):
        print(f"---Processing {label}: {pdf_path} ---")
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        
        # Semantic Chunking breaks text where the meaning changes
        text_splitter = SemanticChunker(self.embed_model, breakpoint_threshold_type="percentile")
        chunks = text_splitter.split_documents(pages)
        
        for chunk in chunks:
            chunk.metadata["doc_label"] = label
            # Anchoring text to source
            chunk.page_content = f"[{label} Report] " + chunk.page_content
            
        self.all_chunks.extend(chunks)

    def finalize_index(self):
        print("---Building Comparative Index---")
        self.vectorstore = Chroma.from_documents(
            self.all_chunks, self.embed_model, collection_name="comp_fin_rag_final"
        )
        
        # Keyword-based retrieval 
        bm25 = BM25Retriever.from_documents(self.all_chunks)
        bm25.k = 15
        
        # Ensemble: 40% keyword, 60% semantic vector
        self.retriever = EnsembleRetriever(
            retrievers=[bm25, self.vectorstore.as_retriever(search_kwargs={"k": 15})],
            weights=[0.4, 0.6]
        )

    def _rerank(self, query: str, docs: List[Document]) -> List[Document]:
        if not docs: return []
        pairs = [[query, d.page_content] for d in docs]
        scores = self.reranker.predict(pairs)
        return [d for d, _ in sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)[:8]]

    def _plot_metrics(self, data_dict):
        """Generate a prof side-by-side bar chart"""
        df = pd.DataFrame(data_dict).T
        
        # Numerical conversion for financial metrics
        target_cols = ['Revenue', 'Net_Income', 'Total_Liabilities']
        available_cols = [col for col in target_cols if col in df.columns]
        
        for col in available_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        if not available_cols:
            print("No valid metrics found for visualization")
            return

        plt.style.use('ggplot')
        ax = df[available_cols].plot(kind='bar', figsize=(12, 7), width=0.8)
        
        plt.title('Comparative Financial Metrics ($ Millions)', fontsize=15, fontweight='bold', pad=20)
        plt.ylabel('Amount in millions (USD)', fontsize=12)
        plt.xlabel('Entity', fontsize=12)
        plt.xticks(rotation=0, fontsize=11)
        plt.legend(title="Metrics", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f'${height:,.0f}', 
                        (p.get_x() + p.get_width() / 2., height), 
                        ha='center', va='center', xytext=(0, 10), 
                        textcoords='offset points', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.show()

    def compare(self, query: str):
        # 1. Retrieve and precision Re-ranking
        initial_hits = self.retriever.invoke(query)
        context_docs = self._rerank(query, initial_hits)
        
        context_text = "\n\n".join([
            f"SOURCE {d.metadata['doc_label']} (Pg {d.metadata.get('page','?') }):\n{d.page_content}" 
            for d in context_docs
        ])
        
        # 2. Prompting with JSON extraction
        prompt = f"""
        Analyze the context to compare entities for: {query}
        Context: {context_text}
        
        Response Format:
        1. A generalized comparative summary in text focusing on trends and risks.
        2. A separate JSON block for visualization at the very end.
        
        The JSON block MUST follow this exact format:
        DATA_JSON: {{"Entity_A_Name": {{"Revenue": 123, "Net_Income": 45, "Total_Liabilities": 80}}, "Entity_B_Name": {{"Revenue": 200, "Net_Income": 60, "Total_Liabilities": 100}}}}
        Ensure all values are numeric representing millions.
        """
        
        full_response = self.llm.invoke(prompt).content
        
        # 3. Handling text response and graphical logic
        try:
            # Regex to find JSON data block
            json_match = re.search(r"DATA_JSON:\s*(\{.*\})", full_response, re.DOTALL)
            if json_match:
                data_dict = json.loads(json_match.group(1))
                self._plot_metrics(data_dict)
                
                # Striping JSON from output txt
                clean_text = full_response.split("DATA_JSON:")[0].strip()
                return clean_text
        except Exception as e:
            print(f"--- Visualisation skipped due to data parsing: {e} ---")
            
        return full_response

# --- STEP 3: EXECUTION FLOW ---

rag = ComparativeFinancialRAG()

# Doc 1
print("1. Upload First Document:")
up1 = files.upload()
if up1:
    name1 = list(up1.keys())[0]
    label1 = input(f"Label for {name1}: ")
    rag.add_document(name1, label1)

# Doc 2
print("\n2. Upload Second Document:")
up2 = files.upload()
if up2:
    name2 = list(up2.keys())[0]
    label2 = input(f"Label for {name2}: ")
    rag.add_document(name2, label2)

# Building index
rag.finalize_index()

# Running analysis
print("\n" + "="*50)
print("--- RUNNING COMPARATIVE ANALYSIS ---")
final_query = f"Compare the revenue, net income, and total liabilities between {label1} and {label2}."
result = rag.compare(final_query)
print(f"\nANALYSIS REPORT:\n{result}")
