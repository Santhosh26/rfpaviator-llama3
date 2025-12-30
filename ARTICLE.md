# Building RFP Aviator: How We Used RAG and LLMs to Automate Enterprise RFP Responses

*A deep dive into building an AI-powered RFP response system using Retrieval-Augmented Generation, LangChain, and Meta's Llama 3.1*

---

## The Problem: RFPs Are a Time Sink

If you've ever worked in enterprise sales or solution consulting, you know the pain of responding to Request for Proposals (RFPs). These documents can contain hundreds of questions about your product's capabilities, security posture, compliance certifications, and integration features.

A typical scenario: A solution consultant receives a 200-question RFP questionnaire on a Thursday afternoon with a Monday deadline. They spend the entire weekend digging through product documentation, data sheets, and previous RFP responses, trying to craft accurate answers while maintaining consistency with what the company has promised before.

We knew there had to be a better way. With the explosion of Large Language Models (LLMs) and the maturation of Retrieval-Augmented Generation (RAG) techniques, we saw an opportunity to build something that could transform how our team handles RFPs.

**The result is RFP Aviator** - an AI-powered system that processes RFP questionnaires, retrieves relevant product documentation, generates accurate responses, and produces a compliance matrix. In testing, we achieved **90% accuracy** and can process **100 questions in just 10 minutes**.

---

## Why RAG Instead of Pure LLMs?

When we started this project in 2024, ChatGPT and Claude were already impressive. Why not just paste questions into an LLM and call it a day?

### The Hallucination Problem

Pure LLMs, no matter how advanced, have a fundamental limitation: they generate responses based on patterns learned during training, not real-time facts. When asked "Does your product support SAML 2.0 authentication?", a vanilla LLM might confidently say "Yes" - even if your product doesn't support it.

For RFP responses, hallucinations aren't just embarrassing; they're potentially contract-breaking. A single incorrect claim about compliance or capability could lead to legal issues down the road.

### The Knowledge Cutoff Problem

LLMs are trained on data up to a certain date. Product features change constantly. New versions ship. Capabilities are added or deprecated. An LLM trained on 2023 data won't know about features released in 2024.

### Enter RAG: Grounding LLMs in Your Data

Retrieval-Augmented Generation solves both problems by:

1. **Retrieving** relevant chunks of your actual documentation before generation
2. **Augmenting** the LLM prompt with this retrieved context
3. **Generating** responses grounded in real, up-to-date information

The LLM becomes a sophisticated language engine that synthesizes information from your knowledge base rather than relying on potentially outdated or incorrect training data.

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Question   │────▶│   Retrieve   │────▶│   Generate   │
│              │     │   Context    │     │   Response   │
└──────────────┘     └──────────────┘     └──────────────┘
                            │
                            ▼
                    ┌──────────────┐
                    │   Product    │
                    │   Docs DB    │
                    └──────────────┘
```

---

## Our Technology Choices: Why We Picked What We Picked

### Python: The Obvious Choice

For AI/ML projects in 2025, Python remains the undisputed champion. The ecosystem is unmatched - every major framework, every vector database, every LLM API has first-class Python support. We didn't spend a second debating this.

### LangChain: The Orchestration Layer

When we evaluated frameworks for building our RAG pipeline, LangChain stood out for several reasons:

1. **Composability**: LangChain's LCEL (LangChain Expression Language) lets us chain components together elegantly
2. **Provider Flexibility**: Easy to swap between OpenAI, Anthropic, AWS Bedrock, or local models
3. **Battle-tested Integrations**: Vector stores, document loaders, retrievers - all built-in
4. **Active Development**: The framework evolves rapidly with the AI landscape

Here's how our core RAG chain looks:

```python
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_aws import BedrockLLM
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def rfp_qa_chain(persist_directory, template, temp):
    # Setup embeddings and vector store
    embedding = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    vectordb = Chroma(persist_directory=persist_directory,
                      embedding_function=embedding)

    # Initialize LLM
    llm = BedrockLLM(
        credentials_profile_name="default",
        model_id="meta.llama3-1-70b-instruct-v1:0",
        model_kwargs={"temperature": 1 if temp else 0},
        region='us-west-2'
    )

    # MMR retriever for diverse, relevant context
    retriever = vectordb.as_retriever(
        search_type='mmr',
        search_kwargs={'lambda_mult': 0.1}
    )

    prompt = ChatPromptTemplate.from_template(template)

    # Compose the chain
    qa_chain = (
        {"context": retriever | format_docs,
         "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return qa_chain
```

This chain elegantly expresses our entire pipeline: retrieve context, format it, inject into prompt, generate response, parse output.

### ChromaDB: Lightweight But Capable

For our vector database, we chose ChromaDB over alternatives like Pinecone, Weaviate, or Milvus. Here's why:

**Pros:**
- **Self-hosted**: No external API calls, data stays on-premise
- **Simple**: File-based persistence, no separate server process
- **Python-native**: Feels natural in our stack
- **Good enough**: For our document corpus size, ChromaDB handles queries with sub-second latency

**Trade-offs:**
- Not designed for massive scale (millions of vectors)
- No built-in multi-tenancy
- Limited advanced features compared to enterprise solutions

For an enterprise internal tool with tens of thousands of document chunks across ~20 products, ChromaDB is perfect. If we needed to scale to hundreds of millions of vectors or support real-time updates from thousands of users, we'd reconsider.

### Llama 3.1 70B via AWS Bedrock: Enterprise-Grade Open Source

The LLM choice was interesting. We had options:

| Option | Pros | Cons |
|--------|------|------|
| GPT-4/GPT-4o | Best quality, fast | Data leaves organization |
| Claude 3.5 Sonnet | Excellent reasoning | Data leaves organization |
| Self-hosted Llama | Full control | GPU infrastructure overhead |
| AWS Bedrock Llama | Control + managed | AWS dependency |

We went with **Meta's Llama 3.1 70B via AWS Bedrock** because:

1. **Data Sovereignty**: Customer data and product docs never leave our AWS environment
2. **No GPU Management**: Bedrock handles scaling and availability
3. **Cost Predictable**: Pay-per-token, no idle GPU costs
4. **Quality**: Llama 3.1 70B is remarkably capable for structured Q&A tasks

In late 2024, with Llama 3.1's release, the open-source model quality reached a point where it genuinely competes with proprietary models for many enterprise use cases.

### HuggingFace Embeddings: all-MiniLM-L6-v2

For converting text to vectors, we use the `all-MiniLM-L6-v2` model from Sentence Transformers:

```python
embedding = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
```

Why this model?
- **384-dimensional vectors**: Compact enough for fast search
- **Well-tested**: Millions of downloads, proven in production
- **Local inference**: No API calls needed
- **General purpose**: Works well across domains

In 2025, newer models like `bge-large-en-v1.5` or `nomic-embed-text` might give better retrieval quality, but MiniLM's speed-to-quality ratio was right for our needs.

### Streamlit: From Prototype to Production

For the UI, we chose Streamlit. Some might call this unconventional for a "production" app, but consider:

1. **Speed**: We went from concept to working UI in days, not weeks
2. **Python-native**: No JavaScript context switching
3. **Good enough UX**: For internal tools, Streamlit's components are perfectly adequate
4. **Session management**: Built-in state handling

```python
import streamlit as st

st.set_page_config(page_title="Opentext RFP Aviator", layout="wide")

def check_authentication():
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if st.session_state["authenticated"]:
        home_main()  # Show main app
    else:
        show_login_page()  # Show login
```

---

## Building the Multi-Product RAG Pipeline

One unique aspect of RFP Aviator is that we support **20+ different products**, each with its own documentation corpus and domain expertise requirements.

### Product-Specific Vector Stores

Instead of one giant vector store, we maintain separate ChromaDB instances per product:

```
chromastore/
├── Fortify_RFP/          # Application Security
├── NetIQ_RFP/            # Identity Management
├── Arcsight_RFP/         # SIEM
├── Voltage_RFP/          # Data Security
├── Documentum_RFP/       # Content Management
├── IDOL_RFP/             # Analytics
├── Vertica_RFP/          # Database
└── ... (15+ more)
```

This architecture has advantages:
- **Focused retrieval**: Questions about Fortify only search Fortify docs
- **Independent updates**: Refresh one product's knowledge without touching others
- **Cleaner separation**: No cross-contamination between product domains

### Document Ingestion Pipeline

Our ingestion script handles multiple file formats:

```python
from langchain_community.document_loaders import (
    PDFMinerLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader,
    TextLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

def process_files(directory_path, files_in_directory):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    for file_type, loader_class in [
        ('.pdf', PDFMinerLoader),
        ('.docx', Docx2txtLoader),
        ('.xlsx', UnstructuredExcelLoader),
        ('.txt', TextLoader)
    ]:
        if any(file.endswith(file_type) for file in files_in_directory):
            loader = DirectoryLoader(
                directory_path,
                glob=f'*{file_type}',
                loader_cls=loader_class
            )
            doc = loader.load()
            chunks = split_documents(doc)
            clean_chunks = clean_text(chunks)

            Chroma.from_documents(
                documents=clean_chunks,
                embedding=embeddings,
                persist_directory=persist_dir
            )
```

**Chunking Strategy:**
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
```

We use 1000-character chunks with 200-character overlap. The overlap ensures we don't lose context at chunk boundaries - crucial when a feature description spans paragraphs.

### Role-Based Prompt Engineering

Different products require different expertise personas. A question about application security needs different framing than one about identity management:

```python
product_configs = {
    "Fortify": "Application Security architect, Fortify",
    "NetIQ": "Identity and access management architect, NetIQ",
    "Arcsight": "Security Operations architect, Arcsight",
    "Voltage": "Data Security architect, Voltage",
    "AppWorks": "Process Automation architect, AppWorks",
    "Documentum": "Content Management architect, Documentum",
    "IDOL": "Intelligent Data Operating Layer architect, IDOL",
    "Vertica": "analytical database expert, Vertica",
    # ... more products
}
```

The prompt template uses these roles:

```python
template = """You are an expert {role} with extensive experience
in Opentext {product_name}. Answer the following question based
on the provided context. If you don't have enough information
to answer, state that you don't know.

Instructions to be followed:
- Begin your response with either "Yes," or "No," depending on
  the sentiment of your answer
- Always provide an explanation for your answer
- Do not mention that your answer is based on retrieved information
- Stick to answering the specific question asked
- Do not include any formatting characters in your response

Context: {context}

Question: {question}

Answer:"""
```

Notice the explicit instruction to start with "Yes," or "No,". This isn't arbitrary - it enables our compliance scoring.

### MMR Retrieval: Beyond Simple Similarity

We use Maximum Marginal Relevance (MMR) for retrieval:

```python
retriever = vectordb.as_retriever(
    search_type='mmr',
    search_kwargs={'lambda_mult': 0.1}
)
```

Standard similarity search might return multiple chunks saying essentially the same thing. MMR balances relevance with diversity - giving the LLM varied context that covers different aspects of the question.

The `lambda_mult=0.1` setting heavily favors diversity. We found this crucial for RFP questions that often touch multiple aspects (features AND compliance AND integration).

### Compliance Scoring: Turning Responses into Metrics

RFPs aren't just about answering questions; they're about demonstrating compliance. Our system automatically classifies responses:

```python
def status_checker(result):
    if result.strip().startswith("Yes"):
        return "Compliant"
    elif result.strip().startswith("No"):
        return "Non-Compliant"
    else:
        return "Review required"
```

This simple heuristic works because we engineered the prompt to enforce Yes/No prefixes. The "Review required" fallback catches edge cases where the LLM doesn't follow instructions.

We then aggregate to a compliance percentage:

```python
def get_compliance_percentage(status_list):
    compliant = status_list.count("Compliant")
    non_compliant = status_list.count("Non-Compliant")
    total = compliant + non_compliant

    if total == 0:
        return 0, 0

    return (compliant/total * 100, non_compliant/total * 100)
```

A product manager can glance at "78% Compliant" and immediately understand the fit for this RFP.

---

## Real-World Performance

After deploying RFP Aviator internally, we measured:

| Metric | Result |
|--------|--------|
| **Accuracy** | 90% (validated against expert responses) |
| **Processing Speed** | 100 questions in ~10 minutes |
| **Time Savings** | ~70% reduction in consultant hours per RFP |
| **Consistency** | Eliminated contradictory responses across RFPs |

The 90% accuracy means human review is still essential - but reviewers are now editing AI drafts rather than writing from scratch.

---

## The 2025 AI Landscape: Where We Are Now

Building RFP Aviator in 2024-2025 feels different than it would have in 2023. The landscape has matured dramatically:

### LLMs: Commoditization and Specialization

GPT-4's release in early 2023 felt revolutionary. By late 2024, we had:
- **Claude 3.5 Sonnet/Opus 4.5**: Anthropic's models excel at nuanced reasoning
- **Llama 3.1 70B/405B**: Open-source catching up to proprietary
- **Gemini 2.0**: Google's multimodal powerhouse
- **Specialized models**: Fine-tuned models for code, medical, legal domains

For enterprise RAG, the "best" LLM often isn't the biggest - it's the one that follows instructions reliably and handles your domain well. Llama 3.1 70B proved excellent for structured Q&A.

### Vector Databases: Enterprise-Ready

The vector database space exploded:
- **Pinecone**: Fully managed, scales massively
- **Weaviate**: Open-source with hybrid search
- **Qdrant**: Rust-based, high performance
- **ChromaDB**: Simple and developer-friendly
- **pgvector**: PostgreSQL extension for existing infra

For greenfield projects in 2025, I'd seriously evaluate **pgvector** - many teams already have PostgreSQL, and adding vector search to your existing database simplifies architecture significantly.

### RAG: From Technique to Standard Practice

RAG went from cutting-edge research to standard practice. Every enterprise AI deployment I see now uses some form of retrieval augmentation. The question isn't "should we use RAG?" but "how should we implement RAG?"

Key patterns that emerged:
- **Hybrid search**: Combining keyword (BM25) with semantic (vector) search
- **Reranking**: Using cross-encoder models to reorder retrieved results
- **Query expansion**: LLM-generated multiple query variants
- **Contextual chunking**: Smarter document splitting that preserves meaning

### Agentic Systems: The Next Frontier

Looking ahead, the next evolution is agentic RAG - systems that don't just retrieve and generate but reason about what to retrieve, when to retrieve more, and how to verify their own outputs.

---

## Improvements We'd Make Today

If we rebuilt RFP Aviator today, here's what we'd change:

### 1. Upgrade Embedding Model

`all-MiniLM-L6-v2` served us well, but newer models offer better retrieval:

```python
# Instead of:
embedding = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

# Consider:
embedding = HuggingFaceEmbeddings(model_name='BAAI/bge-large-en-v1.5')
# Or:
embedding = HuggingFaceEmbeddings(model_name='nomic-ai/nomic-embed-text-v1.5')
```

Benchmarks show 5-15% retrieval quality improvements with modern embeddings.

### 2. Implement Hybrid Search

Pure semantic search misses when users ask questions with specific terminology. Hybrid search combines:

```python
# Conceptual - actual implementation varies by vector DB
from langchain.retrievers import BM25Retriever, EnsembleRetriever

bm25_retriever = BM25Retriever.from_documents(docs)
semantic_retriever = vectordb.as_retriever()

hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, semantic_retriever],
    weights=[0.3, 0.7]
)
```

### 3. Add Response Caching

Many RFP questions are repeated across questionnaires. Caching saves compute and ensures consistency:

```python
import hashlib

def get_cached_response(question, product):
    cache_key = hashlib.md5(f"{question}:{product}".encode()).hexdigest()
    cached = redis_client.get(cache_key)
    if cached:
        return cached

    response = qa_chain.invoke(question)
    redis_client.setex(cache_key, 86400, response)  # 24h TTL
    return response
```

### 4. Implement Feedback Loops

Currently, we don't learn from corrections. A feedback system would:
1. Track when humans modify AI responses
2. Identify patterns in corrections
3. Fine-tune prompts or retrieval based on feedback

### 5. Multi-Modal Document Processing

Product documentation increasingly includes diagrams, screenshots, and videos. Multi-modal models (GPT-4V, Claude 3 Vision, Gemini) can now process these:

```python
# Future enhancement: Process architecture diagrams
from langchain_community.document_loaders import UnstructuredImageLoader

image_loader = UnstructuredImageLoader("architecture.png")
image_doc = image_loader.load()
# Extract and embed visual information
```

### 6. Better Secrets Management

Our current implementation has some API keys in code (a security anti-pattern). Production systems should use:

```python
import boto3
from botocore.exceptions import ClientError

def get_secret(secret_name):
    client = boto3.client('secretsmanager')
    response = client.get_secret_value(SecretId=secret_name)
    return response['SecretString']

# Usage
perplexity_key = get_secret('rfp-aviator/perplexity-api-key')
```

### 7. Add Claude or GPT-4o as Alternative LLMs

Model diversity provides fallbacks and lets users choose based on preference:

```python
def get_llm(provider="bedrock"):
    if provider == "bedrock":
        return BedrockLLM(model_id="meta.llama3-1-70b-instruct-v1:0", ...)
    elif provider == "anthropic":
        return ChatAnthropic(model="claude-3-5-sonnet-20241022", ...)
    elif provider == "openai":
        return ChatOpenAI(model="gpt-4o", ...)
```

---

## The Value Proposition: Why This Matters

### For Solution Consultants

- **Time Reclaimed**: Hours of documentation diving replaced by minutes of response review
- **Consistency**: No more contradicting what a colleague said in last month's RFP
- **Confidence**: Responses grounded in actual documentation, not memory

### For Sales Teams

- **Faster Turnaround**: Meet aggressive RFP deadlines without weekend work
- **Higher Win Rates**: More time for strategic positioning, less for mechanical answers
- **Competitive Intelligence**: (Future) Know how you stack against competitors

### For Product Teams

- **Documentation Feedback**: Questions the system can't answer reveal documentation gaps
- **Feature Requests**: Patterns in "Non-Compliant" responses show market demands

### For the Organization

- **Scalability**: Handle more RFPs without proportionally increasing headcount
- **Knowledge Preservation**: Institutional knowledge captured in vector stores, not just in experts' heads
- **Quality Standardization**: AI enforces consistent messaging across all customer touchpoints

---

## Conclusion: Building AI Tools That Actually Work

RFP Aviator taught us that successful enterprise AI isn't about chasing the most powerful model or the most sophisticated architecture. It's about:

1. **Solving a real pain point**: RFP responses were genuinely painful
2. **Grounding in truth**: RAG prevents hallucination through retrieval
3. **Measuring what matters**: 90% accuracy, 10-minute processing, compliance scores
4. **Building for humans**: AI drafts, humans review - not full automation
5. **Iterating pragmatically**: Ship something useful, improve over time

The system we built in 2024 continues to evolve. We're planning:
- **Competitor Analysis**: Compare our products against competitors on RFP dimensions
- **Knowledge Aviator**: An internal knowledge base using the same RAG infrastructure
- **Multi-modal Processing**: Understanding diagrams and screenshots in documentation

The AI landscape will continue advancing. New models, new techniques, new frameworks will emerge. But the fundamentals - retrieval augmentation, prompt engineering, human-in-the-loop workflows - these patterns will persist.

If you're building enterprise AI tools, start with the problem. Understand your data. Ground your generations in truth. And ship something useful before it's perfect.

---

*RFP Aviator was built using Python, LangChain, ChromaDB, Meta Llama 3.1, and Streamlit. The system processes RFP questionnaires for 20+ products and achieves 90% accuracy with a 70% reduction in response time.*

---

## Technical Appendix: Key Code References

| Component | File | Description |
|-----------|------|-------------|
| RAG Chain | `src/rfpResponder/newrfp_qa_chain.py:21-76` | LangChain pipeline construction |
| Prompt Templates | `src/rfpResponder/chooseTemplate.py:22-77` | Product-specific prompts |
| Vector Store Creation | `enhanced_CreateVectorStore.py:1-57` | Document ingestion |
| Compliance Scoring | `src/rfpResponder/response_status.py` | Yes/No classification |
| Main Orchestration | `RFPAvaitor.py` | UI and processing flow |

## Dependencies

```
langchain>=0.1.0
langchain-community
langchain-aws
chromadb==0.4.15
sentence-transformers==2.2.2
streamlit
pdfminer.six
docx2txt
pandas
```
