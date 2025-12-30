# RFP Aviator

**AI-Powered RFP Response Automation Using RAG and LLMs**

RFP Aviator is an intelligent system that automates responses to Request for Proposal (RFP) and Request for Information (RFI) questionnaires. Built for enterprise environments, it leverages Retrieval-Augmented Generation (RAG) to generate accurate, compliance-scored responses grounded in product documentation.

## Key Features

- **Automated RFP Processing**: Upload CSV/Excel questionnaires and receive AI-generated responses
- **Multi-Product Support**: 20+ products with dedicated knowledge bases
- **Compliance Scoring**: Automatic classification (Compliant/Non-Compliant/Review Required)
- **RAG Architecture**: Responses grounded in actual product documentation, not LLM hallucinations
- **Enterprise Security**: Self-hosted vector stores, AWS Bedrock for LLM inference

## Performance

| Metric | Result |
|--------|--------|
| Accuracy | 90% |
| Processing Speed | 100 questions in ~10 minutes |
| Time Savings | ~70% reduction in consultant hours |

## Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.x |
| Web Framework | Streamlit |
| LLM Framework | LangChain |
| Vector Database | ChromaDB |
| LLM | Meta Llama 3.1 70B (AWS Bedrock) |
| Embeddings | HuggingFace all-MiniLM-L6-v2 |
| Database | PostgreSQL |

## Architecture

```
┌─────────────────┐
│  CSV Upload     │
│  (RFP Questions)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│  Query Format   │────▶│  ChromaDB       │
│  (+ Product     │     │  Vector Store   │
│   Context)      │     │  (MMR Search)   │
└────────┬────────┘     └────────┬────────┘
         │                       │
         │    Retrieved Context  │
         │◀──────────────────────┘
         │
         ▼
┌─────────────────┐
│  LangChain +    │
│  Llama 3.1 70B  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Response +     │
│  Compliance     │
│  Score          │
└─────────────────┘
```

## Installation

### Prerequisites

- Python 3.9+
- AWS Account with Bedrock access
- PostgreSQL database
- AWS CLI configured

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-org/rfpaviator-llama3.git
   cd rfpaviator-llama3
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   .venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**

   Create a `.env` file:
   ```bash
   POSTGRES_USER=your_db_user
   POSTGRES_PASSWORD=your_db_password
   ```

5. **Configure AWS credentials**
   ```bash
   aws configure --profile default
   ```
   Ensure you have access to AWS Bedrock in `us-west-2` region.

6. **Set up PostgreSQL**

   Create a `users` table with columns:
   - `user_id` (PRIMARY KEY)
   - `password_hash` (SHA-256)
   - `fullname`

7. **Initialize vector stores**

   For each product, run:
   ```bash
   python enhanced_CreateVectorStore.py /path/to/product/docs
   ```

## Usage

1. **Start the application**
   ```bash
   streamlit run app.py
   ```

2. **Access the UI**

   Open `http://localhost:8686` in your browser

3. **Process an RFP**
   - Log in with your credentials
   - Accept the compliance consent
   - Select Product Group → Product → Solutions
   - Upload your RFP questionnaire (CSV format)
   - Review generated responses and compliance scores
   - Download results

## Supported Products

### CyberSecurity
- Fortify (SAST, DAST, SCA)
- NetIQ (Identity Management)
- Arcsight (SIEM)
- Voltage (Data Security)

### Content Management
- AppWorks
- Documentum
- Exstream
- Extended ECM
- InfoArchive
- Media Management

### IT Operations
- CMS
- HCMX
- NOM
- Operations Bridge
- SMAX-AMX

### Application Delivery Management
- ALM.NET
- ALM Octane
- Performance Testing
- Functional Testing

### Analytics & AI
- IDOL
- Vertica

## Project Structure

```
rfpaviator-llama3/
├── app.py                      # Entry point
├── RFPAvaitor.py               # Main processing logic
├── login.py                    # Authentication
├── enhanced_CreateVectorStore.py # Vector store creation
├── requirements.txt            # Dependencies
├── src/
│   ├── rfpResponder/           # RAG pipeline
│   │   ├── newrfp_qa_chain.py  # LangChain QA chain
│   │   ├── chooseTemplate.py   # Prompt templates
│   │   └── ...
│   └── compare_comp/           # Competitor analysis
├── chromastore/                # Vector databases
├── static/                     # Sample files
└── .streamlit/                 # Streamlit config
```

## Configuration

### Streamlit (.streamlit/config.toml)

```toml
[server]
port = 8686
enableStaticServing = true

[theme]
primaryColor = "#0066ff"
base = "light"
```

### Model Parameters

- **Temperature**: 0 (standard) or 1 (detailed responses)
- **Retrieval**: MMR with lambda_mult=0.1
- **Chunk Size**: 1000 characters with 200 overlap

## Adding New Products

1. Gather product documentation (PDF, DOCX, XLSX, TXT)
2. Create vector store:
   ```bash
   python enhanced_CreateVectorStore.py /path/to/docs
   ```
3. Add product to `src/rfpResponder/chooseProdcut.py`
4. Add prompt template to `src/rfpResponder/chooseTemplate.py`
5. Map datastore path in `src/rfpResponder/choose_datastore.py`

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

Copyright 2024 Open Text. All rights reserved.

## Acknowledgments

- Built with [LangChain](https://langchain.com/)
- Powered by [Meta Llama 3.1](https://llama.meta.com/) via [AWS Bedrock](https://aws.amazon.com/bedrock/)
- Vector storage by [ChromaDB](https://www.trychroma.com/)
- UI framework [Streamlit](https://streamlit.io/)
