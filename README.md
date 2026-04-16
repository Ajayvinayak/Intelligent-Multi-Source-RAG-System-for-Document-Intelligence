# 🚀 Intelligent RAG System - Quick Start

An enterprise-grade **Retrieval-Augmented Generation (RAG)** system that recognizes data from PDFs, text files, CSV, JSON and provides accurate, fact-based answers grounded in source documents.

## ✨ Key Features

- 📄 **Multi-format Support**: PDF, TXT, CSV, JSON
- 🔍 **Semantic Search**: OpenAI embeddings + Pinecone vector DB
- 🤖 **Intelligent Answers**: LangChain + GPT-3.5-Turbo
- 🚀 **Production Ready**: FastAPI REST API
- 💻 **Easy CLI**: Command-line interface for testing
- 📊 **Source Attribution**: Track where answers come from
- ✅ **Confidence Scoring**: Know how reliable answers are

## 🎯 Quick Start (5 minutes)

### 1️⃣ Install
```bash
# Clone/download files
mkdir rag-system && cd rag-system

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2️⃣ Configure
Create `.env` file:
```env
OPENAI_API_KEY=sk-your-key-here
PINECONE_API_KEY=your-key-here
PINECONE_ENVIRONMENT=your-environment
PINECONE_INDEX_NAME=rag-system
```

### 3️⃣ Upload Documents
```bash
# Single file
python3 rag_client.py --upload document.pdf

# Multiple files
python3 rag_client.py --batch ./documents --pattern "*.pdf"

# Check what's loaded
python3 rag_client.py --stats
```

### 4️⃣ Query the System
```bash
# Interactive mode (recommended for testing)
python3 rag_client.py --interactive

# Single query
python3 rag_client.py --query "What is the main topic?"

# Run tests
python3 rag_client.py --test
```

## 🔧 API Usage (Production)

### Start Server
```bash
python3 rag_system_main.py
# Server runs at http://localhost:8000
```

### Upload Document
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@document.pdf"
```

### Query
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the main topic?",
    "top_k": 5,
    "include_sources": true
  }'
```

### View API Docs
Open: http://localhost:8000/docs

## 📚 Supported Data Formats

| Format | Extension | Example |
|--------|-----------|---------|
| PDF | `.pdf` | research_paper.pdf |
| Text | `.txt` | notes.txt |
| CSV | `.csv` | data.csv |
| JSON | `.json` | config.json |

## 📖 File Structure

```
rag-system/
├── rag_system_main.py      # Main RAG system (FastAPI server)
├── rag_client.py           # CLI client
├── examples.py             # Example usage scripts
├── requirements.txt        # Dependencies
├── .env                    # Configuration (create this)
├── SETUP_GUIDE.md         # Detailed setup guide
└── documents/             # Your documents folder
```

## 🎮 CLI Commands

### Upload
```bash
# Single file
python3 rag_client.py --upload file.pdf

# Directory (batch)
python3 rag_client.py --batch ./documents
python3 rag_client.py --batch ./documents --pattern "*.pdf"
```

### Query
```bash
# Single query
python3 rag_client.py --query "Your question here?"

# Interactive mode
python3 rag_client.py --interactive

# With more sources
python3 rag_client.py --query "Question?" --top-k 10
```

### Utilities
```bash
# System statistics
python3 rag_client.py --stats

# Run test queries
python3 rag_client.py --test

# Show help
python3 rag_client.py --help
```

## 🐍 Python Integration

```python
from rag_system_main import RAGSystem

# Initialize
rag = RAGSystem()

# Add documents
rag.add_documents("document.pdf")

# Query
result = rag.query("What is the summary?", top_k=5)

# Access results
print(result['answer'])           # The answer
print(result['confidence'])       # Confidence score
print(result['sources'])          # Source documents
```

## 🔐 Environment Setup

### Get API Keys

**OpenAI:**
1. Go to https://platform.openai.com/api-keys
2. Create new secret key
3. Copy and paste in `.env`

**Pinecone:**
1. Go to https://www.pinecone.io/
2. Create project
3. Get API key from dashboard
4. Copy environment name

### .env Template
```env
# OpenAI
OPENAI_API_KEY=sk-...

# Pinecone
PINECONE_API_KEY=...
PINECONE_ENVIRONMENT=us-west1-gcp
PINECONE_INDEX_NAME=rag-system
```

## 📊 System Architecture

```
User Input
    ↓
[Document Upload/Query]
    ↓
[Document Processor] ← Handles PDF, TXT, CSV, JSON
    ↓
[Text Splitter] ← Creates manageable chunks
    ↓
[Embeddings] ← OpenAI creates semantic vectors
    ↓
[Vector DB] ← Pinecone stores embeddings
    ↓
[Semantic Search] ← Finds relevant documents
    ↓
[LLM] ← GPT generates answer
    ↓
[Response] ← Answer + Sources + Confidence
```

## 🧪 Test the System

### Option 1: Interactive Mode (Recommended)
```bash
python3 rag_client.py --interactive
```
Then type queries at the prompt.

### Option 2: Run Predefined Tests
```bash
python3 rag_client.py --test
```
Generates `test_results_*.json` with results.

### Option 3: Run Examples
```bash
python3 examples.py
```
Choose example number to run.

## ⚡ Performance Tips

- **First upload slower**: Embeddings are created first time
- **Larger documents**: Automatically chunked for optimization
- **Query speed**: Typically 1-3 seconds per query
- **Batch uploads**: Process many files faster than one-by-one

## 🆘 Troubleshooting

### Error: "API key not found"
✓ Check `.env` file exists
✓ Verify key format (should start with `sk-` for OpenAI)
✓ No quotes in `.env`

### Error: "No relevant documents found"
✓ Check documents uploaded: `python3 rag_client.py --stats`
✓ Try different query phrasing
✓ Upload more documents

### Slow responses
✓ Reduce `top_k` value
✓ Check internet connection
✓ Verify API quotas

## 📚 Full Documentation

See `SETUP_GUIDE.md` for:
- Detailed installation
- Configuration options
- Architecture details
- Optimization guide
- Advanced usage
- Security best practices

## 💡 Example Workflows

### Workflow 1: Document Q&A
```bash
# Upload document
python3 rag_client.py --upload research.pdf

# Ask questions
python3 rag_client.py --query "What are the key findings?"
python3 rag_client.py --query "Who are the authors?"
```

### Workflow 2: Multi-Source Analysis
```bash
# Upload multiple sources
python3 rag_client.py --batch ./documents

# Ask cross-source questions
python3 rag_client.py --query "Correlate data from all sources"
```

### Workflow 3: Interactive Session
```bash
# Start interactive mode
python3 rag_client.py --interactive

# Ask multiple related questions
# Questions build context
```

## 🚀 Next Steps

1. ✅ Install dependencies
2. ✅ Set up environment variables
3. ✅ Upload sample documents
4. ✅ Test with queries
5. ✅ Deploy server for production
6. ✅ Integrate into applications

## 📝 Example Output

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📚 RAG Query Result
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 Answer:
The main findings indicate significant improvements in
data processing efficiency, with a 45% reduction in
processing time through optimized algorithms.

📊 Confidence: 87.3%
📖 Sources Retrieved: 3

📌 Source Documents:
───────────────────────────────────
1. research_paper.pdf
   Type: pdf
   Relevance: 0.892
   Content: "This study demonstrates the effectiveness..."
```

## 🎓 Use Cases

- 📄 **Document Analysis**: Understand large documents quickly
- 🔍 **Research**: Search through academic papers
- 💼 **Business Intelligence**: Analyze reports and data
- 📊 **Data Insights**: Extract insights from datasets
- 🤖 **Chatbots**: Power knowledge-based chatbots
- 📚 **Learning**: Study materials with Q&A

## 🤝 Contributing

Found an issue? Have a suggestion?
1. Check the troubleshooting section
2. Review `SETUP_GUIDE.md`
3. Check logs for error messages

## 📄 License

This project is provided as-is for educational and commercial use.

## 👨‍💻 Author

**Ajay Vinayak Y**
- Data Science Specialization
- SRM Institute of Science and Technology
- Chennai, India

---

## 🔗 Useful Links

- **OpenAI API**: https://platform.openai.com/
- **Pinecone**: https://www.pinecone.io/
- **LangChain**: https://langchain.com/
- **FastAPI**: https://fastapi.tiangolo.com/

---

**Ready to go?** Start with: `python3 rag_client.py --interactive` 🚀
