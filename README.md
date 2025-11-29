# 🏢 Enterprise Knowledge Agent Platform

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> An AI-powered enterprise knowledge management system built with modern Agent architecture, featuring MCP protocol support, multi-LLM integration, and production-ready deployment.

## 🚧 Project Status

Currently in **Phase 2: Advanced Retrieval** - Complete ✅

### Completed Features

**Phase 1: Basic RAG Pipeline** ✅
- [x] Project initialization and structure
- [x] Document loaders (PDF, Markdown, TXT)
- [x] Multiple chunking strategies (Fixed, Sentence, Semantic)
- [x] Qdrant vector store integration
- [x] Basic vector search
- [x] CLI testing tool

**Phase 2: Advanced Retrieval** ✅
- [x] BM25 keyword search implementation
- [x] Hybrid search with Reciprocal Rank Fusion (RRF)
- [x] Cross-encoder reranking (BGE-reranker)
- [x] Response synthesis with automatic citations
- [x] Comprehensive test suite

### Coming Next

**Phase 3: Agent + MCP** (In Progress)
- [ ] Query router for intent classification
- [ ] ReAct reasoning engine
- [ ] Tool definitions and execution
- [ ] MCP server implementation
- [ ] Integration with Claude Desktop

## ✨ Current Features

| Feature | Description | Status |
|---------|-------------|--------|
| 📚 **Multi-format Loading** | PDF, Markdown, TXT support | ✅ Complete |
| ✂️ **Smart Chunking** | Fixed, Sentence, Semantic strategies | ✅ Complete |
| 🔍 **Hybrid Search** | Vector + BM25 with RRF fusion | ✅ Complete |
| 🎯 **Reranking** | Cross-encoder reranking | ✅ Complete |
| 📝 **Auto-Citations** | Automatic source tracking | ✅ Complete |
| 🤖 **AI Agent** | ReAct-based reasoning | 🔄 Phase 3 |
| 🔌 **MCP Protocol** | Model Context Protocol | 🔄 Phase 3 |
| 📊 **Multi-LLM** | OpenAI, Anthropic, Ollama | 🔄 Phase 4 |
| ☁️ **Cloud-Ready** | Docker, Kubernetes | 🔄 Phase 6 |

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Poetry (for dependency management)
- Docker (for Qdrant vector database)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/enterprise-knowledge-agent.git
cd enterprise-knowledge-agent

# Install dependencies using Poetry
poetry install

# Copy environment variables
cp .env.example .env

# Start Qdrant vector database
docker run -p 6333:6333 qdrant/qdrant
```

### Basic Usage

#### 1. Ingest Documents

```bash
# Ingest a single document
poetry run python src/cli.py ingest examples/sample_documents/hr_policies/vacation_policy.md

# Ingest with different chunking strategy
poetry run python src/cli.py ingest document.pdf --strategy sentence
```

#### 2. Search Knowledge Base

```bash
# Basic search
poetry run python src/cli.py search "vacation policy"

# With more results
poetry run python src/cli.py search "remote work requirements" --top-k 10
```

#### 3. Run Hybrid Search Demo

```bash
# Demonstrate Phase 2 capabilities
poetry run python examples/hybrid_search_demo.py
```

## 📊 Phase 2 Architecture

```
Query: "What is the vacation policy?"
    │
    ├─► Vector Search (Semantic)
    │   └─► Top 20 results
    │
    ├─► BM25 Search (Keyword)
    │   └─► Top 20 results
    │
    ▼
Reciprocal Rank Fusion (RRF)
    │
    └─► Fused Top 20 results
          │
          ▼
    Cross-Encoder Reranking (Optional)
          │
          └─► Top 5 results
                │
                ▼
          Response Synthesizer
                │
                └─► Answer + Citations
```

### Key Improvements in Phase 2

**Hybrid Search Benefits:**
- Combines semantic understanding (vector) with exact keyword matching (BM25)
- RRF fusion handles score normalization automatically
- More robust than single-method retrieval

**Reranking Benefits:**
- Cross-encoder models provide more accurate relevance scores
- Improves top-k precision significantly
- Worth the computational cost for final ranking

**Citation System:**
- Automatic source tracking
- Confidence scoring based on retrieval quality
- Structured citation extraction

## 🛠️ Tech Stack

### Core Frameworks
- **RAG**: [LlamaIndex](https://www.llamaindex.ai/) - Document processing
- **Retrieval**: 
  - [Qdrant](https://qdrant.tech/) - Vector similarity search
  - [rank-bm25](https://github.com/dorianbrown/rank_bm25) - Keyword search
  - [sentence-transformers](https://www.sbert.net/) - Reranking models
- **Agent**: [LangChain](https://www.langchain.com/) *(Phase 3)*
- **MCP**: [FastMCP](https://github.com/jlowin/fastmcp) *(Phase 3)*

### Infrastructure (Planned)
- **Cache**: Redis *(Phase 4)*
- **Database**: PostgreSQL *(Phase 4)*
- **Storage**: MinIO/S3 *(Phase 4)*
- **Observability**: Arize Phoenix *(Phase 5)*

## 📁 Project Structure

```
enterprise-knowledge-agent/
├── src/
│   ├── config.py              # Configuration management
│   ├── cli.py                 # Command-line interface
│   │
│   ├── rag/                   # RAG Pipeline
│   │   ├── ingestion/         # ✅ Document loading & chunking
│   │   ├── retrieval/         # ✅ Search & ranking
│   │   │   ├── vector_store.py      # Vector search
│   │   │   ├── bm25_search.py       # Keyword search
│   │   │   ├── hybrid_retriever.py  # Hybrid + RRF
│   │   │   └── reranker.py          # Cross-encoder reranking
│   │   └── generation/        # ✅ Response synthesis
│   │
│   ├── agent/                 # 🔄 Agent orchestrator (Phase 3)
│   ├── mcp_server/            # 🔄 MCP protocol (Phase 3)
│   ├── llm/                   # 🔄 LLM providers (Phase 4)
│   └── api/                   # 🔄 FastAPI (Phase 4)
│
├── tests/
│   └── unit/
│       ├── test_chunkers.py   # Chunking tests
│       └── test_phase2.py     # Phase 2 tests
│
├── examples/
│   ├── hybrid_search_demo.py  # Phase 2 demo
│   └── sample_documents/      # Test documents
│
├── docs/                      # Documentation
└── configs/                   # Configuration files
```

## 🧪 Testing

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=src

# Run specific test file
poetry run pytest tests/unit/test_phase2.py

# Run fast tests only (skip slow reranker tests)
poetry run pytest -m "not slow"
```

## 📊 Evaluation Metrics (Target)

Based on BEST_PRACTICES.md guidelines:

| Metric | Target | Phase 2 Status |
|--------|--------|---------------|
| **Context Relevance** | > 0.85 | 🔄 To be evaluated |
| **Answer Faithfulness** | > 0.90 | 🔄 Phase 4 (needs LLM) |
| **Answer Relevance** | > 0.85 | 🔄 Phase 4 (needs LLM) |
| **Latency (p95)** | < 2s | ✅ < 500ms currently |

*Note: Full RAG evaluation requires LLM integration (Phase 4)*

## 📖 Documentation

Comprehensive documentation available in `/mnt/project`:

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System design and component overview
- **[FRAMEWORK_EVALUATION.md](FRAMEWORK_EVALUATION.md)** - LangChain vs LlamaIndex analysis
- **[BEST_PRACTICES.md](BEST_PRACTICES.md)** - AI Agent development guidelines
- **[PROJECT_BRIEF.md](PROJECT_BRIEF.md)** - Complete development roadmap

## 🔧 Development

### Code Quality

```bash
# Format code
poetry run black src/ tests/
poetry run isort src/ tests/

# Lint
poetry run ruff check src/ tests/

# Type check
poetry run mypy src/
```

### Adding Dependencies

```bash
# Add a new dependency
poetry add package-name

# Add dev dependency
poetry add --group dev package-name

# Update dependencies
poetry update
```

## 🗺️ Roadmap

### Phase 3: Agent + MCP (Next)
- Query router for multi-intent handling
- ReAct reasoning engine
- Tool system for function calling
- MCP server for AI assistant integration

### Phase 4: Multi-LLM + API
- OpenAI, Anthropic, Ollama support
- LLM provider abstraction
- FastAPI REST endpoints
- Streaming responses

### Phase 5: Observability
- Arize Phoenix integration
- Ragas evaluation framework
- Prometheus metrics
- Performance benchmarks

### Phase 6: Deployment
- Docker Compose for development
- Kubernetes manifests
- CI/CD pipelines
- Production deployment guide

## 🤝 Contributing

Contributions are welcome! This is a portfolio project, but feedback and suggestions are appreciated.

### Development Workflow

1. Create a feature branch
2. Make your changes
3. Run tests and linting
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LlamaIndex](https://www.llamaindex.ai/) for RAG framework
- [Qdrant](https://qdrant.tech/) for vector database
- [sentence-transformers](https://www.sbert.net/) for reranking models
- [rank-bm25](https://github.com/dorianbrown/rank_bm25) for BM25 implementation

## 📬 Contact

For questions or discussions about this project, please open an issue on GitHub.

---

<p align="center">
  Built with ❤️ for the AI engineering community
  <br>
  <em>Currently in active development - Phase 2 Complete!</em>
</p>
