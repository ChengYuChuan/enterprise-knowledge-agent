# 🏢 Enterprise Knowledge Agent Platform

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> An AI-powered enterprise knowledge management system built with modern Agent architecture, featuring MCP protocol support, multi-LLM integration, and production-ready deployment.

## 🚧 Project Status

Currently in **Phase 1: Basic RAG Pipeline** development.

- [x] Project initialization
- [x] Document loaders (PDF, MD, TXT)
- [x] Chunking strategies
- [x] Qdrant integration
- [x] Basic retrieval
- [x] CLI testing tool

## ✨ Planned Features

| Feature | Description |
|---------|-------------|
| 🤖 **AI Agent Architecture** | ReAct-based reasoning with intelligent tool calling |
| 🔌 **MCP Protocol Support** | Model Context Protocol for AI integration |
| 📚 **Advanced RAG Pipeline** | Hybrid search (Vector + BM25) with reranking |
| 📊 **Multi-LLM Support** | OpenAI, Anthropic Claude, and Ollama |
| ☁️ **Cloud-Ready** | Docker and Kubernetes deployment |

## 🚀 Quick Start (Coming Soon)

### Prerequisites

- Python 3.11+
- Poetry
- Docker & Docker Compose (for Qdrant)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/enterprise-knowledge-agent.git
cd enterprise-knowledge-agent

# Install dependencies using Poetry
poetry install

# Copy environment variables
cp .env.example .env

# Start Qdrant (vector database)
docker run -p 6333:6333 qdrant/qdrant
```

## 📖 Documentation

Documentation will be available in the `docs/` directory:

- Architecture Design
- Framework Evaluation
- Best Practices
- API Reference
- Deployment Guide

## 🛠️ Tech Stack

### Frameworks
- **RAG**: LlamaIndex - Document ingestion and retrieval
- **Agent**: LangChain - Agent orchestration (coming in Phase 3)
- **MCP**: FastMCP - Model Context Protocol (coming in Phase 3)
- **API**: FastAPI - REST API (coming in Phase 4)

### Infrastructure
- **Vector DB**: Qdrant
- **Cache**: Redis (coming in Phase 4)
- **Database**: PostgreSQL (coming in Phase 4)

## 📁 Project Structure

```
enterprise-knowledge-agent/
├── src/
│   ├── core/              # Shared business logic
│   ├── rag/               # RAG pipeline
│   │   ├── ingestion/     # Document loading & chunking
│   │   ├── retrieval/     # Search & ranking
│   │   └── generation/    # Response synthesis
│   ├── agent/             # Agent orchestrator (Phase 3)
│   ├── mcp_server/        # MCP protocol (Phase 3)
│   └── api/               # FastAPI routes (Phase 4)
├── configs/               # Configuration files
├── tests/                 # Unit, integration, e2e tests
├── examples/              # Sample documents & notebooks
└── docs/                  # Documentation
```

## 🧪 Development

```bash
# Run tests
poetry run pytest

# Run linting
poetry run ruff check .
poetry run black --check .

# Run type checking
poetry run mypy src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LlamaIndex](https://www.llamaindex.ai/) and [LangChain](https://www.langchain.com/) communities
- [Anthropic](https://www.anthropic.com/) for the MCP specification

---

<p align="center">
  Built with ❤️ for the AI engineering community
</p>
