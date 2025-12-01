# Enterprise Knowledge Agent Platform

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![MCP Compatible](https://img.shields.io/badge/MCP-Compatible-green.svg)](https://modelcontextprotocol.io/)

> A production-ready enterprise knowledge management system built with modern AI Agent architecture, featuring MCP protocol support, multi-LLM integration, hybrid RAG retrieval, and comprehensive observability.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
  - [CLI](#cli)
  - [REST API](#rest-api)
  - [MCP Server](#mcp-server)
- [Configuration](#configuration)
- [Development](#development)
- [Deployment](#deployment)
- [Testing](#testing)
- [Documentation](#documentation)
- [Roadmap](#roadmap)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

The Enterprise Knowledge Agent is an AI-powered system designed to help organizations manage and query their internal knowledge bases. It combines state-of-the-art retrieval techniques with intelligent agent capabilities to provide accurate, contextual responses with proper source citations.

### Key Capabilities

- **Intelligent Document Retrieval**: Hybrid search combining vector similarity and BM25 keyword matching
- **AI Agent with Reasoning**: ReAct-based agent that can plan multi-step queries
- **MCP Protocol Support**: Integrate with Claude Desktop, Cursor, and other MCP-compatible clients
- **Multi-LLM Flexibility**: Switch between OpenAI, Anthropic Claude, or local Ollama models
- **Production-Ready**: Complete with authentication, rate limiting, observability, and deployment configs

### Target Use Cases

- Enterprise knowledge base search
- Internal documentation Q&A
- HR policy inquiries
- Technical documentation assistance
- Customer support knowledge management

## Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                    Enterprise Knowledge Agent Platform                  │
├────────────────────────────────────────────────────────────────────────┤
│  Interface Layer                                                       │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐│
│  │ FastAPI REST    │  │  CLI Tool       │  │ MCP Server              ││
│  │ (Human/Systems) │  │  (Development)  │  │ (Claude, Cursor, etc.)  ││
│  └────────┬────────┘  └────────┬────────┘  └────────────┬────────────┘│
├───────────┴────────────────────┴────────────────────────┴─────────────┤
│  Agent Core Layer                                                      │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │                    Agent Orchestrator                           │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐    │   │
│  │  │Query Router  │→ │ReAct Engine  │→ │Response Synthesizer│    │   │
│  │  │(Intent)      │  │(Reasoning)   │  │(Citation + Format) │    │   │
│  │  └──────────────┘  └──────────────┘  └────────────────────┘    │   │
│  └────────────────────────────────────────────────────────────────┘   │
├───────────────────────────────────────────────────────────────────────┤
│  RAG Pipeline Layer                                                    │
│  ┌─────────────┐  ┌────────────────┐  ┌─────────────┐  ┌───────────┐  │
│  │  Ingestion  │  │ Hybrid Search  │  │  Reranker   │  │  Memory   │  │
│  │  Pipeline   │  │ Vector + BM25  │  │  (BGE)      │  │  Store    │  │
│  └─────────────┘  └────────────────┘  └─────────────┘  └───────────┘  │
├───────────────────────────────────────────────────────────────────────┤
│  Infrastructure Layer                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐  │
│  │  Qdrant     │  │  Redis      │  │  Postgres   │  │  S3/MinIO    │  │
│  │  (Vectors)  │  │  (Cache)    │  │  (Metadata) │  │  (Files)     │  │
│  └─────────────┘  └─────────────┘  └─────────────┘  └──────────────┘  │
├───────────────────────────────────────────────────────────────────────┤
│  Observability Layer                                                   │
│  ┌─────────────────────────┐  ┌────────────────────────────────────┐  │
│  │  Arize Phoenix          │  │  Prometheus + Grafana              │  │
│  │  (LLM Tracing)          │  │  (System Metrics)                  │  │
│  └─────────────────────────┘  └────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────┘
```

### Design Principles

The system follows the **Hexagonal Architecture (Ports & Adapters)** pattern:

- **Core Layer**: RAG pipeline and agent logic, independent of interfaces
- **Adapters**: FastAPI REST, MCP Server, CLI — all share the same core
- **Extensible**: Easy to add new LLM providers, tools, or interfaces

## Features

| Category | Feature | Description |
|----------|---------|-------------|
| **RAG Pipeline** | Multi-format Loading | PDF, Markdown, TXT document support |
| | Smart Chunking | Fixed, Sentence, and Semantic chunking strategies |
| | Hybrid Search | Vector similarity + BM25 with Reciprocal Rank Fusion |
| | Cross-encoder Reranking | BGE-reranker for improved precision |
| | Auto-Citations | Automatic source tracking and citation |
| **Agent** | Intent Classification | Query router for optimal response strategy |
| | ReAct Reasoning | Multi-step planning and execution |
| | Tool System | Extensible registry pattern for custom tools |
| **Interfaces** | REST API | Full FastAPI implementation with OpenAPI docs |
| | MCP Protocol | Compatible with Claude Desktop, Cursor, etc. |
| | Streaming | Server-Sent Events for real-time responses |
| **Multi-LLM** | OpenAI | GPT-4, GPT-4-turbo, GPT-3.5-turbo |
| | Anthropic | Claude 3 Opus, Sonnet, Haiku |
| | Ollama | Local models (Llama 3, Mistral, etc.) |
| **Security** | Authentication | API key and JWT Bearer token support |
| | Rate Limiting | Configurable per-endpoint limits |
| **Observability** | Tracing | Arize Phoenix for LLM call tracing |
| | Metrics | Prometheus metrics export |
| | Evaluation | Ragas framework integration |
| **Deployment** | Docker | Dockerfile and Docker Compose setup |
| | Kubernetes | Full manifest set for k8s deployment |
| | CI/CD | GitHub Actions workflows |

## Quick Start

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- OpenAI API key (or Anthropic/Ollama)

### 5-Minute Setup

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/enterprise-knowledge-agent.git
cd enterprise-knowledge-agent

# 2. Install dependencies
poetry install

# 3. Set up environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 4. Start infrastructure (Qdrant, Redis, PostgreSQL)
docker compose -f deployment/docker-compose/docker-compose.yml up -d qdrant redis

# 5. Ingest sample documents
poetry run python -m src.cli ingest examples/sample_documents/

# 6. Start the API server
poetry run uvicorn src.api.main:app --reload

# 7. Try a query
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-key" \
  -d '{"message": "What are the remote work guidelines?"}'
```

## Installation

### Using Poetry (Recommended)

```bash
# Install Poetry if you haven't
curl -sSL https://install.python-poetry.org | python3 -

# Install project dependencies
poetry install

# Activate virtual environment
poetry shell
```

### Using pip

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or .venv\Scripts\activate  # Windows

# Install dependencies
pip install -e .
```

### Infrastructure Setup

```bash
# Start all services with Docker Compose
docker compose -f deployment/docker-compose/docker-compose.yml up -d

# Or start individual services
docker run -d -p 6333:6333 qdrant/qdrant          # Vector DB
docker run -d -p 6379:6379 redis:alpine            # Cache
docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=password postgres:15  # Metadata
```

## Usage

### CLI

```bash
# Ingest documents
poetry run python -m src.cli ingest ./documents/

# Interactive query
poetry run python -m src.cli query "What is our vacation policy?"

# Get system status
poetry run python -m src.cli status
```

### REST API

Start the server:

```bash
poetry run uvicorn src.api.main:app --reload --port 8000
```

API Endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/chat` | Chat with knowledge base |
| `POST` | `/api/v1/chat/stream` | Streaming chat (SSE) |
| `POST` | `/api/v1/search` | Search documents |
| `POST` | `/api/v1/ingest` | Upload document |
| `GET` | `/api/v1/documents` | List documents |
| `GET` | `/health` | Health check |
| `GET` | `/metrics` | Prometheus metrics |

Example requests:

```bash
# Chat (non-streaming)
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "message": "How many vacation days do new employees get?",
    "top_k": 5
  }'

# Search documents
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "query": "remote work",
    "top_k": 10
  }'

# Ingest document
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "X-API-Key: your-api-key" \
  -F "file=@document.pdf"
```

Interactive docs available at: `http://localhost:8000/docs`

### MCP Server

The MCP server enables integration with Claude Desktop, Cursor, and other MCP-compatible clients.

#### Setup for Claude Desktop

Add to your Claude Desktop config (`~/.config/claude/mcp_servers.json`):

```json
{
  "knowledge-agent": {
    "command": "python",
    "args": ["-m", "src.mcp_server.server"],
    "cwd": "/path/to/enterprise-knowledge-agent",
    "env": {
      "OPENAI_API_KEY": "your-key"
    }
  }
}
```

#### Available MCP Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `query_knowledge_base` | Search and answer from knowledge base | `query: str`, `top_k: int` |
| `get_knowledge_base_stats` | Get collection statistics | — |
| `search_documents` | Find documents by metadata | `filename_pattern: str`, `file_type: str` |
| `agent_query` | Execute multi-step reasoning query | `query: str`, `max_iterations: int` |

#### Available MCP Resources

| Resource | Description |
|----------|-------------|
| `documents://list` | List all documents in knowledge base |
| `stats://knowledge-base` | Knowledge base statistics |

## Configuration

### Environment Variables

Create a `.env` file based on `.env.example`:

```bash
# LLM Providers
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
OLLAMA_BASE_URL=http://localhost:11434

# Default LLM Settings
LLM_PROVIDER=openai          # openai, anthropic, ollama
LLM_MODEL=gpt-4              # Model name
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=2048

# Vector Database
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=knowledge_base

# Cache & Database
REDIS_URL=redis://localhost:6379
POSTGRES_URL=postgresql://user:pass@localhost:5432/knowledge_agent

# Observability
PHOENIX_ENDPOINT=http://localhost:6006
PROMETHEUS_ENABLED=true

# API Settings
APP_ENV=development
LOG_LEVEL=INFO
```

### Configuration Files

| File | Purpose |
|------|---------|
| `configs/settings.yaml` | Main application settings |
| `configs/llm_configs/default.yaml` | LLM provider configuration |
| `configs/observability.yaml` | Tracing and metrics settings |

## Development

### Project Structure

```
enterprise-knowledge-agent/
├── src/
│   ├── agent/              # Agent orchestrator, router, ReAct engine
│   ├── api/                # FastAPI routes, middleware, schemas
│   ├── llm/                # LLM provider abstraction
│   ├── mcp_server/         # MCP protocol implementation
│   ├── observability/      # Tracing, metrics, evaluation
│   └── rag/                # RAG pipeline (ingestion, retrieval, generation)
├── configs/                # YAML configuration files
├── deployment/             # Docker, Kubernetes, scripts
├── docs/                   # Documentation
├── examples/               # Sample documents and notebooks
├── tests/                  # Unit, integration, e2e tests
└── scripts/                # Utility scripts
```

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

### Adding a New LLM Provider

1. Create provider class in `src/llm/providers/`:

```python
from src.llm.base import BaseLLMProvider, LLMConfig

class MyProvider(BaseLLMProvider):
    @property
    def provider_name(self) -> str:
        return "my_provider"
    
    async def generate(self, messages, **kwargs) -> LLMResponse:
        # Implementation
        pass
    
    async def generate_stream(self, messages, **kwargs):
        # Implementation
        pass
```

2. Register in `src/llm/factory.py`
3. Add configuration in `configs/llm_configs/`

### Adding a New MCP Tool

1. Create tool in `src/agent/tools/`:

```python
from src.agent.tools import BaseTool, ToolParameter, ToolResult

class MyTool(BaseTool):
    @property
    def name(self) -> str:
        return "my_tool"
    
    @property
    def description(self) -> str:
        return "Description for LLM"
    
    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(name="param1", type=str, required=True)
        ]
    
    async def execute(self, **kwargs) -> ToolResult:
        # Implementation
        return ToolResult(success=True, data={...})
```

2. Register in `src/agent/tools/__init__.py`
3. Add MCP wrapper in `src/mcp_server/server.py`

## Deployment

### Docker Compose (Development/Staging)

```bash
# Start all services
docker compose -f deployment/docker-compose/docker-compose.yml up -d

# View logs
docker compose -f deployment/docker-compose/docker-compose.yml logs -f app

# Stop services
docker compose -f deployment/docker-compose/docker-compose.yml down
```

### Kubernetes (Production)

```bash
# Apply manifests
kubectl apply -f deployment/kubernetes/

# Check status
kubectl get pods -l app=knowledge-agent

# View logs
kubectl logs -l app=knowledge-agent -f
```

### Environment-Specific Configurations

| Environment | Configuration |
|-------------|---------------|
| Development | Single replica, debug logging, hot reload |
| Staging | 2 replicas, info logging, integration tests |
| Production | 4+ replicas, warn logging, full monitoring |

## Testing

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=src --cov-report=html

# Run specific test categories
poetry run pytest tests/unit/           # Unit tests
poetry run pytest tests/integration/    # Integration tests
poetry run pytest tests/e2e/            # End-to-end tests

# Run MCP server tests
poetry run python scripts/test_mcp_server.py
```

### Test Categories

| Category | Description | Markers |
|----------|-------------|---------|
| Unit | Individual component tests | — |
| Integration | Multi-component tests | `@pytest.mark.integration` |
| E2E | Full system tests | `@pytest.mark.e2e` |
| Slow | Long-running tests | `@pytest.mark.slow` |

## Documentation

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | System architecture details |
| [FRAMEWORK_EVALUATION.md](docs/FRAMEWORK_EVALUATION.md) | LangChain vs LlamaIndex analysis |
| [BEST_PRACTICES.md](docs/BEST_PRACTICES.md) | AI Agent development guidelines |
| [API_REFERENCE.md](docs/API_REFERENCE.md) | Complete API documentation |
| [DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md) | Production deployment guide |

## Roadmap

### Completed

- [x] **Phase 1**: Basic RAG Pipeline — Document ingestion, chunking, vector search
- [x] **Phase 2**: Advanced Retrieval — Hybrid search, reranking, citations
- [x] **Phase 3**: Agent + MCP — Query router, ReAct engine, MCP server
- [x] **Phase 4**: Multi-LLM + API — Provider abstraction, FastAPI, streaming
- [x] **Phase 5**: Observability — Phoenix tracing, Prometheus metrics, Ragas evaluation
- [x] **Phase 6**: Deployment — Docker, Kubernetes, CI/CD

### Future Enhancements

- [ ] Multi-tenancy support
- [ ] Document versioning
- [ ] Advanced conversation memory
- [ ] Custom embedding model fine-tuning
- [ ] GraphRAG integration
- [ ] Voice interface support

## Performance Metrics

| Metric | Target | Achieved |
|--------|--------|----------|
| Query Latency (p95) | < 2s | ~660ms |
| Context Relevance | > 0.85 | Evaluation ready |
| Answer Faithfulness | > 0.90 | Evaluation ready |
| MCP Tool Success Rate | > 95% | 100% (6/6 tests) |

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [FastMCP](https://github.com/jlowin/fastmcp) — MCP implementation framework
- [LlamaIndex](https://www.llamaindex.ai/) — RAG pipeline inspiration
- [LangChain](https://langchain.com/) — Agent framework patterns
- [Qdrant](https://qdrant.tech/) — Vector database
- [Arize Phoenix](https://phoenix.arize.com/) — LLM observability
- [Ragas](https://docs.ragas.io/) — RAG evaluation framework
- [sentence-transformers](https://www.sbert.net/) — Reranking models
- [rank-bm25](https://github.com/dorianbrown/rank_bm25) — BM25 implementation

## Contact

For questions or discussions about this project, please open an issue on GitHub.

---

<p align="center">
  Built with ❤️ as a portfolio project demonstrating enterprise AI Agent development
  <br><br>
  <strong>All 6 phases complete! 🎉</strong>
</p>