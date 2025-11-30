# 🏢 Enterprise Knowledge Agent Platform

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> An AI-powered enterprise knowledge management system built with modern Agent architecture, featuring MCP protocol support, multi-LLM integration, and production-ready deployment.

## 🚧 Project Status

Currently in **Phase 3: Agent + MCP** - Complete ✅

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

**Phase 3: Agent + MCP** ✅
- [x] Query router for intent classification
- [x] ReAct reasoning engine with multi-step planning
- [x] Tool system with registry pattern
- [x] MCP server implementation (FastMCP)
- [x] 4 MCP tools + 2 resources
- [x] Integration testing suite

### Coming Next

**Phase 4: Multi-LLM + API Layer** (Next)
- [ ] LLM provider abstraction layer
- [ ] OpenAI, Anthropic, Ollama integration
- [ ] FastAPI REST endpoints
- [ ] Authentication middleware
- [ ] API documentation

## ✨ Current Features

| Feature | Description | Status |
|---------|-------------|--------|
| 📚 **Multi-format Loading** | PDF, Markdown, TXT support | ✅ Complete |
| ✂️ **Smart Chunking** | Fixed, Sentence, Semantic strategies | ✅ Complete |
| 🔍 **Hybrid Search** | Vector + BM25 with RRF fusion | ✅ Complete |
| 🎯 **Reranking** | Cross-encoder reranking | ✅ Complete |
| 📝 **Auto-Citations** | Automatic source tracking | ✅ Complete |
| 🤖 **AI Agent** | ReAct-based reasoning | ✅ Complete |
| 🔌 **MCP Protocol** | Model Context Protocol | ✅ Complete |
| 🛠️ **Tool System** | Extensible tool registry | ✅ Complete |
| 📊 **Multi-LLM** | OpenAI, Anthropic, Ollama | 🔄 Phase 4 |
| 🌐 **REST API** | FastAPI endpoints | 🔄 Phase 4 |
| ☁️ **Cloud-Ready** | Docker, Kubernetes | 🔄 Phase 6 |

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Poetry (for dependency management)
- Docker (for Qdrant vector database)
- OpenAI API key (for embeddings)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/enterprise-knowledge-agent.git
cd enterprise-knowledge-agent

# Install dependencies using Poetry
poetry install

# Copy environment variables
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=sk-...

# Start Qdrant vector database
docker run -p 6333:6333 qdrant/qdrant
```

### Basic Usage

#### 1. Initialize and Ingest Documents

```bash
# Reset and populate knowledge base with sample documents
poetry run python src/cli.py reset

# Or ingest individual documents
poetry run python src/cli.py ingest examples/sample_documents/hr_policies/vacation_policy.md
```

#### 2. Query the Knowledge Base

```bash
# Simple query
poetry run python src/cli.py query "What is the vacation policy?"

# Using hybrid search with reranking
poetry run python src/cli.py query "remote work guidelines" --use-hybrid --rerank
```

#### 3. Test MCP Server

```bash
# Run comprehensive MCP server tests
poetry run python scripts/test_mcp_server.py
```

#### 4. Use with Claude Desktop (MCP Integration)

Add to your Claude Desktop configuration (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "enterprise-knowledge": {
      "command": "poetry",
      "args": ["run", "python", "-m", "src.mcp_server.server"],
      "cwd": "/path/to/enterprise-knowledge-agent"
    }
  }
}
```

Then ask Claude: "What documents are in the knowledge base?" or "Query the knowledge base about vacation policies"

## 🏗️ Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────┐
│                   MCP Server Layer                       │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Tools: query_kb, get_stats, search, agent_query│   │
│  │  Resources: documents://list, stats://kb        │   │
│  └────────────────────┬─────────────────────────────┘   │
├────────────────────────┼─────────────────────────────────┤
│              Agent Orchestrator                          │
│  ┌────────────┬──────▼────────┬────────────────────┐   │
│  │Query Router│ ReAct Engine  │ Tool Executor      │   │
│  │(Intent)    │ (Reasoning)   │ (Actions)          │   │
│  └────────────┴───────────────┴────────────────────┘   │
├──────────────────────────────────────────────────────────┤
│                 RAG Pipeline                             │
│  ┌─────────────┬────────────────┬──────────────────┐   │
│  │ Ingestion   │ Hybrid Search  │ Reranker         │   │
│  │ (Chunking)  │ (Vector + BM25)│ (Cross-encoder)  │   │
│  └─────────────┴────────────────┴──────────────────┘   │
├──────────────────────────────────────────────────────────┤
│                Infrastructure                            │
│  ┌─────────────────────────────────────────────────┐   │
│  │ Qdrant (Vector DB) + OpenAI (Embeddings)        │   │
│  └─────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
```

### Project Structure

```
enterprise-knowledge-agent/
├── src/
│   ├── config.py              # Configuration management
│   ├── cli.py                 # Command-line interface
│   │
│   ├── rag/                   # ✅ RAG Pipeline
│   │   ├── ingestion/         # Document loading & chunking
│   │   ├── retrieval/         # Hybrid search & reranking
│   │   └── generation/        # Response synthesis
│   │
│   ├── agent/                 # ✅ Agent System
│   │   ├── router.py          # Query routing
│   │   ├── react/             # ReAct reasoning engine
│   │   └── tools/             # Tool definitions & registry
│   │
│   ├── mcp_server/            # ✅ MCP Server
│   │   └── server.py          # FastMCP implementation
│   │
│   ├── llm/                   # 🔄 LLM providers (Phase 4)
│   └── api/                   # 🔄 FastAPI (Phase 4)
│
├── tests/
│   ├── unit/                  # Unit tests
│   └── integration/           # Integration tests
│
├── scripts/
│   └── test_mcp_server.py     # MCP server test suite
│
├── examples/
│   └── sample_documents/      # Test documents
│
└── docs/                      # Documentation
```

## 🧪 Testing

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=src

# Run MCP server tests
poetry run python scripts/test_mcp_server.py

# Run specific test categories
poetry run pytest tests/unit/
poetry run pytest tests/integration/
```

### Test Results (Phase 3)

```
✅ All 6/6 MCP Server Tests Passed
  ✓ Server initialization
  ✓ Knowledge base stats tool
  ✓ Query knowledge base tool
  ✓ Search documents tool
  ✓ Agent query with ReAct engine
  ✓ MCP resources
```

## 🤖 MCP Tools & Resources

### Available Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `query_knowledge_base` | Search knowledge base with citations | `query: str`, `top_k: int` |
| `get_knowledge_base_stats` | Get collection statistics | None |
| `search_documents` | Find documents by metadata | `filename_pattern: str`, `file_type: str` |
| `agent_query` | Execute multi-step reasoning query | `query: str`, `max_iterations: int` |

### Available Resources

| Resource | Description |
|----------|-------------|
| `documents://list` | List all documents in knowledge base |
| `stats://knowledge-base` | Knowledge base statistics |

## 📊 Performance Metrics

| Metric | Target | Current Status |
|--------|--------|---------------|
| **Context Relevance** | > 0.85 | 🔄 Phase 5 (evaluation) |
| **Answer Faithfulness** | > 0.90 | 🔄 Phase 5 (evaluation) |
| **Answer Relevance** | > 0.85 | 🔄 Phase 5 (evaluation) |
| **Query Latency (p95)** | < 2s | ✅ ~660ms (agent query) |
| **MCP Tool Success Rate** | > 95% | ✅ 100% (6/6 tests) |

## 📖 Documentation

Comprehensive documentation available in `docs/`:

- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - System design and component overview
- **[FRAMEWORK_EVALUATION.md](docs/FRAMEWORK_EVALUATION.md)** - LangChain vs LlamaIndex analysis
- **[BEST_PRACTICES.md](docs/BEST_PRACTICES.md)** - AI Agent development guidelines
- **[PROJECT_BRIEF.md](docs/PROJECT_BRIEF.md)** - Complete development roadmap

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

### Adding New Tools

1. Create tool class inheriting from `BaseTool` in `src/agent/tools/`
2. Implement `name`, `description`, `parameters`, and `execute()` methods
3. Register in `get_default_tools()` in `src/agent/tools/__init__.py`
4. Add MCP wrapper in `src/mcp_server/server.py`
5. Write tests in `tests/unit/test_agent_tools.py`

Example:
```python
from src.agent.tools import BaseTool, ToolParameter, ToolResult

class MyCustomTool(BaseTool):
    @property
    def name(self) -> str:
        return "my_custom_tool"
    
    @property
    def description(self) -> str:
        return "Description of what this tool does"
    
    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="param1",
                type=str,
                description="Parameter description",
                required=True
            )
        ]
    
    async def execute(self, param1: str) -> ToolResult:
        # Your implementation here
        return ToolResult(success=True, data={"result": "value"})
```

## 🗺️ Roadmap

### ✅ Phase 1: Basic RAG Pipeline (Week 1) - Complete
- Document ingestion with multiple formats
- Chunking strategies
- Vector search foundation

### ✅ Phase 2: Advanced Retrieval (Week 2) - Complete
- Hybrid search (Vector + BM25)
- Cross-encoder reranking
- Response synthesis with citations

### ✅ Phase 3: Agent + MCP Server (Week 3) - Complete
- Query router with intent classification
- ReAct reasoning engine
- Tool system with extensible registry
- MCP protocol implementation

### 🔄 Phase 4: Multi-LLM + API Layer (Week 3-4) - Next
- LLM provider abstraction
- OpenAI, Anthropic, Ollama support
- FastAPI REST endpoints
- Streaming responses
- Authentication

### 📋 Phase 5: Observability + Evaluation (Week 4)
- Arize Phoenix integration
- Ragas evaluation framework
- Prometheus metrics
- Performance benchmarks

### 📋 Phase 6: Deployment (Week 4-5)
- Docker Compose setup
- Kubernetes manifests
- CI/CD pipelines
- Production deployment guide

## 🤝 Contributing

This is a portfolio project, but feedback and suggestions are welcome!

### Development Workflow

1. Create a feature branch
2. Make your changes with type hints and docstrings
3. Run tests and linting
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [FastMCP](https://github.com/jlowin/fastmcp) for MCP implementation
- [LlamaIndex](https://www.llamaindex.ai/) for RAG inspiration
- [Qdrant](https://qdrant.tech/) for vector database
- [sentence-transformers](https://www.sbert.net/) for reranking models
- [rank-bm25](https://github.com/dorianbrown/rank_bm25) for BM25 implementation

## 📬 Contact

For questions or discussions about this project, please open an issue on GitHub.

---

<p align="center">
  Built with ❤️ for the AI engineering community
  <br>
  <em>Currently in active development - Phase 3 Complete! 🎉</em>
</p>