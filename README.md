# AI Copilot

An interactive RAG-powered chatbot for engineering documentation built on Strands AI Agent SDK, AWS Bedrock, and FAISS vector search. The application serves as an AI-powered assistant capable of answering questions about engine troubleshooting, configuration parameters, and maintenance procedures.

## Features

### Core Technologies

- **[Strands AI Agent SDK](https://strandsagents.com/)** - Modern AI agent framework with A2A protocol support
- **[Amazon Bedrock](https://aws.amazon.com/)** - Serverless AI (Nova Lite model for LLM inference)
- **[FAISS](https://github.com/facebookresearch/faiss)** - Vector similarity search for RAG retrieval
- **[OpenAI Embeddings](https://platform.openai.com/)** - text-embedding-3-small (1536D) for semantic search
- **[FastAPI](https://fastapi.tiangolo.com/)** - High-performance async web framework
- **[Streamlit](https://streamlit.io/)** - Interactive web UI for chatbot interface

### Key Capabilities

- **RAG (Retrieval-Augmented Generation)** - Searches 8,524+ engineering document chunks
- **Real-time Streaming** - Live AI responses via A2A protocol
- **Source Citations** - Automatic filename and page number references
- **Conversational AI** - Natural language understanding with clarifying questions
- **Tool-based Retrieval** - Agent autonomously decides when to search documentation
- **High-Contrast UI** - Professional Streamlit interface with WCAG AA compliance

## Quick Start

### Prerequisites

- Python 3.12+ with Anaconda
- OpenAI API key (for embeddings)
- AWS credentials with Bedrock access (profile: `bai-core-gbl-ai-ai_developer`)

### Setup (2 Simple Steps!)

**Step 1: Start the A2A Server**

```bash
./run_server.sh
```

The script will:
- ✅ Load environment from `.env` file
- ✅ Verify AWS credentials
- ✅ Check all required packages
- ✅ Start server on http://localhost:8080
- ✅ Load RAG index (8,524 document vectors)

**Step 2: Launch the Chatbot UI**

In a new terminal:

```bash
./run_chatbot.sh
```

The Streamlit UI will open at http://localhost:8501

### Configuration

All credentials are stored in `.env` (already configured):

```bash
export OPENAI_API_KEY="your-key-here"
export AWS_PROFILE="bai-core-gbl-ai-ai_developer"
export CONFIG_DIR="./config"
```

This file is in `.gitignore` and won't be committed to git.

## Usage Examples

### Example Queries

- "What are the daily maintenance procedures for ASC systems?"
- "How do I troubleshoot engine RPM issues?"
- "What does error code XYZ mean?"
- "Explain the voltage reading configuration parameters"
- "What's the proper air gap setting for speed sensors?"

### Expected Behavior

The chatbot will:
1. Search the documentation database using FAISS
2. Retrieve the top 5 most relevant chunks
3. Provide an answer based on retrieved content
4. Cite sources with filename and page numbers
5. Ask clarifying questions if needed

## Architecture

```
User Query → Streamlit UI → A2A Server → Strands Agent
                                              ↓
                                    ┌─────────┴─────────┐
                                    ↓                   ↓
                            RAG Service          Bedrock LLM
                            (FAISS + OpenAI)     (Nova Lite)
                                    ↓
                            8,524 Document Chunks
```

### System Configuration

- **LLM Model**: AWS Bedrock Nova Lite (us.amazon.nova-lite-v1:0)
- **Vector Database**: FAISS IndexFlatIP (cosine similarity)
- **Embeddings**: OpenAI text-embedding-3-small (1536D)
- **Document Count**: 8,524 chunks
- **Top-K Retrieval**: 5 documents per query
- **Region**: us-west-2
- **Server Port**: 8080
- **UI Port**: 8501

## Project Structure

```
.
├── config/
│   └── settings.toml          # Agent & RAG configuration
├── data/
│   └── rag_index/             # FAISS index + document chunks
├── src/
│   └── brightai/
│       └── ai_copilot/
│           ├── services/
│           │   ├── rag_service.py      # RAG retrieval logic
│           │   └── agent_service.py    # Strands agent setup
│           └── core/
│               └── config.py           # Configuration schema
├── scripts/
│   └── streamlit.py           # Chatbot UI
├── .env                       # Environment variables (gitignored)
├── run_server.sh              # Server launcher
├── run_chatbot.sh             # Chatbot UI launcher
├── test_rag_retrieval.py      # RAG testing script
└── RAG_IMPLEMENTATION.md      # Detailed technical docs
```

## Testing

### Test RAG Retrieval

```bash
python3 test_rag_retrieval.py
```

Expected output:
```
✅ OpenAI API is working! Embedding dimension: 1536
✅ FAISS index loaded: 8524 vectors
✅ Retrieved 3 chunks with citations
```

### Test Server Health

```bash
curl http://localhost:8080/health
```

Expected response:
```json
{"status":"healthy","service":"ai-copilot"}
```

### Test End-to-End

```bash
python3 test_chatbot_e2e.py "What are the ASC maintenance procedures?"
```

## Configuration

### Agent Settings (`config/settings.toml`)

```toml
[app.agent]
name = "AI Copilot"
bedrock_model_id = "us.amazon.nova-lite-v1:0"
description = "An AI Agent capable of assisting with problems that can be solved by referencing engineering documentation"
system_prompt = """
You are a knowledgeable assistant helping with engineering documentation.
You are providing information about engine troubleshooting, configuration parameters, and error codes.
You should use a conversational style and ask clarifying questions if needed.
"""
```

### RAG Settings

```toml
[app.rag]
enabled = true
index_dir = "data/rag_index"
top_k = 5
embedding_provider = "openai"
embedding_model = "text-embedding-3-small"
similarity_threshold = 0.0
```

## Troubleshooting

### Port Already in Use

```bash
lsof -ti:8080 | xargs kill -9
./run_server.sh
```

### AWS Authentication Failed

```bash
gimme-aws-creds
export AWS_PROFILE=bai-core-gbl-ai-ai_developer
aws sts get-caller-identity
```

### Missing Packages

```bash
pip install openai faiss-cpu boto3 pandas pyarrow structlog uvicorn streamlit
```

### RAG Not Finding Documents

Check server logs for:
```
[info] Loaded FAISS index with 8524 vectors
[info] RAG service initialization completed
```

If missing, verify:
1. `data/rag_index/` contains: index.faiss, chunks.parquet, meta.json
2. OPENAI_API_KEY is set in `.env`
3. Server logs show no errors during startup

## Development

### Original Setup (for reference)

The project can also be run using the original `uv` and `just` setup:

**Prerequisites:**
- [`uv`](https://docs.astral.sh/uv/) >= v0.9.0
- [`just`](https://just.systems/man/en/) >= 1.42.4

**Commands:**
```bash
just deps        # Install dependencies
just dev         # Run development server
just test        # Run tests
```

**Note:** CodeArtifact authentication is required for this approach.

### Simplified Setup (recommended)

The simplified setup bypasses CodeArtifact and uses system Python packages:

```bash
./run_server.sh   # Uses .env and system packages
./run_chatbot.sh  # Launches Streamlit UI
```

This is the recommended approach for local development.

## Deployment

The application is container-ready and can be deployed to AWS EKS using the existing Terraform configuration in the `terraform/` directory.

**Deployment files:**
- `Dockerfile` - Container definition
- `terraform/` - Infrastructure as code
- `.deploy/` - Deployment configurations
- `.github/workflows/` - CI/CD pipelines

## Documentation

- **[RAG_IMPLEMENTATION.md](RAG_IMPLEMENTATION.md)** - Comprehensive technical implementation details
- **API Docs** - http://localhost:8080/docs (when server is running)
- **Health Check** - http://localhost:8080/health

## Support

For issues or questions:
- Check [RAG_IMPLEMENTATION.md](RAG_IMPLEMENTATION.md) for technical details
- Review server logs for error messages
- Test individual components with test scripts

## License

Internal BrightAI project - All rights reserved

---

**Status:** ✅ Fully operational
**Last Updated:** 2026-01-12
**Version:** 1.0.0 (RAG-enabled)
