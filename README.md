# Code Indexer

A Python-based server for crawling and indexing source code into ChromaDB using LlamaIndex for document processing. The server provides a REST API for code indexing and semantic search capabilities.

## Features

- REST API server for code indexing and searching
- Automatic crawling of specified directories, ignoring common project directories
- Intelligent code chunking using LlamaIndex's `CodeSplitter`
- Semantic search using ChromaDB with HuggingFace embeddings
- Docker-based deployment for easy setup
- Support for multiple project collections
- Real-time indexing with configurable intervals

## Installation

### Using Docker (Recommended)

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd code-indexer
   ```

2. Create a `.env` file with your configuration:
   ```env
   PROJECTS_ROOT=/projects
   FOLDERS_TO_INDEX=project1,project2
   ```

3. Run using docker-compose:
   ```bash
   docker-compose up -d
   ```

The server will start on port 8000 and begin indexing the specified projects.

### Manual Installation

1. Setup a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the server:
   ```bash
   python code_indexer_server.py
   ```

## API Endpoints

### GET /status
Get the current status of the indexing process and configuration.

### GET /search
Search for code using natural language queries.

Parameters:
- `query`: Natural language query about the codebase
- `project`: Collection/folder name to search in
- `n_results`: Number of results to return (default: 5)
- `threshold`: Minimum relevance percentage (default: 30.0)

Example:
```bash
curl "http://localhost:8000/search?query=find all database connections&project=my_project"
```

Response:
```json
{
    "results": [
        {
            "text": "code chunk content...",
            "file_path": "/path/to/file.py",
            "language": "python",
            "start_line": 10,
            "end_line": 20,
            "relevance": 85.5
        }
    ],
    "total_results": 1
}
```

## Configuration

The server can be configured through environment variables:

- `PROJECTS_ROOT`: Root directory containing projects to index (default: "/projects")
- `FOLDERS_TO_INDEX`: Comma-separated list of folders to index (default: all folders in PROJECTS_ROOT)
- `IGNORE_DIRS`: Comma-separated list of directories to ignore (default: common build/cache directories)
- `FILE_EXTENSIONS`: Comma-separated list of file extensions to include (default: common code file extensions)

## Docker Setup

The project includes a `Dockerfile` and `docker-compose.yml` for containerized deployment:

```yaml
version: '3'
services:
  code-indexer:
    build: .
    volumes:
      - ./projects:/projects
      - ./chroma_db:/app/chroma_db
    environment:
      - PROJECTS_ROOT=/projects
      - FOLDERS_TO_INDEX=project1,project2
    ports:
      - "8000:8000"
```

## Development

To modify the code or add new features:

1. Make your changes to the source code
2. Rebuild the Docker container:
   ```bash
   docker-compose build
   docker-compose up -d
   ```

## License

[Your License Here] 