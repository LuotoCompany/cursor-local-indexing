# Local Code Indexing for Cursor

A Python-based server that provides semantic code search capabilities for your local projects through Cursor IDE. It indexes your source code into ChromaDB using LlamaIndex for document processing and exposes a REST API for semantic search.

## Setup

1. Clone and enter the repository:
   ```bash
   git clone <repository-url>
   cd cursor-local-indexing
   ```

2. Create a `.env` file by copying `.env.example`:
   ```bash
   cp .env.example .env
   ```

3. Configure your `.env` file:
   ```env
   PROJECTS_ROOT=~/your/projects/root    # Path to your projects directory
   FOLDERS_TO_INDEX=project1,project2    # Comma-separated list of folders to index
   ```

   Example:
   ```env
   PROJECTS_ROOT=~/projects
   FOLDERS_TO_INDEX=project1,project2
   ```

4. Start the indexing server:
   ```bash
   docker-compose up -d
   ```

5. Configure Cursor to use the local search server:
   Create or edit `~/.cursor/mcp.json`:
   ```json
   {
     "mcpServers": {
       "workspace-code-search": {
         "url": "http://localhost:8000/sse"
       }
     }
   }
   ```

6. Restart Cursor IDE to apply the changes.

The server will start indexing your specified projects, and you'll be able to use semantic code search within Cursor when those projects are active.

7. Open a project that you configured as indexed.

Create a `.cursorrules` file and add the following:
```
<instructions>
For any request, use the @search_code tool to check what the code does.
Prefer that first before resorting to command line grepping etc.
</instructions>
```

8. Start using the Cursor Agent mode and see it doing local vector searches!