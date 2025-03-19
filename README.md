# Local Code Indexing for Cursor

An experimental Python-based server that **locally** indexes codebases using ChromaDB and provides a semantic search tool via an MCP (Model Context Protocol) server for tools like Cursor.

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
         "url": "http://localhost:8978/sse"
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