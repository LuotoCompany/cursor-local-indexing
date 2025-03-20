# Local Code Indexing for AI Coding Assistants

An experimental Python-based server that **locally** indexes codebases using ChromaDB and provides a semantic search tool via an MCP (Model Context Protocol) server for tools like Cursor IDE and Claude Code CLI.

## Setup for Cursor IDE

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

## Using with Claude Code CLI

If you want to use this indexing server with Claude Code CLI instead of Cursor IDE, follow these steps:

1. Complete steps 1-4 from the setup instructions above to set up the indexing server.

2. Add the local search server to Claude Code:
   ```bash
   claude mcp add "workspace-code-search" http://localhost:8978/sse
   ```

3. Verify the MCP server was added:
   ```bash
   claude mcp list
   ```

4. Start using Claude Code CLI with enhanced local code search capabilities:
   ```bash
   claude
   ```

5. Inside the Claude Code session, you can now use the semantic search functionality:
   ```
   Search my codebase for implementations of user authentication
   ```

Claude Code will now be able to search your codebase using the semantic search capabilities provided by the indexing server.

## How It Works

This project creates a vector database of your code using ChromaDB and exposes a semantic search tool via the MCP protocol. The `search_code` tool allows querying your codebase using natural language.

When integrated with either Cursor IDE or Claude Code CLI, the AI assistant gains the ability to search through your codebase semantically, finding relevant code snippets that match the intent of your queries rather than just matching keywords.

## Troubleshooting

- If you're having connection issues, make sure the indexing server is running: `docker-compose ps`
- Check the logs for any errors: `docker-compose logs code-indexer`
- Verify your projects are being indexed correctly: `docker-compose logs -f code-indexer`
- If using Claude Code CLI, ensure the MCP server has been added correctly: `claude mcp get workspace-code-search`

### Handling Symlinks

The indexer is configured to skip symbolic links during the indexing process. This prevents issues that can occur when:

- Absolute symlinks in the host system point to paths that don't exist in the container
- Symbolic links create directory cycles that could cause infinite recursion
- Links point to files outside the intended indexing scope

This behavior is intentional to ensure stable and predictable indexing. If you need to include content that is linked via symlinks, consider:

1. Copying the actual content to a regular directory within your project
2. Adjusting your project structure to avoid symlinks for critical code
3. Adding the symlink targets directly to your `FOLDERS_TO_INDEX` configuration