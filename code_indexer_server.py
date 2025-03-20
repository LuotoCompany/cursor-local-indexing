#!/usr/bin/env python3

import os
import logging
from typing import List, Set
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from tree_sitter_language_pack import get_parser
import asyncio
from fastmcp import FastMCP
from contextlib import asynccontextmanager

# Import LlamaIndex components
try:
    from llama_index.core import Document
    from llama_index.core.node_parser import CodeSplitter
    from llama_index.core import SimpleDirectoryReader
    print("LlamaIndex dependencies found.")
except ImportError as e:
    print(f"Error: {e}")
    print("Please install the required dependencies:")
    print("pip install llama-index llama-index-readers-file "
          "llama-index-embeddings-huggingface")
    exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default directories to ignore
DEFAULT_IGNORE_DIRS = {
    "__pycache__",
    "node_modules",
    ".git",
    "build",
    "dist",
    ".venv",
    "venv",
    "env",
    ".pytest_cache",
    ".ipynb_checkpoints"
}

# Default files to ignore
DEFAULT_IGNORE_FILES = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
    "Pipfile.lock",
    "Gemfile.lock",
    "composer.lock",
    ".DS_Store",
    ".env",
    ".env.local",
    ".env.development",
    ".env.test",
    ".env.production",
    "*.pyc",
    "*.pyo",
    "*.pyd",
    "*.so",
    "*.dll",
    "*.dylib",
    ".coverage",
    "coverage.xml",
    ".eslintcache",
    ".tsbuildinfo"
}

# Default file extensions to include
DEFAULT_FILE_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".c", ".cpp", ".h", ".hpp",
    ".cs", ".go", ".rb", ".php", ".swift", ".kt", ".rs", ".scala", ".sh",
    ".html", ".css", ".sql", ".md", ".json", ".yaml", ".yml", ".toml"
}

# Global variables
config = None
chroma_client = None
embedding_function = None
mcp = FastMCP(title="Code Indexer Server")


def get_config_from_env():
    """Get configuration from environment variables."""
    projects_root = os.getenv("PROJECTS_ROOT", "/projects")
    folders_to_index = os.getenv("FOLDERS_TO_INDEX", "").split(",")
    folders_to_index = [f.strip() for f in folders_to_index if f.strip()]

    if not folders_to_index:
        logger.warning("No folders specified to index. Using root directory.")
        folders_to_index = [""]

    return {
        "projects_root": projects_root,
        "folders_to_index": folders_to_index,
        "ignore_dirs": list(DEFAULT_IGNORE_DIRS),
        "file_extensions": list(DEFAULT_FILE_EXTENSIONS)
    }


async def initialize_chromadb():
    """Initialize ChromaDB and embedding function asynchronously."""
    global config, chroma_client, embedding_function

    try:
        # Get configuration from environment
        config = get_config_from_env()
        logger.info("Configuration loaded successfully")

        # Initialize ChromaDB client with telemetry disabled
        chroma_client = chromadb.PersistentClient(
            path="chroma_db",
            settings=Settings(anonymized_telemetry=False)
        )
        logger.info("ChromaDB client initialized")

        # Initialize embedding function
        embedding_function = (
            embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
        )
        logger.info("Embedding function initialized")
        
        return True
    except Exception as e:
        logger.error(f"Error during initialization: {e}")
        # Still need to assign default values to global variables
        if config is None:
            config = {"projects_root": "", "folders_to_index": [""]}
        if chroma_client is None:
            # Create an empty client as a fallback
            try:
                chroma_client = chromadb.PersistentClient(path="chroma_db")
            except Exception as db_err:
                logger.error(
                    f"Failed to create fallback ChromaDB client: {db_err}"
                )
        if embedding_function is None:
            try:
                embedding_function = (
                    embedding_functions.SentenceTransformerEmbeddingFunction(
                        model_name="all-MiniLM-L6-v2"
                    )
                )
            except Exception as embed_err:
                logger.error(
                    f"Failed to create fallback embedding function: "
                    f"{embed_err}"
                )
        return False


def is_valid_file(
    file_path: str,
    ignore_dirs: Set[str],
    file_extensions: Set[str]
) -> bool:
    """Check if a file should be processed based on its path and extension."""
    # Check if path contains ignored directory
    parts = file_path.split(os.path.sep)
    for part in parts:
        if part in ignore_dirs:
            return False

    # Get file name and check against ignored files
    file_name = os.path.basename(file_path)

    # Check exact matches
    if file_name in DEFAULT_IGNORE_FILES:
        return False

    # Check wildcard patterns
    for pattern in DEFAULT_IGNORE_FILES:
        if pattern.startswith("*"):
            if file_name.endswith(pattern[1:]):
                return False

    # Check file extension
    _, ext = os.path.splitext(file_path)
    return ext.lower() in file_extensions if file_extensions else True


def load_documents(
    directory: str, 
    ignore_dirs: Set[str] = DEFAULT_IGNORE_DIRS,
    file_extensions: Set[str] = DEFAULT_FILE_EXTENSIONS
) -> List[Document]:
    """Load documents from a directory, filtering out ignored paths."""
    try:
        # Get all files recursively
        all_files = []
        for root, dirs, files in os.walk(directory):
            # Skip ignored directories
            dirs[:] = [
                d for d in dirs
                if d not in ignore_dirs and not d.startswith('.')
            ]

            for file in files:
                abs_file_path = os.path.join(root, file)
                if is_valid_file(abs_file_path, ignore_dirs, file_extensions):
                    # Calculate relative path from the directory being indexed
                    rel_file_path = os.path.relpath(abs_file_path, directory)
                    all_files.append((abs_file_path, rel_file_path))

        if not all_files:
            logger.warning(f"No valid files found in {directory}")
            return []

        # Load the filtered files using absolute paths for reading
        reader = SimpleDirectoryReader(
            input_files=[abs_path for abs_path, _ in all_files],
            exclude_hidden=True
        )
        documents = reader.load_data()

        # Update the metadata to use relative paths
        for doc, (_, rel_path) in zip(documents, all_files):
            doc.metadata["file_path"] = rel_path

        logger.info(f"Loaded {len(documents)} documents from {directory}")
        return documents
    except Exception as e:
        logger.error(f"Error loading documents: {e}")
        return []


def process_and_index_documents(
    documents: List[Document],
    collection_name: str,
    persist_directory: str
) -> None:
    """Process documents with CodeSplitter and index them in ChromaDB."""
    if not documents:
        logger.warning("No documents to process.")
        return

    try:
        # Try to get collection if it exists or create a new one
        collection = chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"}
        )
    except Exception as e:
        logger.error(f"Error creating collection: {e}")
        return

    # Process each document
    total_nodes = 0
    for doc in documents:
        try:
            # Extract file path from metadata
            file_path = doc.metadata.get("file_path", "unknown")
            file_name = os.path.basename(file_path)

            # Determine language from file extension
            _, ext = os.path.splitext(file_name)
            language = ext[1:] if ext else "text"  # Remove the dot

            # Handle Markdown and other text files differently
            code_file_extensions = [
                "py", "python", "js", "jsx", "ts", "tsx", "java", "c", 
                "cpp", "h", "hpp", "cs", "go", "rb", "php", "swift", 
                "kt", "rs", "scala"
            ]

            if language in code_file_extensions:
                # Determine parser language based on file extension
                parser_language = "python"  # Default fallback
                if language in ["py", "python"]:
                    parser_language = "python"
                elif language in ["js", "jsx", "ts", "tsx"]:
                    parser_language = "javascript"
                elif language in ["java"]:
                    parser_language = "java"
                elif language in ["c", "cpp", "h", "hpp"]:
                    parser_language = "cpp"
                elif language in ["cs"]:
                    parser_language = "csharp"
                elif language in ["go"]:
                    parser_language = "go"
                elif language in ["rb"]:
                    parser_language = "ruby"
                elif language in ["php"]:
                    parser_language = "php"
                elif language in ["swift"]:
                    parser_language = "swift"
                elif language in ["kt"]:
                    parser_language = "kotlin"
                elif language in ["rs"]:
                    parser_language = "rust"
                elif language in ["scala"]:
                    parser_language = "scala"

                # Create parser and splitter for this specific language
                try:
                    code_parser = get_parser(parser_language)
                    splitter = CodeSplitter(
                        language=parser_language,
                        chunk_lines=40,
                        chunk_lines_overlap=15,
                        max_chars=1500,
                        parser=code_parser
                    )
                    nodes = splitter.get_nodes_from_documents([doc])
                except Exception as e:
                    logger.warning(
                        f"Could not create parser for {parser_language}, "
                        f"falling back to text-based splitting: {e}"
                    )
                    # Fall back to text-based splitting
                    nodes = []
                    lines = doc.text.split("\n")
                    chunk_size = 40
                    overlap = 15

                    for i in range(0, len(lines), chunk_size - overlap):
                        start_idx = i
                        end_idx = min(i + chunk_size, len(lines))

                        if start_idx >= len(lines):
                            continue

                        chunk_text = "\n".join(lines[start_idx:end_idx])

                        if not chunk_text.strip():
                            continue

                        from llama_index.core.schema import TextNode
                        node = TextNode(
                            text=chunk_text,
                            metadata={
                                "start_line_number": start_idx + 1,
                                "end_line_number": end_idx,
                                "file_path": file_path,
                                "file_name": file_name,
                            }
                        )
                        nodes.append(node)
            else:
                # For non-code files, manually split by lines
                nodes = []
                lines = doc.text.split("\n")
                chunk_size = 40
                overlap = 15

                for i in range(0, len(lines), chunk_size - overlap):
                    start_idx = i
                    end_idx = min(i + chunk_size, len(lines))

                    if start_idx >= len(lines):
                        continue

                    chunk_text = "\n".join(lines[start_idx:end_idx])

                    if not chunk_text.strip():
                        continue

                    from llama_index.core.schema import TextNode
                    node = TextNode(
                        text=chunk_text,
                        metadata={
                            "start_line_number": start_idx + 1,
                            "end_line_number": end_idx,
                            "file_path": file_path,
                            "file_name": file_name,
                        }
                    )
                    nodes.append(node)

            if not nodes:
                logger.warning(f"No nodes generated for {file_path}")
                continue

            logger.info(f"Processing {file_path}: {len(nodes)} chunks")

            # Prepare data for ChromaDB
            ids = []
            texts = []
            metadatas = []

            for i, node in enumerate(nodes):
                start_line = node.metadata.get("start_line_number", 0)
                end_line = node.metadata.get("end_line_number", 0)

                if start_line == 0 or end_line == 0:
                    start_line = 1
                    end_line = len(node.text.split("\n"))

                chunk_id = f"{file_path}_{start_line}_{end_line}_{i}"

                metadata = {
                    "file_path": file_path,
                    "file_name": file_name,
                    "language": language,
                    "start_line": start_line,
                    "end_line": end_line,
                }

                ids.append(chunk_id)
                texts.append(node.text)
                metadatas.append(metadata)

            # Add nodes to ChromaDB collection
            collection.upsert(
                ids=ids,
                documents=texts,
                metadatas=metadatas
            )

            total_nodes += len(nodes)

        except Exception as e:
            logger.error(
                f"Error processing document "
                f"{doc.metadata.get('file_path', 'unknown')}: {e}"
            )

    logger.info(
        f"Successfully indexed {total_nodes} code chunks "
        f"across {len(documents)} files"
    )


async def index_projects():
    """Index all configured projects."""
    while True:
        try:
            for folder in config["folders_to_index"]:
                folder_path = os.path.join(config["projects_root"], folder)
                if not os.path.exists(folder_path):
                    logger.error(f"Folder not found: {folder_path}")
                    continue

                logger.info(f"Starting indexing of {folder}")

                # Load documents
                documents = load_documents(
                    folder_path,
                    set(config["ignore_dirs"]),
                    set(config["file_extensions"])
                )

                # Process and index documents
                process_and_index_documents(
                    documents,
                    folder,  # Use folder name as collection name
                    "chroma_db"
                )

                logger.info(f"Completed indexing of {folder}")

            # Wait for the configured interval
            await asyncio.sleep(60)  # Wait a minute before retrying

        except Exception as e:
            logger.error(f"Error in indexing loop: {e}")
            await asyncio.sleep(60)  # Wait a minute before retrying


@mcp.tool(
    name="search_code",
    description="""Search code using natural language queries.
        Args:
            query: Natural language query about the codebase
            project: Collection/folder name to search in. Use the current workspace name.
            n_results: Number of results to return (default: 5)
            threshold: Minimum relevance percentage to include results 
                (default: 30.0)
    """
)
async def search_code(
    query: str,
    project: str,
    n_results: int = 5,
    threshold: float = 30.0
):
    try:
        if not chroma_client:
            logger.error("ChromaDB client not initialized")
            return {
                "error": "ChromaDB client not initialized",
                "results": [],
                "total_results": 0
            }

        if not embedding_function:
            logger.error("Embedding function not initialized")
            return {
                "error": "Embedding function not initialized",
                "results": [],
                "total_results": 0
            }

        # Check if collection exists
        try:
            collection = chroma_client.get_collection(
                name=project,
                embedding_function=embedding_function
            )
        except Exception as e:
            logger.error(f"Collection {project} not found: {e}")
            return {
                "error": f"Collection {project} not found",
                "results": [],
                "total_results": 0
            }

        # Perform the search
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        # Filter results by threshold
        if (not results or not results["documents"] 
                or not results["documents"][0]):
            return {
                "error": "No results found",
                "results": [],
                "total_results": 0
            }

        filtered_docs = []
        filtered_metadatas = []
        filtered_distances = []

        for doc, meta, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            # Convert distance to similarity percentage
            similarity = (1 - distance) * 100

            # Only include results above threshold
            if similarity >= threshold:
                filtered_docs.append(doc)
                filtered_metadatas.append(meta)
                filtered_distances.append(distance)

        # Format results
        formatted_results = []
        for doc, meta, distance in zip(
            filtered_docs, filtered_metadatas, filtered_distances
        ):
            score = (1 - distance) * 100
            formatted_results.append({
                "text": doc,
                "file_path": meta.get("file_path", "Unknown file"),
                "language": meta.get("language", "text"),
                "start_line": int(meta.get("start_line", 0)),
                "end_line": int(meta.get("end_line", 0)),
                "relevance": round(score, 1)
            })

        return {
            "results": formatted_results,
            "total_results": len(formatted_results)
        }

    except Exception as e:
        logger.error(f"Error in search_code: {str(e)}")
        return {
            "error": str(e),
            "results": [],
            "total_results": 0
        }


# Run initialization before starting MCP
async def main():
    # Initialize ChromaDB before starting MCP
    success = await initialize_chromadb()

    if success:
        # Start indexing in background
        asyncio.create_task(index_projects())
        logger.info("Background indexing task started")

    await mcp.run_sse_async()

if __name__ == "__main__":
    asyncio.run(main())
