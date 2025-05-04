# GDS File Classifier

A Go-based CLI tool that uses AI vector embeddings to classify and organize files according to a user-defined category structure defined in a YAML file. This tool automatically matches files to the appropriate category based on their semantic content, using configurable embedding providers like OpenAI (including compatible local servers) or Ollama.

*(Note: While originally named for "Gary's Directory System", the tool is flexible and works with **any** category structure you define in the YAML mapping file.)*

## How It Works

The tool classifies files based on their content's semantic similarity to your defined categories. It works by:

1.  Loading your category system (hierarchy, names, descriptions) from the YAML file specified via `-mapping`.
2.  Generating rich vector embeddings for each valid category (`_valid: true`) using its path and `_description`. This uses a configured embedding provider (OpenAI/compatible or Ollama). Embeddings are cached in a `.embeddings.json` file for faster subsequent runs.
3.  Extracting text content from your source files. *(Current limitation: Text extraction is basic, primarily reading text-based files directly. Binary files like PDF/DOCX are handled simplistically. See "Extending the System".)*
4.  Creating embeddings for the extracted file content using the same provider.
5.  Calculating the cosine similarity between each file's embedding and every valid category's embedding.
6.  Matching each file to the category with the highest similarity score. Files with low similarity to any specific category are placed in a fallback "unsorted" category (typically defined as `00.00/unsorted` or similar in your YAML).
7.  Organizing the original files into output directories mirroring your category structure by copying (default) or moving (`-move` flag).
8.  Optionally logging detailed classification results to a CSV file (`-log-file` flag).

## Prerequisites

*   **Go:** Version 1.16 or later installed.
*   **API Access/Local Model:**
    *   For OpenAI: An OpenAI API key.
    *   For Ollama: A running Ollama instance with the desired embedding model pulled (e.g., `ollama pull nomic-embed-text`).
    *   For local OpenAI-compatible servers (LM Studio, Jan, etc.): A running server endpoint.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/dotcommander/gds-file-classifier
    cd gds-file-classifier
    ```

2.  **Install Go dependencies:**
    ```bash
    go mod download
    ```

3.  **Build the executable:**
    ```bash
    # This creates the executable in the current directory
    go build -o gds-classifier gds_classifier.go
    ```
    *(Alternatively, if using the Taskfile: `task build` which places it in `./bin/gds-classifier`)*

4.  **Configure API Key (if using OpenAI):**
    Set your OpenAI API key. You can do this via an environment variable:
    ```bash
    export OPENAI_API_KEY="your_openai_api_key_here"
    ```
    Or, create a `.env` file in the project root (recommended):
    ```dotenv
    # .env
    OPENAI_API_KEY=your_openai_api_key_here
    ```
    The tool will automatically load the key from the `.env` file if present. *(Note: If using a local server that doesn't require a key, you might set `OPENAI_API_KEY="NA"` or similar).*

## Minimal Setup

To build and run the `gds-classifier` with its default settings, you only need the following core files from the repository:

1.  `gds_classifier.go` - The main application source code.
2.  `go.mod` - Defines the project module and dependencies.
3.  `go.sum` - Ensures reproducible dependency builds.
4.  `gds.yaml` - The default category mapping file required by the classifier at runtime (or your own custom YAML file).

With these files, the Go compiler installed, and API access configured (if needed), you can build and run the tool as shown in the Installation and Usage sections.

## Usage

Basic usage (uses `./gds.yaml` by default):
```bash
./gds-classifier -source /path/to/directory/to/classify
```

Specify a custom mapping file and output directory:
```bash
./gds-classifier -source /path/to/classify -mapping ./my_categories.yaml -output ./organized_files
```

This will:
- Load your category system from the specified YAML file.
- Scan all supported files in the source directory.
- Classify files into the appropriate categories based on semantic similarity.
- Copy them to the output directory, organized according to your defined structure.

### Options

```
./gds-classifier [options] [source_directory]
```

**Core Options:**
*   `-source`: Source directory containing files to classify (required, can also be specified as the first argument).
*   `-output`: Output directory for sorted files (default: `"classified_files"`).
*   `-mapping`: Path to your category mapping YAML file (default: `"gds.yaml"`).
*   `-move`: Move files instead of copying them (default: `false`).
*   `-dry-run`: Show what would be done without actually moving/copying files (default: `false`).
*   `-log-file`: Optional: Path to CSV file for logging classification details (path, filename, filesize, category\_id, score).

**Filtering & Performance:**
*   `-include`: Additional file extensions to include (comma-separated, e.g., `.epub,.log`).
*   `-exclude`: Directories or file patterns to exclude (comma-separated, e.g., `node_modules,.git,*.tmp`).
*   `-concurrency`: Number of concurrent file processing operations (default: `4`).

**Embedding Provider Configuration:**
*   `-embedding-provider`: Embedding provider to use (`'openai'` or `'ollama'`, default: `'openai'`).
*   `-openai-base-url`: Optional: Custom base URL for OpenAI or OpenAI-compatible API endpoint (e.g., for local models via LM Studio: `http://localhost:1234/v1`).
*   `-openai-model`: Model name for OpenAI or compatible provider (default: `'text-embedding-3-small'`).
*   `-ollama-url`: URL for Ollama API (default: `'http://localhost:11434'`).
*   `-ollama-model`: Ollama embedding model name (e.g., `nomic-embed-text`, required if provider is `'ollama'`).

### Examples

Classify files using your custom system:
```bash
./gds-classifier -source ~/Documents/ToSort -mapping ~/my_org_structure.yaml
```

Move files instead of copying:
```bash
./gds-classifier -source ~/Downloads -mapping ./gds.yaml -move
```

Test classification without moving/copying:```bash
./gds-classifier -source ~/Documents -mapping ./gds.yaml -dry-run
```

Classify only specific additional file types:
```bash
./gds-classifier -source ~/Documents -mapping ./gds.yaml -include "csv,xlsx,json"
```

Exclude specific patterns:
```bash
./gds-classifier -source ~/Documents -mapping ./gds.yaml -exclude "node_modules,venv,.git,*.tmp"
```

Classify using a local OpenAI-compatible endpoint (e.g., LM Studio, Jan):
```bash
# Set API key (even if dummy)
export OPENAI_API_KEY="NA"

./gds-classifier -source ~/Documents -mapping ./gds.yaml \
  -embedding-provider openai \
  -openai-base-url http://localhost:1234/v1 \
  -openai-model <model_name_loaded_on_your_server>
```

Classify using Ollama:
```bash
./gds-classifier -source ~/Documents -mapping ./gds.yaml \
  -embedding-provider ollama \
  -ollama-model nomic-embed-text # Or your chosen Ollama embedding model
```

Log classification details to CSV:
```bash
./gds-classifier -source ~/Documents -mapping ./gds.yaml -log-file classification_log.csv
```

### Category YAML Structure

Your YAML file defines your desired classification structure.

```yaml
# Example: my_categories.yaml
"Projects": { # Top-level category
  _valid: false, # Not a target for files itself
  _description: "Active and planned project work.", # Description used for context if needed
  "Project X": {
    _valid: true, # Files *can* be classified directly into this category
    _description: "Research, notes, and documents related to the secret Project X initiative focused on AI ethics." # Specific description for embedding
  }
}
"Reference": {
  _valid: true, # Can classify general reference files here
  _description: "General reference material, articles, code snippets, and documentation."
}
"Archive": { # Another valid target category
  _valid: true,
  _description: "Completed projects, old logs, and historical documents no longer in active use."
}
"Unsorted": { # Recommended fallback category
  _valid: true,
  _description: "Files awaiting classification or those that don't clearly match other categories."
}
```

Key elements:
*   **Hierarchy:** Defined by YAML nesting.
*   **Category Names:** Keys in the YAML (e.g., `"Projects"`, `"Project X"`).
*   **`_valid: true/false`:** Marks if a category is a valid target for file classification. Only `_valid: true` categories get embeddings generated.
*   **`_description: "..."`:** **Crucial.** Provides the semantic context for the category. This text is used to generate the category's embedding vector, enabling accurate matching with file content. Write clear, descriptive text here.

## Understanding the Results

After running the classifier, you'll typically find:

1.  **Organized Files:** Files copied or moved into the `-output` directory, structured according to your YAML definition and the classification results.
2.  **`category_stats.json`:** (In the output directory, if not `-dry-run`) A JSON file summarizing how many files were placed into each category.
3.  **`.embeddings.json`:** (Next to your `-mapping` YAML file) A cache of the generated category embeddings. Deleting this file will force regeneration on the next run. It's safe to delete if you modify your category descriptions.
4.  **CSV Log File:** (If `-log-file` was specified) A CSV file containing detailed logs for each processed file:
    *   `path`: Full original path of the file.
    *   `filename`: Base name of the file.
    *   `filesize`: Size of the file in bytes.
    *   `category_id`: The category path the file was matched to.
    *   `score`: The cosine similarity score (confidence) of the match (higher is better).
5.  **Terminal Output:** Log messages showing progress, matches found (including scores), and any errors encountered during processing.

Files that don't match well with any specific `_valid: true` category are typically placed in your designated "unsorted" fallback category (e.g., `"Unsorted"` or `"00.00/unsorted"` if using the GDS example).

## Troubleshooting

*   **API Key Issues:** Ensure the `OPENAI_API_KEY` environment variable is set correctly or present in `.env` if using OpenAI. Check if your local server requires an API key.
*   **Provider Configuration:** Double-check `--embedding-provider`, `--openai-base-url`, `--openai-model`, `--ollama-url`, and `--ollama-model` flags match your setup (correct URLs, model names available on the server).
*   **Connection Errors:** Ensure Ollama or your local OpenAI-compatible server is running and accessible from where you run the classifier. Check firewalls.
*   **YAML Parsing Errors:** Verify your category mapping YAML file has correct syntax. Use a YAML validator if needed.
*   **Ollama Issues:** Ensure the specified `--ollama-model` is downloaded (`ollama pull <model>`) and is an embedding model. Check Ollama server logs.
*   **No Matches / Poor Matches:**
    *   Improve the `_description` fields in your YAML for valid categories to be more specific and semantically rich.
    *   Ensure the embedding model you are using is suitable for your content type.
    *   Consider if the file content itself is too short or ambiguous for semantic matching.
*   **Path Errors:** Ensure your YAML uses consistent path formatting. The tool attempts to handle separators correctly, but consistency helps. Check permissions for source and output directories.
*   **Cache Issues:** If classification seems wrong after changing descriptions, delete the `.embeddings.json` file next to your YAML file to force regeneration.

## Extending the System

You can adapt and extend the classifier:

1.  **Modify Categories:** Update your category mapping YAML file with new categories, descriptions, or hierarchy changes. Remember to add `_valid: true` and a good `_description` for new target categories.
2.  **Improve Text Extraction:** The current `extractTextFromFile` function in `gds_classifier.go` is basic. You could enhance it to use external tools (like `pdftotext`, `pandoc`) or Go libraries to properly extract text from PDFs, DOCX, ODT, etc., for better classification of those file types.
3.  **Adjust Matching:** Modify the `cosineSimilarity` function or the logic in `matchFilesToCategories` if you need a different similarity metric or matching threshold.
4.  **Add Features:** Contribute new features like different output formats, alternative classification algorithms, or UI interfaces.

## Contributing

Contributions are welcome! Please feel free to submit Issues for bugs or feature requests, or Pull Requests with improvements. (Consider adding more specific contribution guidelines here if needed, e.g., code style, testing requirements).

## License

MIT
