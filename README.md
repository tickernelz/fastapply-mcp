# FastApply MCP Server

A streamlined Model Context Protocol server for efficient AI-powered code editing using FastApply. Inspired by opencode-fast-apply's simplicity and partial editing approach.

## Overview

FastApply MCP Server provides intelligent code editing through partial file editing, achieving **80-98% token savings** compared to full-file approaches. The server uses smart matching to locate and replace code sections automatically, making it ideal for editing large files efficiently.

## Key Features

- **Partial File Editing**: Edit only the sections you need (50-500 lines recommended)
- **Smart Matching**: Automatic exact and normalized whitespace matching
- **XML Safety**: Built-in protection against prompt injection
- **Token Efficiency**: 80-98% token savings vs full-file editing
- **Binary Detection**: Automatic detection and rejection of binary files
- **Atomic Operations**: Safe file writes with automatic rollback on failure
- **Clear Error Messages**: Actionable suggestions for troubleshooting

## Installation

### Requirements

- Python 3.13 or higher
- FastApply-compatible server (LM Studio, Ollama, or OpenAI-compatible endpoint)

### Using uvx (Recommended)

Run directly without installation:

```bash
uvx fastapply-mcp
```

### Manual Installation

```bash
git clone https://github.com/your-org/fastapply-mcp.git
cd fastapply-mcp

# Using uv
uv sync
source .venv/bin/activate
uv pip install -e .

# Or using pip
pip install -e .
```

## Configuration

Configure the server with just 3 environment variables:

```bash
# .env file
FAST_APPLY_URL=http://localhost:1234/v1
FAST_APPLY_MODEL=fastapply-1.5b
FAST_APPLY_API_KEY=optional-api-key
```

That's it! No complex configuration needed.

## MCP Integration

### Claude Desktop

Add to your Claude Desktop configuration:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

#### Using uvx (Recommended)

```json
{
  "mcpServers": {
    "fastapply": {
      "command": "uvx",
      "args": ["fastapply-mcp"],
      "env": {
        "FAST_APPLY_URL": "http://localhost:1234/v1",
        "FAST_APPLY_MODEL": "fastapply-1.5b"
      }
    }
  }
}
```

#### Manual Installation

```json
{
  "mcpServers": {
    "fastapply": {
      "command": "python",
      "args": ["/path/to/fastapply-mcp/src/fastapply_mcp/main.py"],
      "env": {
        "FAST_APPLY_URL": "http://localhost:1234/v1",
        "FAST_APPLY_MODEL": "fastapply-1.5b"
      }
    }
  }
}
```

### Other MCP Clients

The server implements the standard MCP protocol and works with any compatible client.

## Tool: fast_apply_edit

The server provides a single, focused tool for efficient code editing.

### Parameters

- **target_filepath** (required): Path to the file to edit (relative or absolute)
- **original_code** (required): The exact section of code to modify (50-500 lines recommended)
- **code_edit** (required): The changes to apply

### How It Works

1. **Read** the file to get current content
2. **Extract** the relevant section (50-500 lines with context)
3. **Call** FastApply API with partial content
4. **Smart match** finds the section in the full file
5. **Replace** the section atomically
6. **Generate** diff for verification

### Example Usage

```json
{
  "target_filepath": "src/utils.py",
  "original_code": "def parse_config(path):\n    with open(path) as f:\n        return json.load(f)",
  "code_edit": "def parse_config(path):\n    try:\n        with open(path) as f:\n            return json.load(f)\n    except FileNotFoundError:\n        raise ConfigError(f'Config not found: {path}')"
}
```

### Lazy Edit Markers

Use `... existing code ...` markers for unchanged sections:

```python
# ... existing code ...
def updated_function():
    return "modified"
# ... existing code ...
```

This tells the AI to skip regenerating unchanged parts, making edits faster.

## Token Efficiency

Partial editing provides massive token savings:

| File Size | Full File | Partial (100 lines) | Savings |
|-----------|-----------|---------------------|---------|
| 100 lines | 2,500 tokens | 500 tokens | **80%** |
| 500 lines | 12,500 tokens | 1,000 tokens | **92%** |
| 1000 lines | 25,000 tokens | 1,500 tokens | **94%** |
| 5000 lines | 125,000 tokens | 2,000 tokens | **98%** |

## Smart Matching

The tool uses a two-tier matching system:

### 1. Exact Match (Priority)
Finds exact string match in the file.

### 2. Normalized Match (Fallback)
Handles CRLF/LF differences automatically:
- Normalizes `\r\n` → `\n`
- Normalizes `\r` → `\n`
- Matches whitespace-normalized content

### 3. Uniqueness Check
Ensures the section appears only once in the file to prevent ambiguous replacements.

## XML Safety

Built-in protection against prompt injection:

```python
# User code with XML tags
original_code = "<code>malicious</code>"

# Automatically escaped before API call
# "&lt;code&gt;malicious&lt;/code&gt;"

# Safely processed and unescaped after
```

All XML special characters (`&`, `<`, `>`, `"`, `'`) are automatically escaped and unescaped.

## FastApply Backend Options

### LM Studio

1. Install LM Studio from https://lmstudio.ai
2. Download a FastApply-compatible model
3. Start the local server (default: http://localhost:1234)
4. Configure `FAST_APPLY_URL=http://localhost:1234/v1`

### Ollama

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a FastApply model
ollama pull fastapply-1.5b

# Start the server
ollama serve
```

Configure `FAST_APPLY_URL=http://localhost:11434/v1`

### OpenAI or Custom Servers

Any OpenAI-compatible API works:

```bash
FAST_APPLY_URL=https://api.openai.com/v1
FAST_APPLY_MODEL=gpt-4
FAST_APPLY_API_KEY=sk-...
```

## Security

- **Workspace Isolation**: All operations confined to current working directory
- **Path Validation**: Prevents directory traversal attacks
- **File Size Limits**: 10MB default maximum
- **Binary Detection**: Rejects binary files automatically
- **UTF-8 Validation**: Ensures proper file encoding
- **Atomic Writes**: Safe file operations with rollback

## Error Handling

Clear, actionable error messages:

```
❌ Error: Cannot locate original_code in file (whitespace mismatch detected).

💡 Troubleshooting:
  1. Re-read the file to get current content
  2. Ensure original_code matches exactly (including whitespace)
  3. Provide more context to make the section unique
```

## Troubleshooting

### Connection Issues

Verify your FastApply server is running:

```bash
curl http://localhost:1234/v1/models
```

### File Not Found

Use the tool only for **existing files**. For new files, use your MCP client's write tool.

### Cannot Locate Section

- Re-read the file to get current content
- Ensure whitespace matches exactly (tabs vs spaces)
- Provide more context to make the section unique

### Whitespace Mismatch

The tool handles CRLF/LF differences automatically, but tabs vs spaces must match exactly.

## Development

### Project Structure

```
fastapply-mcp/
├── src/
│   └── fastapply_mcp/
│       ├── __init__.py
│       └── main.py          # Single-file implementation (~487 lines)
├── .env.example
├── pyproject.toml
└── README.md
```

### Code Quality

```bash
# Format code
ruff format .

# Lint code
ruff check .

# Type checking
mypy src/

# Syntax check
python -m py_compile src/fastapply_mcp/main.py
```

## Design Philosophy

This implementation follows these principles:

1. **Do one thing well** - Focus on efficient file editing
2. **Trust the client** - MCP client handles undo, concurrency, etc.
3. **Optimize for common case** - Partial editing is 10x more efficient
4. **Clear errors** - Help users fix problems quickly
5. **No premature optimization** - Remove unused features

## Performance

### Original Approach (Full File)
- Read: 5000 lines
- Send to API: 125,000 tokens
- Process: ~30 seconds
- Cost: High

### Simplified Approach (Partial)
- Read: 5000 lines (send only 100)
- Send to API: 2,000 tokens
- Process: ~3 seconds
- Cost: **98% cheaper**

## Contributing

Contributions are welcome! Please:

1. Fork the repository and create a feature branch
2. Write tests for new functionality
3. Ensure code meets quality standards
4. Submit a pull request with clear description

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- **Inspiration**: opencode-fast-apply for the partial editing approach
- **MCP Community**: For the Model Context Protocol specification
- **FastApply**: For the efficient code merging models

## Support

- **GitHub Issues**: Report bugs and request features
- **Discussions**: Ask questions and share ideas
- **Documentation**: See inline code comments for implementation details
