# FastApply MCP Server

A Model Context Protocol server that provides AI-powered code editing capabilities through FastApply integration.

## Overview

FastApply MCP Server enables intelligent code editing by connecting MCP-compatible clients to FastApply language models. The server provides two core tools for applying code changes with AI assistance, featuring automatic backup management and comprehensive validation.

## Features

- AI-guided code editing through FastApply models
- Dry-run preview mode for safe change validation
- Automatic backup system with environment-based control
- Atomic file operations with optimistic concurrency
- Comprehensive input validation and security checks
- Support for multiple FastApply-compatible backends

## Installation

### Requirements

- Python 3.13 or higher
- FastApply-compatible server (LM Studio, Ollama, or custom OpenAI-compatible endpoint)

### Setup

#### Using uvx (Recommended)

Run directly without installation:

```bash
uvx fastapply-mcp
```

#### Manual Installation

Clone the repository and install dependencies:

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

Create a `.env` file with your configuration:

```bash
cp .env.example .env
```

## Configuration

Configure the server through environment variables in your `.env` file:

```bash
# FastApply Server Configuration
FAST_APPLY_URL=http://localhost:1234/v1
FAST_APPLY_MODEL=fastapply-1.5b
FAST_APPLY_TIMEOUT=300.0
FAST_APPLY_MAX_TOKENS=8000
FAST_APPLY_TEMPERATURE=0.05

# Security Settings
MAX_FILE_SIZE=10485760

# Backup Control (default: disabled)
FAST_APPLY_AUTO_BACKUP=False
```

### Backup System

The automatic backup feature is disabled by default. To enable automatic backups before file modifications:

```bash
FAST_APPLY_AUTO_BACKUP=True
```

When enabled, the server creates timestamped backups in the format `{filename}.bak_{timestamp}` before applying changes.

## MCP Integration

### Claude Desktop

Add the server to your Claude Desktop configuration file:

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
      "args": ["/path/to/fastapply-mcp/src/fastapply/main.py"],
      "env": {
        "FAST_APPLY_URL": "http://localhost:1234/v1",
        "FAST_APPLY_MODEL": "fastapply-1.5b"
      }
    }
  }
}
```

The server operates on the current working directory where the MCP client is running, similar to other MCP tools.

### Other MCP Clients

The server implements the standard MCP protocol and works with any compatible client. Refer to your client's documentation for integration instructions.

## Available Tools

### edit_file

Applies AI-guided code edits to a target file with comprehensive validation and safety checks.

**Parameters:**
- `target_file` (required): Path to the file to edit
- `instructions` (required): Natural language description of desired changes
- `code_edit` (required): Code snippet or edit instructions
- `force` (optional): Override safety checks and optimistic concurrency
- `output_format` (optional): Response format, either "text" or "json"

**Features:**
- Atomic file operations with rollback capability
- SHA-256 content verification for optimistic concurrency
- Automatic syntax validation for supported languages
- Optional automatic backup creation
- Unified diff generation for change visualization

**Example:**

```json
{
  "target_file": "src/utils.py",
  "instructions": "Add error handling to the parse_config function",
  "code_edit": "def parse_config(path):\n    try:\n        with open(path) as f:\n            return json.load(f)\n    except FileNotFoundError:\n        raise ConfigError(f'Config file not found: {path}')\n    except json.JSONDecodeError as e:\n        raise ConfigError(f'Invalid JSON in config: {e}')"
}
```

### dry_run_edit_file

Previews code edits without modifying the target file, allowing safe validation of changes.

**Parameters:**
- `target_file` (required): Path to the file to preview
- `instruction` (optional): Natural language description of desired changes
- `code_edit` (required): Code snippet or edit instructions
- `output_format` (optional): Response format, either "text" or "json"

**Features:**
- Complete edit preview with unified diff
- Validation results without file modification
- First 20 lines of merged code preview
- Safety information and warnings

**Example:**

```json
{
  "target_file": "src/utils.py",
  "code_edit": "def parse_config(path):\n    try:\n        with open(path) as f:\n            return json.load(f)\n    except Exception as e:\n        raise ConfigError(f'Failed to parse config: {e}')"
}
```

## FastApply Backend Options

The server supports multiple FastApply-compatible backends:

### LM Studio

Download and run FastApply models through LM Studio's GUI:

1. Install LM Studio from https://lmstudio.ai
2. Download a FastApply-compatible model
3. Start the local server (default: http://localhost:1234)
4. Configure FAST_APPLY_URL in your environment

### Ollama

Run FastApply models through Ollama's CLI:

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a FastApply model
ollama pull fastapply-1.5b

# Start the server
ollama serve
```

Configure FAST_APPLY_URL to point to your Ollama instance.

### Custom OpenAI-Compatible Servers

Any server implementing the OpenAI API specification can be used as a backend. Configure the appropriate URL and model identifier in your environment.

## Security

The server implements multiple security layers:

- **Workspace Isolation**: All file operations are confined to the current working directory
- **Path Validation**: Strict path resolution prevents directory traversal attacks
- **File Size Limits**: Configurable maximum file size prevents resource exhaustion
- **Input Sanitization**: Comprehensive validation of all user inputs
- **Atomic Operations**: File changes are atomic with automatic rollback on failure

## Development

### Project Structure

```
fastapply-mcp/
├── src/
│   └── fastapply-mcp/
│       ├── __init__.py
│       └── main.py          # Core server implementation
├── .env.example
├── pyproject.toml
└── README.md
```

### Code Quality

The project uses standard Python tooling for code quality:

```bash
# Format code
ruff format .

# Lint code
ruff check .

# Type checking
mypy src/
```

## Troubleshooting

### Connection Issues

Verify your FastApply server is accessible:

```bash
curl http://localhost:1234/v1/models
```

Check the server logs for connection errors and verify your FAST_APPLY_URL configuration.

### Permission Errors

Ensure the server process has appropriate file system permissions for the current working directory:

```bash
pwd
ls -la
```

### Performance Issues

For large files or complex edits, consider:

- Increasing FAST_APPLY_TIMEOUT
- Adjusting FAST_APPLY_MAX_TOKENS
- Reducing FAST_APPLY_TEMPERATURE for more deterministic output

## Contributing

Contributions are welcome. Please follow these guidelines:

1. Fork the repository and create a feature branch
2. Write tests for new functionality
3. Ensure all tests pass and code meets quality standards
4. Submit a pull request with a clear description of changes

## License

MIT License - see LICENSE file for details.

## Support

- GitHub Issues: Report bugs and request features
- Discussions: Ask questions and share ideas
- Documentation: Refer to inline code documentation for implementation details

## Acknowledgments

This project integrates with FastApply models and implements the Model Context Protocol specification. Thanks to the MCP community and FastApply model developers for their foundational work.
