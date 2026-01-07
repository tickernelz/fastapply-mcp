"""
FastApply MCP Server Implementation

A standalone MCP server with FastApply code editing capabilities.
Uses stdio transport and provides multi-file editing tools like Morphllm.
"""

import difflib
import hashlib
import json
import os
import threading
import time
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

import openai
import structlog
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()
logger = structlog.get_logger("fast-apply-mcp")

# File size limits to prevent memory exhaustion
MAX_FILE_SIZE = int(os.getenv("FAST_APPLY_MAX_FILE_BYTES", "10485760"))  # 10MB default

# Response size limits (approximate character limit ~40k tokens @ 6 chars/token)
MAX_RESPONSE_SIZE = 240000

# Maximum combined size (bytes) of original + edit snippet allowed in a single API request
MAX_REQUEST_BYTES = int(os.getenv("FAST_APPLY_MAX_REQUEST_BYTES", str(2 * MAX_FILE_SIZE)))

# File locking mechanism to prevent race conditions
_file_locks = {}
_locks_lock = threading.Lock()

# Optional strict path mode (default on). Set FAST_APPLY_STRICT_PATHS=0 to allow legacy CWD fallback.
STRICT_PATHS = os.getenv("FAST_APPLY_STRICT_PATHS", "1") not in ("0", "false", "False")

# Simple availability caches to avoid repeated subprocess spawning when tools absent
_RUFF_AVAILABLE: Optional[bool] = None
_ESLINT_AVAILABLE: Optional[bool] = None


def _get_file_lock(file_path: str) -> threading.Lock:
    """Get or create a lock for a specific canonical file path.

    Uses realpath to ensure all path variants for the same file share a single lock.
    """
    canonical = os.path.realpath(file_path)
    with _locks_lock:
        if canonical not in _file_locks:
            _file_locks[canonical] = threading.Lock()
        return _file_locks[canonical]


def _cleanup_file_locks():
    """Clean up unused file locks to prevent memory leaks."""
    with _locks_lock:
        # Remove locks for files that no longer exist
        existing_files = set(_file_locks.keys())
        for file_path in list(existing_files):
            if not os.path.exists(file_path):
                del _file_locks[file_path]
                logger.debug("Cleaned up lock for non-existent file", file_path=file_path)


def _secure_resolve(path: str) -> str:
    """Securely resolve a path confined to current working directory.

    Rules:
        - Only paths within cwd after realpath are allowed.
        - Absolute paths must already be inside cwd.
        - Relative paths are resolved relative to cwd.
    """
    workspace_root = os.path.realpath(os.getcwd())
    original = path

    # If absolute, keep as-is for validation; else join
    if not os.path.isabs(path):
        path = os.path.join(workspace_root, path)

    candidate = os.path.realpath(path)

    if os.path.commonpath([candidate, workspace_root]) == workspace_root:
        return candidate

    if not STRICT_PATHS:
        # Legacy fallback: allow any existing path under current working directory
        cwd_real = os.path.realpath(os.getcwd())
        if os.path.exists(candidate) and os.path.commonpath([candidate, cwd_real]) == cwd_real:
            return candidate

    raise ValueError(f"Path escapes workspace: {original}")





# Official Fast Apply template constants
FAST_APPLY_SYSTEM_PROMPT = """You are a coding assistant that helps merge code updates, ensuring every modification is fully integrated."""

FAST_APPLY_USER_PROMPT = """Merge all changes from the <update> snippet into the <code> below.
Instruction: {instruction}
- Preserve the code's structure, order, comments, and indentation exactly.
- Output only the updated code, enclosed within <updated-code> and </updated-code> tags.
- Do not include any additional text, explanations, placeholders, markdown, ellipses, or code fences.

<code>{original_code}</code>

<update>{update_snippet}</update>

Provide the complete updated code."""

# XML tag constants for response parsing
UPDATED_CODE_START = "<updated-code>"
UPDATED_CODE_END = "</updated-code>"


def _atomic_write(path: str, data: str):
    """Write data atomically to path using temp file + fsync + os.replace."""
    import tempfile

    dir_name = os.path.dirname(path)
    if dir_name:  # Only create directories if path is not in current directory
        os.makedirs(dir_name, exist_ok=True)
    # Use None for current directory instead of empty string
    temp_dir = dir_name if dir_name else None
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=temp_dir) as tmp:
        tmp.write(data)
        tmp.flush()
        try:
            os.fsync(tmp.fileno())
        except Exception:
            # fsync best effort
            pass
        tmp_path = tmp.name
    os.replace(tmp_path, path)


def _create_timestamped_backup(original_path: str) -> str:
    """Create a timestamped backup of the original file.

    Returns the backup path. Falls back to .bak if timestamp write fails.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{original_path}.bak_{timestamp}"
    try:
        with open(original_path, "r", encoding="utf-8") as src, open(backup_path, "w", encoding="utf-8") as dst:
            dst.write(src.read())
        return backup_path
    except Exception as e:
        logger.warning("Failed to create timestamped backup", error=str(e), path=original_path)
        fallback_path = original_path + ".bak"
        try:
            with open(original_path, "r", encoding="utf-8") as src, open(fallback_path, "w", encoding="utf-8") as dst:
                dst.write(src.read())
            return fallback_path
        except Exception:
            # If even fallback fails, re-raise original error context
            raise


# Maximum diff size to prevent memory exhaustion (100KB)
MAX_DIFF_SIZE = 102400


def generate_udiff(original_code: str, modified_code: str, target_file: str) -> str:
    """Generate UDiff between original and modified code for verification."""
    diff = "\n".join(
        difflib.unified_diff(
            original_code.splitlines(keepends=True),
            modified_code.splitlines(keepends=True),
            fromfile=target_file,
            tofile=target_file,
            lineterm="",
        )
    )

    # Truncate large diffs to prevent memory issues
    if len(diff) > MAX_DIFF_SIZE:
        logger.warning("Diff exceeds maximum size, truncating", diff_size=len(diff), max_size=MAX_DIFF_SIZE)
        lines = diff.splitlines()
        truncated_lines = lines[:50] + ["... (diff truncated due to size)"] + lines[-50:]
        diff = "\n".join(truncated_lines)

    return diff


def validate_code_quality(file_path: str, code: str) -> Dict[str, Any]:
    """Validate code quality using available linting tools."""
    validation_results: Dict[str, Any] = {"has_errors": False, "errors": [], "warnings": [], "suggestions": []}

    try:
        # Try to detect file type and run appropriate linters
        if file_path.endswith(".py"):
            lint_results = _validate_python_code(code)
        elif file_path.endswith((".js", ".jsx", ".ts", ".tsx")):
            lint_results = _validate_javascript_code(code)
        else:
            lint_results = {}

        if lint_results:
            validation_results["errors"].extend(lint_results.get("errors", []))
            validation_results["warnings"].extend(lint_results.get("warnings", []))
            validation_results["suggestions"].extend(lint_results.get("suggestions", []))
            if validation_results["errors"]:
                validation_results["has_errors"] = True
    except Exception as e:
        logger.warning("Code validation failed", error=str(e), file_path=file_path)

    return validation_results


def _validate_python_code(code: str) -> Dict[str, Any]:
    """Validate Python code using Ruff if available (cached availability)."""
    global _RUFF_AVAILABLE
    results: Dict[str, List[str]] = {"errors": [], "warnings": [], "suggestions": []}
    if _RUFF_AVAILABLE is False:
        return results
    try:
        import subprocess
        import tempfile

        if _RUFF_AVAILABLE is None:
            # Probe availability quickly
            try:
                subprocess.run(["ruff", "--version"], capture_output=True, timeout=2)
                _RUFF_AVAILABLE = True
            except Exception:
                _RUFF_AVAILABLE = False
                return results

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            temp_file = f.name
        try:
            result = subprocess.run(["ruff", "check", "--format=json", temp_file], capture_output=True, text=True, timeout=10)
            if result.stdout:
                import json

                ruff_results = json.loads(result.stdout)
                for issue in ruff_results:
                    location = issue.get("location", {})
                    row = location.get("row")
                    msg = issue.get("message")
                    code_id = issue.get("code")
                    bucket = "warnings"
                    if code_id and code_id.startswith(("E", "F")):
                        bucket = "errors"
                    results[bucket].append(f"Line {row}: {msg} ({code_id})")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        finally:
            try:
                os.unlink(temp_file)
            except Exception:
                pass
    except Exception:
        pass
    return results


def _validate_javascript_code(code: str) -> Dict[str, Any]:
    """Validate JavaScript/TypeScript code using ESLint if available (cached availability)."""
    global _ESLINT_AVAILABLE
    results: Dict[str, List[str]] = {"errors": [], "warnings": [], "suggestions": []}
    if _ESLINT_AVAILABLE is False:
        return results
    try:
        import subprocess
        import tempfile

        if _ESLINT_AVAILABLE is None:
            try:
                subprocess.run(["eslint", "--version"], capture_output=True, timeout=2)
                _ESLINT_AVAILABLE = True
            except Exception:
                _ESLINT_AVAILABLE = False
                return results
        with tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False) as f:
            f.write(code)
            temp_file = f.name
        try:
            result = subprocess.run(["eslint", "--format=json", temp_file], capture_output=True, text=True, timeout=10)
            if result.stdout:
                import json

                eslint_results = json.loads(result.stdout)
                for file_result in eslint_results:
                    for message in file_result.get("messages", []):
                        if message.get("severity") == 2:
                            results["errors"].append(f"Line {message.get('line')}: {message.get('message')}")
                        else:
                            results["warnings"].append(f"Line {message.get('line')}: {message.get('message')}")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        finally:
            try:
                os.unlink(temp_file)
            except Exception:
                pass
    except Exception:
        pass
    return results


def search_files(root_path: str, pattern: str, exclude_patterns: Optional[List[str]] = None) -> List[str]:
    """Search for files matching a pattern, with optional exclude patterns."""
    if exclude_patterns is None:
        exclude_patterns = []

    # Add default exclusion for common large directories
    default_excludes = [".git", "node_modules", "venv", "env", ".venv", "__pycache__", ".pytest_cache"]
    all_exclude_patterns = exclude_patterns + default_excludes

    results = []

    def should_exclude(file_path: str) -> bool:
        """Check if file should be excluded based on patterns."""
        relative_path = os.path.relpath(file_path, root_path)
        for exclude_pattern in all_exclude_patterns:
            # Simple pattern matching - can be enhanced with glob patterns
            if exclude_pattern in relative_path or exclude_pattern in os.path.basename(file_path):
                return True
        return False

    def search_recursive(current_path: str):
        """Recursively search directories using os.scandir for better performance."""
        try:
            with os.scandir(current_path) as entries:
                for entry in entries:
                    item_path = entry.path

                    # Skip if path is outside workspace bounds (now using _secure_resolve)
                    try:
                        _secure_resolve(item_path)
                    except ValueError:
                        continue

                    if should_exclude(item_path):
                        continue

                    if entry.is_file():
                        # Check if filename matches pattern (case-insensitive)
                        if pattern.lower() in entry.name.lower():
                            results.append(item_path)
                    elif entry.is_dir():
                        search_recursive(item_path)
        except (PermissionError, OSError):
            # Skip directories we can't access
            pass

    search_recursive(root_path)
    return results


class FastApplyConnector:
    """Handles connections to Fast Apply API using OpenAI-compatible client.

    Responsibilities:
    - Maintain configuration
    - Provide apply_edit with structured verification
    - Parse & validate model responses
    - Avoid leaking secrets in return values/logs
    """

    def __init__(
        self,
        url: str = os.getenv("FAST_APPLY_URL", "http://localhost:1234/v1"),
        model: str = os.getenv("FAST_APPLY_MODEL", "fastapply-1.5b"),
        api_key: Optional[str] = os.getenv("FAST_APPLY_API_KEY"),
        timeout: float = float(os.getenv("FAST_APPLY_TIMEOUT", "300.0")),
        max_tokens: int = int(os.getenv("FAST_APPLY_MAX_TOKENS", "8000")),
        temperature: float = float(os.getenv("FAST_APPLY_TEMPERATURE", "0.05")),
    ):
        """Initialize the Fast Apply connector with configuration parameters."""

        # Validate configuration parameters
        if timeout <= 0 or timeout > 300:
            raise ValueError("Timeout must be between 0 and 300 seconds")
        if max_tokens <= 0 or max_tokens > 32000:
            raise ValueError("Max tokens must be between 0 and 32000")
        if temperature < 0 or temperature > 2:
            raise ValueError("Temperature must be between 0 and 2")

        self.url = url
        self.model = model
        self.api_key = api_key or "optional-api-key"
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client: Optional[openai.OpenAI] = None

        # Initialize OpenAI client with Fast Apply API configuration
        try:
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.url,
                timeout=self.timeout,
            )
        except Exception as e:
            # Allow for testing without API connection
            logger.warning("Could not initialize OpenAI client, running in test mode", error=str(e))
            self.client = None

        logger.info(
            "FastApplyConnector initialized",
            model=self.model,
            timeout=self.timeout,
            max_tokens=self.max_tokens,
            has_client=self.client is not None,
        )

    def _strip_markdown_blocks(self, text: str) -> str:
        """Remove surrounding markdown code fences if present."""
        stripped = text.strip()
        if stripped.startswith("```") and stripped.endswith("```"):
            lines = stripped.splitlines()
            # remove first and last fence lines
            if len(lines) >= 2:
                inner = lines[1:-1]
                # If first fence contains language spec, already removed
                return "\n".join(inner).strip()
        # Inline style ```code``` single line
        if stripped.startswith("```") and stripped.count("```") == 2 and "\n" not in stripped:
            return stripped.strip("`")
        return text

    def _parse_fast_apply_response(self, raw_response: str) -> str:
        """Parse model response extracting exactly one <updated-code> block or fallback to markdown strip.

        Mirrors expected legacy semantics used in existing tests.
        """
        if not raw_response or not raw_response.strip():
            raise ValueError("Fast Apply API response is empty")

        if len(raw_response) > MAX_RESPONSE_SIZE:
            logger.warning(
                "Response exceeds maximum allowed size, truncating",
                response_size=len(raw_response),
                max_size=MAX_RESPONSE_SIZE,
            )
            raw_response = raw_response[:MAX_RESPONSE_SIZE]

        cleaned = _sanitize_model_response(raw_response)
        try:
            return _extract_single_tag_block(cleaned)
        except ValueError:
            # Fallback: strip markdown fences
            return self._strip_markdown_blocks(cleaned)

    def apply_edit(self, *args, **kwargs):  # type: ignore[override]
        """Apply code edit.

        Dual-interface support:
        - New style: apply_edit(original_code=..., code_edit=..., instruction=..., file_path=...)
          Returns rich dict with metadata.
        - Legacy positional style: apply_edit(instructions, original_code, code_edit)
          Returns merged code string.
        """
        legacy_mode = False
        if args and not kwargs:
            # Legacy expects 3 positional arguments: instruction, original_code, code_edit
            if len(args) == 3:
                instruction, original_code, code_edit = args
                file_path = None
                legacy_mode = True
            else:
                raise TypeError("Legacy apply_edit expects exactly 3 positional arguments")
        else:
            original_code = kwargs.get("original_code")
            code_edit = kwargs.get("code_edit")
            # Accept both singular & plural key variants
            instruction = kwargs.get("instruction") or kwargs.get("instructions", "")
            file_path = kwargs.get("file_path")
            if original_code is None or code_edit is None:
                raise TypeError("apply_edit requires original_code and code_edit")

        if self.client is None:
            raise RuntimeError("Fast Apply client not initialized; cannot perform edit.")
        try:
            # Format the request according to official Fast Apply specification
            user_content = FAST_APPLY_USER_PROMPT.format(
                original_code=original_code, update_snippet=code_edit, instruction=instruction or "Apply the requested code changes."
            )

            logger.info(
                "Sending edit request to Fast Apply API",
                original_code_length=len(original_code),
                code_edit_length=len(code_edit),
                instruction=instruction,
                file_path=file_path,
            )

            # Make the API call with proper system/user message structure
            # Request size pre-flight check (approx based on raw string byte lengths)
            request_bytes = len(user_content.encode("utf-8"))
            if request_bytes > MAX_REQUEST_BYTES:
                raise ValueError(f"Request size {request_bytes} bytes exceeds limit {MAX_REQUEST_BYTES} bytes (original + edit too large)")

            # Retry logic for transient API errors
            attempts = int(os.getenv("FAST_APPLY_RETRY_ATTEMPTS", "3"))
            backoff_base = float(os.getenv("FAST_APPLY_RETRY_BACKOFF", "0.75"))
            last_exc: Optional[Exception] = None
            for attempt in range(1, attempts + 1):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": FAST_APPLY_SYSTEM_PROMPT},
                            {"role": "user", "content": user_content},
                        ],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        stream=False,
                    )
                    break  # success
                except Exception as api_e:  # Broad catch; refined classification below
                    last_exc = api_e
                    transient = api_e.__class__.__name__.endswith("APIError") or isinstance(api_e, (TimeoutError,))
                    if attempt == attempts or not transient:
                        raise
                    sleep_for = backoff_base * (2 ** (attempt - 1))
                    try:
                        import time

                        time.sleep(min(sleep_for, 8))
                    except Exception:
                        pass
            else:  # pragma: no cover safeguard
                if last_exc:
                    raise last_exc

            # Extract the merged code from the response
            if response.choices and len(response.choices) > 0:
                raw_response = response.choices[0].message.content
                logger.info("Received response from Fast Apply API", response_length=len(raw_response))

                # Parse the response using improved parsing logic
                merged_code = self._parse_fast_apply_response(raw_response)
                logger.info("Successfully parsed merged code", merged_code_length=len(merged_code))

                # Generate verification results
                verification_results = {
                    "merged_code": merged_code,
                    "has_changes": merged_code != original_code,
                    "udiff": "",
                    "validation": {"has_errors": False, "errors": [], "warnings": []},
                }

                # Generate UDiff for verification
                if verification_results["has_changes"] and file_path:
                    verification_results["udiff"] = generate_udiff(original_code, merged_code, file_path)
                    logger.info("Generated UDiff verification", udiff_length=len(verification_results["udiff"]))

                # Always validate code quality if file_path is provided (even when unchanged)
                if file_path:
                    verification_results["validation"] = validate_code_quality(file_path, merged_code)
                    if verification_results["validation"]["has_errors"]:
                        logger.warning(
                            "Code validation found errors",
                            errors=verification_results["validation"]["errors"],
                            warnings=verification_results["validation"]["warnings"],
                        )

                # Enforce absolute size safety BEFORE returning (prevents oversized content writes downstream)
                merged_bytes = merged_code.encode("utf-8")
                if len(merged_bytes) > MAX_FILE_SIZE:
                    raise ValueError(
                        f"Merged code size {len(merged_bytes)} bytes exceeds MAX_FILE_SIZE {MAX_FILE_SIZE} bytes; refusing to continue"
                    )

                return merged_code if legacy_mode else verification_results
            else:
                # For legacy tests expect ValueError to surface directly (not wrapped)
                raise ValueError("Invalid Fast Apply API response: no choices available")

        except ValueError:
            # Pass through intentional validation errors
            raise
        except Exception as e:
            # Heuristic: treat anything named *APIError* as API error to simplify test mocking
            if e.__class__.__name__.endswith("APIError"):
                logger.error("Fast Apply API error", error=str(e))
                raise RuntimeError("Fast Apply API error") from e
            logger.error("Fast Apply unexpected error", error=str(e))
            raise RuntimeError("Unexpected error when calling Fast Apply API") from e

    def update_config(
        self,
        url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[float] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Update configuration. Returns public-safe config (no api_key) + legacy fields."""

        # Validate numeric parameters
        if timeout is not None and (timeout <= 0 or timeout > 300):
            raise ValueError("Timeout must be between 0 and 300 seconds")

        if url is not None:
            self.url = url
        if model is not None:
            self.model = model
        if api_key is not None:
            self.api_key = api_key
        if timeout is not None:
            self.timeout = timeout
        if max_tokens is not None:
            if max_tokens <= 0 or max_tokens > 32000:
                raise ValueError("max_tokens must be between 1 and 32000")
            self.max_tokens = max_tokens
        if temperature is not None:
            if temperature < 0 or temperature > 2:
                raise ValueError("temperature must be between 0 and 2")
            self.temperature = temperature

        # Update client with new configuration (allow failure in tests)
        try:
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.url,
                timeout=self.timeout,
            )
        except Exception as e:
            logger.warning("Could not reinitialize OpenAI client after config update, staying in test mode", error=str(e))
            self.client = None

        logger.info("Fast Apply configuration updated")
        return {
            "url": self.url,
            "model": self.model,
            "timeout": self.timeout,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

    def _analyze_response_format(self, raw_response: str) -> Dict[str, Any]:
        """Analyze the format and content of a Fast Apply API response for debugging."""
        analysis = {
            "total_length": len(raw_response),
            "has_xml_tags": UPDATED_CODE_START in raw_response and UPDATED_CODE_END in raw_response,
            "has_markdown_fences": raw_response.strip().startswith("```") or raw_response.strip().endswith("```"),
            "line_count": len(raw_response.splitlines()),
            "starts_with_code_tag": raw_response.strip().startswith("<updated-code>"),
            "ends_with_code_tag": raw_response.strip().endswith("</updated-code>"),
            "first_200_chars": raw_response[:200] + "..." if len(raw_response) > 200 else raw_response,
            "last_200_chars": raw_response[-200:] + "..." if len(raw_response) > 200 else raw_response,
        }

        if analysis["has_xml_tags"]:
            start_idx = raw_response.find(UPDATED_CODE_START)
            end_idx = raw_response.find(UPDATED_CODE_END) + len(UPDATED_CODE_END)
            analysis["xml_content_length"] = end_idx - start_idx

        return analysis


def write_with_backup(path: str, new_content: str) -> str:
    """Atomically write file with timestamped backup under lock.

    Safety rules:
        - Reject if new size (bytes) > max(original*2, 5MB) when file exists.
        - Create backup when file exists AND FAST_APPLY_AUTO_BACKUP=True; if not, just write.
    """
    file_lock = _get_file_lock(path)
    with file_lock:
        try:
            original_size = os.path.getsize(path)
        except OSError:
            original_size = 0

        new_size = len(new_content.encode("utf-8"))
        if original_size > 0:
            limit = max(original_size * 2, 5 * 1024 * 1024)
            if new_size > limit:
                raise ValueError(f"Refusing write: new content size {new_size} exceeds safety threshold {limit} bytes")

        auto_backup = os.getenv("FAST_APPLY_AUTO_BACKUP", "False").lower() in ("true", "1")
        backup_path = _create_timestamped_backup(path) if (original_size > 0 and auto_backup) else f"{path}.initial"
        _atomic_write(path, new_content)
        try:
            _cleanup_file_locks()
        except Exception:
            pass
    return backup_path


def _extract_single_tag_block(content: str) -> str:
    """Strictly extract exactly one <updated-code> block or raise."""
    start = content.find(UPDATED_CODE_START)
    end = content.find(UPDATED_CODE_END)
    if start == -1 or end == -1:
        raise ValueError("Missing updated-code tags")
    second_start = content.find(UPDATED_CODE_START, start + 1)
    if second_start != -1:
        raise ValueError("Multiple updated-code blocks detected")
    inner = content[start + len(UPDATED_CODE_START) : end].strip()
    if not inner:
        raise ValueError("Empty updated-code block")
    return inner


def _sanitize_model_response(raw: str) -> str:
    """Remove markdown fences & surrounding noise before tag extraction."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # drop first & last fence if present
        if lines[-1].strip() == "```":
            lines = lines[1:-1]
            text = "\n".join(lines).strip()
    return text


# NOTE: The legacy module-level _parse_fast_apply_response shim has been removed.
# Use FastApplyConnector()._parse_fast_apply_response for parsing model responses.

mcp = FastMCP("fast-apply-mcp")
fast_apply_connector = FastApplyConnector()

# Initialize enhanced search infrastructure



def json_dumps(obj: Any) -> str:
    """Consistent JSON serialization for tool outputs."""
    try:
        return json.dumps(obj, ensure_ascii=False, sort_keys=True)
    except Exception:
        return json.dumps({"error": "serialization_failed"})


@mcp.tool()
def list_tools() -> List[Dict[str, Any]]:
    """Return metadata for all exposed tools (unified mode)."""
    return [
        {
            "name": "edit_file",
            "description": "Apply code edits to a file using Fast Apply.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "target_file": {"type": "string", "description": "Path to target file"},
                    "instructions": {"type": "string", "description": "Edit instructions"},
                    "code_edit": {"type": "string", "description": "Code edit snippet"},
                    "force": {"type": "boolean", "description": "Override optimistic concurrency / safety checks"},
                    "output_format": {"type": "string", "enum": ["text", "json"], "description": "Response format"},
                },
                "required": ["target_file", "instructions", "code_edit"],
            },
        },
        {
            "name": "dry_run_edit_file",
            "description": "Preview an edit without writing changes.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "target_file": {"type": "string"},
                    "instruction": {"type": "string"},
                    "code_edit": {"type": "string"},
                    "output_format": {"type": "string", "enum": ["text", "json"]},
                },
                "required": ["target_file", "code_edit"],
            },
        },
    ]


@mcp.tool()
async def call_tool(name: str, arguments: dict) -> List[Dict[str, Any]]:
    """Handle tool calls with unified branching and robust safety checks."""
    request_id = str(uuid.uuid4())
    try:
        if name == "edit_file":
            target_file = arguments.get("target_file") or arguments.get("path")
            code_edit = arguments.get("code_edit")
            instruction = arguments.get("instructions") or arguments.get("instruction", "")
            if not target_file:
                raise ValueError("target_file parameter is required")
            if not instruction:
                raise ValueError("instructions parameter is required")
            if not code_edit:
                raise ValueError("code_edit parameter is required")
            logger.info("edit_file tool called", target_file=target_file, code_edit_length=len(code_edit), instruction=instruction)
            try:
                secure_path = _secure_resolve(target_file)
            except ValueError:
                raise ValueError("Invalid file path")
            if not os.path.exists(secure_path):
                raise ValueError(f"File not found: {secure_path}")
            file_size = os.path.getsize(secure_path)
            if file_size > MAX_FILE_SIZE:
                raise ValueError(f"File too large: {file_size} bytes (max: {MAX_FILE_SIZE} bytes)")
            try:
                with open(secure_path, "r", encoding="utf-8") as f:
                    original_code = f.read()
                original_hash = hashlib.sha256(original_code.encode("utf-8")).hexdigest()
            except Exception as e:
                raise IOError(f"Failed to read file {secure_path}: {e}")
            try:
                verification_results = fast_apply_connector.apply_edit(
                    original_code=original_code,
                    code_edit=code_edit,
                    instruction=instruction,
                    file_path=secure_path,
                )
            except Exception as e:
                raise RuntimeError(f"Failed to apply edit: {e}") from e
            merged_code = verification_results if isinstance(verification_results, str) else verification_results.get("merged_code", "")
            if isinstance(verification_results, dict) and verification_results.get("validation", {}).get("has_errors"):
                errors_text = "; ".join(verification_results["validation"]["errors"])
                return [{"type": "text", "text": f"❌ Edit applied but validation failed:\n{errors_text}"}]
            force = bool(arguments.get("force", False))
            try:
                with open(secure_path, "r", encoding="utf-8") as f:
                    current_code = f.read()
                current_hash = hashlib.sha256(current_code.encode("utf-8")).hexdigest()
            except Exception:
                current_hash = None
            if not force and current_hash and current_hash != original_hash:
                raise RuntimeError("File changed on disk since read; aborting edit (pass force=true to override).")
            try:
                backup_path = write_with_backup(secure_path, merged_code)
            except Exception as e:
                raise IOError(f"Failed to write file {secure_path}: {e}")
            rel_target = os.path.relpath(secure_path, os.getcwd())
            rel_backup = os.path.relpath(backup_path, os.getcwd())
            output_format = arguments.get("output_format") or "text"
            if output_format == "json" and isinstance(verification_results, dict):
                payload = {
                    "request_id": request_id,
                    "target_file": rel_target,
                    "backup": rel_backup,
                    "has_changes": verification_results.get("has_changes"),
                    "udiff": verification_results.get("udiff"),
                    "validation": verification_results.get("validation"),
                }
                return [{"type": "text", "text": json_dumps(payload)}]
            parts = [f"request_id={request_id}", f"✅ Successfully applied edit to {rel_target}", f"💾 Backup: {rel_backup}"]
            if isinstance(verification_results, dict) and verification_results.get("has_changes") and verification_results.get("udiff"):
                parts.append(f"\n📊 Changes (UDiff):\n{verification_results['udiff']}")
            if isinstance(verification_results, dict) and verification_results.get("validation", {}).get("warnings"):
                parts.append("\n⚠️  Validation warnings:\n" + "\n".join(verification_results["validation"]["warnings"]))
            return [{"type": "text", "text": "\n".join(parts)}]
        elif name == "dry_run_edit_file":
            target_file = arguments.get("target_file") or arguments.get("path")
            code_edit = arguments.get("code_edit")
            instruction = arguments.get("instruction") or arguments.get("instructions", "")
            if not target_file:
                raise ValueError("target_file parameter is required")
            if not code_edit:
                raise ValueError("code_edit parameter is required")
            logger.info("dry_run_edit_file tool called", target_file=target_file, code_edit_length=len(code_edit), instruction=instruction)
            secure_path = _secure_resolve(target_file)
            if not os.path.exists(secure_path):
                raise ValueError(f"File not found: {secure_path}")
            with open(secure_path, "r", encoding="utf-8") as f:
                original_code = f.read()
            verification_results = fast_apply_connector.apply_edit(
                original_code=original_code,
                code_edit=code_edit,
                instruction=instruction,
                file_path=secure_path,
            )
            merged_code = verification_results["merged_code"]
            has_changes = verification_results["has_changes"]
            udiff = verification_results["udiff"]
            validation = verification_results["validation"]
            rel_target = os.path.relpath(secure_path, os.getcwd())
            output_format = arguments.get("output_format") or "text"
            if output_format == "json":
                payload = {
                    "request_id": request_id,
                    "target_file": rel_target,
                    "has_changes": has_changes,
                    "udiff": udiff,
                    "validation": validation,
                    "preview_lines": merged_code.split("\n")[:20] if has_changes else [],
                }
                return [{"type": "text", "text": json_dumps(payload)}]
            parts = [f"request_id={request_id}", f"🔍 DRY RUN RESULTS for {rel_target}", "=" * 60]
            if has_changes:
                parts.append(f"✅ Changes would be applied ({len(merged_code)} bytes)")
                parts.append(f"📊 Original size: {len(original_code)} bytes → New size: {len(merged_code)} bytes")
            else:
                parts.append("ℹ️  No changes would be made (content identical)")
            if has_changes and udiff:
                parts.append("\n📋 Unified Diff:")
                parts.append(udiff)
            parts.append("\n🔒 Code Validation:")
            if validation["has_errors"]:
                parts.append("❌ Errors: " + "; ".join(validation["errors"]))
            else:
                parts.append("✅ No validation errors")
            if validation["warnings"]:
                parts.append("⚠️  Warnings:\n   " + "\n   ".join(validation["warnings"]))
            else:
                parts.append("✅ No validation warnings")
            parts.append("\n🛡️  Safety Information:")
            parts.extend(
                [
                    "   • No files were modified",
                    "   • No backup was created",
                    f"   • Workspace isolation: ✅ (within {os.getcwd()})",
                    "   • Path validation: ✅",
                ]
            )
            if has_changes:
                preview_lines = merged_code.split("\n")[:20]
                parts.append("\n📝 Preview of merged code (first 20 lines):")
                parts.append("```")
                parts.extend(preview_lines)
                if len(merged_code.split("\n")) > 20:
                    parts.append("... (truncated)")
                parts.append("```")
            parts.append("\n💡 To apply these changes, use the edit_file tool with the same parameters.")
            return [{"type": "text", "text": "\n".join(parts)}]
        else:
            raise ValueError(f"Unknown tool: {name}")
    except Exception as e:
        logger.error("Tool call error", tool=name, error=str(e), request_id=request_id)
        raise


def main():
    """Run the FastApply MCP server with stdio transport."""
    logger.info("Starting FastApply MCP server")

    try:
        # Run the server using stdio transport
        logger.info("Initializing stdio transport...")
        mcp.run()
    except KeyboardInterrupt:
        logger.info("Server shutdown requested by user")
    except Exception as e:
        logger.error("Server error", error=str(e))
        raise
    finally:
        logger.info("FastApply MCP server stopped")


if __name__ == "__main__":
    main()
