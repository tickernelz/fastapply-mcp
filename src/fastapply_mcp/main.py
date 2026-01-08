"""
FastApply MCP Server - Simplified Implementation

A streamlined MCP server for efficient code editing using FastApply.
Inspired by opencode-fast-apply's simplicity and partial editing approach.
"""

import difflib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

import openai
import structlog
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()
logger = structlog.get_logger("fast-apply-mcp")

# Simplified configuration - only essential settings
FAST_APPLY_URL = os.getenv("FAST_APPLY_URL", "http://localhost:1234/v1")
FAST_APPLY_MODEL = os.getenv("FAST_APPLY_MODEL", "fastapply-1.5b")
FAST_APPLY_API_KEY = os.getenv("FAST_APPLY_API_KEY", "optional-api-key")
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Official Fast Apply prompts
SYSTEM_PROMPT = """You are a coding assistant that helps merge code updates, ensuring every modification is fully integrated."""

USER_PROMPT = """Merge all changes from the <update> snippet into the <code> below.

RULES:
- Preserve the code's structure, order, comments, and indentation exactly.
- Output only the updated code, enclosed within <updated-code> and </updated-code> tags.
- Do not include any additional text, explanations, placeholders, markdown, ellipses, or code fences.

<code>{original_code}</code>

<update>{code_edit}</update>

Provide the complete updated code."""


def escape_xml_tags(text: str) -> str:
    """Escape XML special characters to prevent prompt injection."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def unescape_xml_tags(text: str) -> str:
    """Unescape XML special characters."""
    return (
        text.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&apos;", "'")
        .replace("&amp;", "&")
    )


def extract_updated_code(raw_response: str) -> str:
    """Extract code from <updated-code> tags."""
    start_tag = "<updated-code>"
    end_tag = "</updated-code>"

    start = raw_response.find(start_tag)
    end = raw_response.find(end_tag)

    if start == -1 or end == -1:
        raise ValueError("Missing <updated-code> tags in response")

    # Check for multiple blocks
    second_start = raw_response.find(start_tag, start + 1)
    if second_start != -1 and second_start < end:
        raise ValueError("Multiple <updated-code> blocks detected")

    code = raw_response[start + len(start_tag):end].strip()

    if not code:
        raise ValueError("Empty <updated-code> block")

    return unescape_xml_tags(code)


def generate_unified_diff(filepath: str, original: str, modified: str) -> str:
    """Generate unified diff between original and modified code."""
    diff_lines = difflib.unified_diff(
        original.splitlines(keepends=True),
        modified.splitlines(keepends=True),
        fromfile=filepath,
        tofile=filepath,
        lineterm=""
    )
    return "".join(diff_lines)


def count_changes(diff: str) -> Dict[str, int]:
    """Count insertions and deletions from unified diff."""
    insertions = 0
    deletions = 0

    for line in diff.split("\n"):
        if line.startswith("+") and not line.startswith("+++"):
            insertions += 1
        elif line.startswith("-") and not line.startswith("---"):
            deletions += 1

    return {"insertions": insertions, "deletions": deletions}


def find_exact_match(haystack: str, needle: str) -> Optional[int]:
    """Find exact match of needle in haystack. Returns start index or None."""
    index = haystack.find(needle)
    return index if index != -1 else None


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace for fuzzy matching (handles CRLF/LF differences)."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def find_normalized_match(haystack: str, needle: str) -> Optional[int]:
    """Find match with normalized whitespace. Returns start index or None."""
    norm_haystack = normalize_whitespace(haystack)
    norm_needle = normalize_whitespace(needle)

    index = norm_haystack.find(norm_needle)
    if index == -1:
        return None

    # Check uniqueness
    second_index = norm_haystack.find(norm_needle, index + 1)
    if second_index != -1:
        raise ValueError(f"Section appears multiple times in file (found at positions {index} and {second_index})")

    return index


def apply_partial_edit(filepath: str, original_code: str, merged_code: str, full_content: str) -> str:
    """Apply partial edit by finding and replacing the original section in full file."""

    # Try exact match first
    start_index = find_exact_match(full_content, original_code)

    if start_index is None:
        # Try normalized match
        start_index = find_normalized_match(full_content, original_code)

        if start_index is None:
            raise ValueError(
                "Cannot locate original_code section in file. "
                "This may be due to:\n"
                "  • The file was modified since you read it\n"
                "  • Whitespace differences (tabs vs spaces)\n"
                "  • The section doesn't exist in the file\n"
                "Suggestion: Re-read the file and try again with the exact content."
            )

    # Replace the section
    end_index = start_index + len(original_code)
    updated_content = full_content[:start_index] + merged_code + full_content[end_index:]

    return updated_content


async def call_fast_apply(original_code: str, code_edit: str) -> str:
    """Call FastApply API to merge code changes."""

    # Escape XML in user content to prevent injection
    escaped_original = escape_xml_tags(original_code)
    escaped_edit = escape_xml_tags(code_edit)

    user_content = USER_PROMPT.format(
        original_code=escaped_original,
        code_edit=escaped_edit
    )

    client = openai.OpenAI(
        api_key=FAST_APPLY_API_KEY,
        base_url=FAST_APPLY_URL,
        timeout=300.0
    )

    logger.info(
        "Calling FastApply API",
        original_length=len(original_code),
        edit_length=len(code_edit)
    )

    try:
        response = client.chat.completions.create(
            model=FAST_APPLY_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ],
            temperature=0.0,  # Deterministic output
            stream=False
        )

        if not response.choices:
            raise ValueError("No response from FastApply API")

        raw_response = response.choices[0].message.content
        merged_code = extract_updated_code(raw_response)

        logger.info("Successfully merged code", merged_length=len(merged_code))
        return merged_code

    except Exception as e:
        logger.error("FastApply API error", error=str(e))
        raise RuntimeError(f"FastApply API error: {str(e)}") from e


def is_binary_file(filepath: str) -> bool:
    """Check if file is binary by looking for null bytes."""
    try:
        with open(filepath, "rb") as f:
            chunk = f.read(8192)
            return b"\x00" in chunk
    except Exception:
        return False


# Initialize MCP server
mcp = FastMCP("fast-apply-mcp")


@mcp.tool()
def list_tools() -> List[Dict[str, Any]]:
    """Return metadata for available tools."""
    return [
        {
            "name": "fast_apply_edit",
            "description": "Apply AI-guided code edits to existing files using partial context for efficiency. "
                         "Supports 50-500 line sections with 80-98% token savings compared to full-file edits. "
                         "Uses smart matching to locate and replace code sections automatically.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "target_filepath": {
                        "type": "string",
                        "description": "Path to the file to edit (relative or absolute)"
                    },
                    "original_code": {
                        "type": "string",
                        "description": "The exact section of code to be modified (50-500 lines recommended). "
                                     "Must exist in the target file. Include enough context for unique identification."
                    },
                    "code_edit": {
                        "type": "string",
                        "description": "The changes to apply. Can use '... existing code ...' markers for unchanged sections. "
                                     "Be specific about what to change."
                    }
                },
                "required": ["target_filepath", "original_code", "code_edit"]
            }
        }
    ]


@mcp.tool()
async def call_tool(name: str, arguments: dict) -> List[Dict[str, Any]]:
    """Handle tool calls."""
    request_id = str(uuid.uuid4())

    try:
        if name != "fast_apply_edit":
            raise ValueError(f"Unknown tool: {name}")

        # Extract arguments
        target_filepath = arguments.get("target_filepath")
        original_code = arguments.get("original_code")
        code_edit = arguments.get("code_edit")

        # Validate required arguments
        if not target_filepath:
            raise ValueError("target_filepath is required")
        if not original_code:
            raise ValueError("original_code is required")
        if not code_edit:
            raise ValueError("code_edit is required")

        logger.info(
            "fast_apply_edit called",
            filepath=target_filepath,
            original_length=len(original_code),
            edit_length=len(code_edit)
        )

        # Resolve file path
        if not os.path.isabs(target_filepath):
            target_filepath = os.path.join(os.getcwd(), target_filepath)

        target_filepath = os.path.realpath(target_filepath)

        # Security: ensure file is within workspace
        workspace_root = os.path.realpath(os.getcwd())
        if not target_filepath.startswith(workspace_root):
            raise ValueError(f"File path outside workspace: {target_filepath}")

        # Check file exists
        if not os.path.exists(target_filepath):
            raise ValueError(
                f"File not found: {target_filepath}\n"
                "Suggestion: Use a 'write' tool to create new files."
            )

        # Check if binary
        if is_binary_file(target_filepath):
            raise ValueError(f"Cannot edit binary file: {target_filepath}")

        # Check file size
        file_size = os.path.getsize(target_filepath)
        if file_size > MAX_FILE_SIZE:
            raise ValueError(f"File too large: {file_size} bytes (max: {MAX_FILE_SIZE})")

        # Read full file content
        try:
            with open(target_filepath, "r", encoding="utf-8") as f:
                full_content = f.read()
        except UnicodeDecodeError:
            raise ValueError(f"File encoding error. Expected UTF-8: {target_filepath}")
        except Exception as e:
            raise IOError(f"Failed to read file: {str(e)}")

        # Verify original_code exists in file
        if original_code not in full_content:
            # Try normalized match for better error message
            try:
                find_normalized_match(full_content, original_code)
            except ValueError as e:
                raise ValueError(str(e))

            raise ValueError(
                "Cannot locate original_code in file (whitespace mismatch detected).\n"
                "Suggestion: Ensure tabs/spaces match exactly, or re-read the file."
            )

        # Call FastApply API to merge changes
        try:
            merged_code = await call_fast_apply(original_code, code_edit)
        except Exception as e:
            raise RuntimeError(f"FastApply merge failed: {str(e)}")

        # Apply partial edit to full file
        try:
            updated_content = apply_partial_edit(
                target_filepath,
                original_code,
                merged_code,
                full_content
            )
        except ValueError as e:
            raise ValueError(f"Failed to apply edit: {str(e)}")

        # Generate diff for verification
        diff = generate_unified_diff(target_filepath, full_content, updated_content)
        changes = count_changes(diff)

        # Write updated content atomically
        try:
            # Write to temp file first
            temp_path = target_filepath + ".tmp"
            with open(temp_path, "w", encoding="utf-8") as f:
                f.write(updated_content)

            # Atomic replace
            os.replace(temp_path, target_filepath)

            logger.info(
                "Successfully applied edit",
                filepath=target_filepath,
                insertions=changes["insertions"],
                deletions=changes["deletions"]
            )

        except Exception as e:
            # Clean up temp file if it exists
            if os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass
            raise IOError(f"Failed to write file: {str(e)}")

        # Format response
        rel_path = os.path.relpath(target_filepath, os.getcwd())

        response_parts = [
            f"✅ Successfully edited {rel_path}",
            f"",
            f"📊 Changes:",
            f"  • +{changes['insertions']} insertions",
            f"  • -{changes['deletions']} deletions",
            f"  • Original section: {len(original_code)} bytes",
            f"  • Merged section: {len(merged_code)} bytes",
            f"  • Full file: {len(full_content)} → {len(updated_content)} bytes",
        ]

        # Add diff preview (truncated if too long)
        if diff:
            diff_lines = diff.split("\n")
            if len(diff_lines) > 50:
                preview = "\n".join(diff_lines[:25] + ["... (truncated) ..."] + diff_lines[-25:])
            else:
                preview = diff

            response_parts.extend([
                f"",
                f"📝 Diff:",
                f"```diff",
                preview,
                f"```"
            ])

        response_parts.extend([
            f"",
            f"💡 Token efficiency: Edited {len(original_code)} bytes instead of {len(full_content)} bytes "
            f"({100 - int(len(original_code) / len(full_content) * 100)}% savings)"
        ])

        return [{"type": "text", "text": "\n".join(response_parts)}]

    except Exception as e:
        logger.error("Tool call error", tool=name, error=str(e), request_id=request_id)

        error_message = [
            f"❌ Error: {str(e)}",
            f"",
            f"Request ID: {request_id}"
        ]

        # Add helpful suggestions based on error type
        if "Cannot locate original_code" in str(e):
            error_message.extend([
                f"",
                f"💡 Troubleshooting:",
                f"  1. Re-read the file to get current content",
                f"  2. Ensure original_code matches exactly (including whitespace)",
                f"  3. Provide more context to make the section unique"
            ])
        elif "File not found" in str(e):
            error_message.extend([
                f"",
                f"💡 Suggestion: Use a 'write' tool to create new files"
            ])

        return [{"type": "text", "text": "\n".join(error_message)}]


def main():
    """Run the FastApply MCP server."""
    logger.info("Starting FastApply MCP server (simplified)")
    logger.info(
        "Configuration",
        url=FAST_APPLY_URL,
        model=FAST_APPLY_MODEL,
        max_file_size=MAX_FILE_SIZE
    )

    try:
        mcp.run()
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error("Server error", error=str(e))
        raise
    finally:
        logger.info("FastApply MCP server stopped")


if __name__ == "__main__":
    main()
