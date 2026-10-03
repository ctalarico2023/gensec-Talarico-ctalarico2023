"""A minimal terminal-driven LangChain agent backed by OpenRouter."""

import os
import subprocess
import sys
from pathlib import Path

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI


@tool
def terminal(command: str) -> str:
    """Run a shell command from the homework directory and return its output."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return "Command timed out after 30 seconds."

    output = result.stdout
    if result.stderr:
        output += ("\n" if output else "") + result.stderr
    if result.returncode != 0:
        output += ("\n" if output else "") + f"Command exited with status {result.returncode}."
    return output or "Command completed with no output."


@tool
def read_file(relative_path: str) -> str:
    """Read a text file whose path is relative to the homework directory."""
    homework_dir = Path(__file__).resolve().parent
    requested_path = Path(relative_path)

    # Check both separator styles so traversal is rejected consistently.
    path_parts = relative_path.replace("\\", "/").split("/")
    if requested_path.is_absolute() or ".." in path_parts:
        return "Error: file path must stay inside the homework directory."

    file_path = (homework_dir / requested_path).resolve()
    try:
        file_path.relative_to(homework_dir)
    except ValueError:
        return "Error: file path must stay inside the homework directory."

    try:
        return file_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return f"Error: file not found: {relative_path}"
    except IsADirectoryError:
        return f"Error: path is a directory, not a file: {relative_path}"
    except UnicodeDecodeError:
        return f"Error: file is not valid UTF-8 text: {relative_path}"
    except OSError as error:
        return f"Error reading {relative_path}: {error}"


@tool
def list_files() -> str:
    """List files and directories in the homework directory, excluding .venv."""
    homework_dir = Path(__file__).resolve().parent
    paths: list[str] = []

    for current_dir, dir_names, file_names in os.walk(homework_dir, followlinks=False):
        dir_names[:] = sorted(name for name in dir_names if name != ".venv")
        current_path = Path(current_dir)
        paths.extend(
            f"{(current_path / name).relative_to(homework_dir).as_posix()}/"
            for name in dir_names
        )
        paths.extend(
            (current_path / name).relative_to(homework_dir).as_posix()
            for name in file_names
        )

    if not paths:
        return "No files or directories found."
    return "\n".join(sorted(paths))


def build_agent():
    """Create the OpenRouter-backed agent with terminal and file tools."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not set.")

    model = ChatOpenAI(
        model="openrouter/free",
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )
    return create_agent(model=model, tools=[terminal, read_file, list_files])


def main() -> int:
    """Read one prompt from the terminal and print the agent's response."""
    try:
        agent = build_agent()
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1

    prompt = input("Prompt: ").strip()
    if not prompt:
        print("Please enter a prompt.", file=sys.stderr)
        return 1

    try:
        result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    except Exception as error:
        print(f"Agent request failed: {error}", file=sys.stderr)
        return 1

    print(result["messages"][-1].content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
