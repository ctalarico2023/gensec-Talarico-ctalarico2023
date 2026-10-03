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


def build_agent():
    """Create the OpenRouter-backed agent and make the terminal tool available."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not set.")

    model = ChatOpenAI(
        model="openrouter/free",
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )
    return create_agent(model=model, tools=[terminal])


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
