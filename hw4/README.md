# Homework 04 Local Target

This Flask application is an intentionally vulnerable, local-only target for
security-testing-agent demonstrations. Its routes contain hostile prompt text,
a fake seeded secret, and a harmless input-injection marker. It uses synthetic
data only, executes no commands or SQL, and binds only to `127.0.0.1` on port
`5000`.

From this directory, run:

```powershell
python target.py
```

Visit `http://127.0.0.1:5000/` to list the routes. The input-injection route
accepts the fixed demo payload `demo' OR '1'='1` as the `payload` query value;
other values are rejected. Keep all demonstrations on this local target.

## MCP Filesystem integration

The project uses the official
[`@modelcontextprotocol/server-filesystem`](https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem)
package, pinned to version `1.0.0`. The launch configuration is in
`filesystem-server.json`; it passes only the absolute `hw4/reports/` directory
as the server's allowed directory. It does not enable MCP Roots.

The Filesystem server has tools that can modify files, so the future agent must
filter its tool list and expose only `read_text_file` to the LLM. It must not
expose write, edit, delete, move, or other Filesystem tools. The agent should
use the server only to read generated reports from `hw4/reports/`.

In Windows PowerShell, the equivalent launch command is:

```powershell
npx.cmd -y @modelcontextprotocol/server-filesystem@1.0.0 "C:\Users\colet\Downloads\gensec-Talarico-ctalarico2023\hw4\reports"
```
