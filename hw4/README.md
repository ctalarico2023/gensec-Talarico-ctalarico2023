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

The project will use the official MCP Filesystem server only to read generated
reports. Configure its allowed directory as `hw4/reports/` and no broader path;
the agent should expose only report-reading tools for this integration.
