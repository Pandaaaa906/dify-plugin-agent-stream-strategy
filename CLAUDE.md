# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Dify Plugin** (type: `agent-strategy`) that implements `StreamingReactAgentStrategy` - an advanced agent strategy using Content Blocks format inspired by Anthropic Claude API.

Key features:
- Real-time streaming of final answers for responsive UX
- Parallel tool execution support
- Visible thinking process in logs
- Robust JSON parsing with streaming support via `json_repair`
- Supports both Dify tools and MCP (Model Context Protocol) tools via SSE/Streamable HTTP

## Development Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the plugin locally (requires .env configuration)
python -m main

# Run tests
pytest

# Package the plugin for distribution
dify plugin package -o outs/agent-stream-strategy.difypkg .
```

## Debug Configuration

Copy `.env.example` to `.env` and configure:
```
INSTALL_METHOD=remote
REMOTE_INSTALL_URL=debug.dify.ai:5003
REMOTE_INSTALL_KEY=your-debug-key
```

Then run `python -m main` and refresh your Dify instance to see the plugin (marked as "debugging").

## Architecture

### Plugin Entry Point
- `main.py` - Creates the Plugin instance with `DifyPluginEnv(MAX_REQUEST_TIMEOUT=120)`

### Manifest & Provider Configuration
- `manifest.yaml` - Plugin metadata, permissions, and strategy registration
- `provider/agent.yaml` - Strategy identity and label definitions
- `provider/agent.py` - Minimal `LanggeniusAgentProvider` class extending `AgentProvider`

### Core Strategy Implementation
- `strategies/streaming_react.py` - Main `StreamingReactAgentStrategy` class:
  - Extends `AgentStrategy` with mixins: `AgentScratchpadStorageMixin`, `FilterHistoryMessageByModelFeaturesMixin`
  - Parameters defined in `StreamingReactParams` (pydantic model)
  - Main loop handles tool execution, LLM streaming, and scratchpad management
  - Supports both Dify tools and MCP tools

### Content Parsing
- `output_parser/streaming_content_parser.py` - `StreamingContentParser` class:
  - Parses streaming JSON with Content Blocks format
  - Yields `ThinkingBlock`, `ToolUseBlock`, `TextDelta`, `StopReason`
  - Uses `json_repair` with `stream_stable=True` for robust parsing
  - Tracks partial state for incremental text streaming

### Utilities
- `utils/agent_scratchpad_storage.py` - `AgentScratchpadStorageMixin`:
  - Persists scratchpad across invocations using Dify Plugin storage API
  - Storage key format: `agent_scratchpad:{app_id}:{conversation_id}`
  - Max 5 entries retained (configurable via `_max_scratchpad`)

- `utils/mcp_client.py` - MCP client implementations:
  - `McpSseClient` - HTTP with SSE transport
  - `McpStreamableHttpClient` - Streamable HTTP transport
  - `McpClients` - Multi-server tool management with parallel fetching
  - Supports tools, resources (as tools), and prompts (as tools)

- `utils/filter_history_message.py` - `FilterHistoryMessageByModelFeaturesMixin`:
  - Filters history messages based on model features (vision, audio, etc.)

- `utils/types/__init__.py` - TypedDict definitions for scratchpad entries

### Prompt Templates
- `prompt/streaming_content_template.py` - System prompt templates defining the JSON response format expected from the LLM

### Response Format
The LLM is instructed to respond with this JSON structure:
```json
{
  "content": [
    {"type": "thinking", "thinking": "reasoning..."},
    {"type": "tool_use", "id": "tool_1", "name": "...", "input": {...}},
    {"type": "text", "text": "final answer"}
  ],
  "stop_reason": "tool_use|end_turn|max_tokens"
}
```

## Testing

- `tests/test_stream_parser.py` - Unit tests for `StreamingContentParser`
- `tests/fixtures/streaming_samples.py` - Test data samples
- `tests/utils.py` - `mock_llm_stream()` helper for simulating LLM token-by-token output

## Important Implementation Notes

1. **Mixin Import Issue**: The code imports `FilterHistoryMessageByModelFeaturesMixin` from `utils.base`, but the class is defined in `utils/filter_history_message.py`. The import path in `strategies/streaming_react.py` may need correction.

2. **MCP Tool Execution**: Tools from MCP servers are executed via `mcp_clients.execute_tool()`. The `McpClients` class maintains a mapping of tool names to actions (`ToolAction`) to route calls to the correct server.

3. **Scratchpad Persistence**: The scratchpad is stored via Dify's storage API and retrieved on each invocation keyed by conversation_id, enabling multi-turn conversations.

4. **Streaming Parser State**: The parser maintains internal state (`buffer`, `_text_progress`, etc.) across `feed()` calls to handle partial JSON from streaming LLM responses.

5. **Tool Result Formatting**: Dify tool results and MCP tool results have different formats - see `_format_dify_tool_result()` and `_format_mcp_result()` methods.
