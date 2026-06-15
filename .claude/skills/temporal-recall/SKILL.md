# Temporal Recall

Surface and navigate prior-session memory when the user references something from the past — "remember when…", "yesterday we…", "last time we tried…", "what did we decide about…".

## When to use

Auto-invoked when the user's message contains temporal references to past work that may live in long-term memory but is not in the current conversation transcript. Examples:

- "Do you remember why we picked X over Y?"
- "Last week you suggested an approach that didn't quite work — what was wrong with it?"
- "Pick up where we left off on the auth refactor"

Do **not** invoke for references to the current session's own history (the current transcript already contains that).

## How to use

The memory store is exposed via four MCP tools registered under `memory-pkg` in `.mcp.json`:

| Tool | What it does |
|---|---|
| `mcp__memory-pkg__searchMemory` | Fuzzy trigram search across all stored events |
| `mcp__memory-pkg__getMemoryContext` | Scale forward/backward in time around a matched event |
| `mcp__memory-pkg__unwindFromEvent` | Replay every event in the session from start up to a chosen anchor |
| `mcp__memory-pkg__getSessionTimeline` | Full chronological dump of one session |

### Typical flow

1. **Search.** Call `searchMemory` with the most concrete identifiers from the user's message (file names, function names, concept names, quoted phrases). Trigram match is forgiving on word order and partial matches.

2. **Identify the anchor.** Look at the top hits and pick the one most likely to be the moment the user is referring to. Use the `event_type`, `tool_name`, `file_path`, `summary`, and timestamp to disambiguate.

3. **Expand.** Once you have an anchor `event_id`:
   - Use `getMemoryContext` (with `before` and `after`) to see what led up to and followed the event.
   - Use `unwindFromEvent` when the user wants the full path of how the session arrived at that moment.
   - Use `getSessionTimeline` when the user wants to scan the whole session, not just a window.

4. **Answer.** Quote the relevant events back to the user. Be explicit that the information comes from prior sessions, not the current one. If the memory references a file or function that may have changed since, **verify it still exists in current code** before acting on the memory.

## Key rules

- **Memory is frozen in time.** If a memory names a file, function, or flag, it was true *when the memory was written*. Verify against current code before acting.
- **Auto-injection already happens.** Every user prompt fires a memory-injection hook that surfaces up to 3 relevant past events as `<memory-context>`. Only call MCP tools when you need *more* than what was auto-injected.
- **The current session already has its own transcript.** Don't use these tools to reference the message the user sent five minutes ago — that's in your context already.
- **Search wide, then narrow.** Start with a single broad `searchMemory` call. Don't fire 4 tool calls in parallel hoping one hits — that's noisy and rarely faster.
