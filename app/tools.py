"""MCP-style commands a character can issue, plus parsing of the model output.

Commands are exposed as OpenAI/MCP tool schemas (llama-server --jinja turns them
into Qwen's native <tool_call> format). If the model writes the tool call into
plain text instead, the same <tool_call>{json}</tool_call> blocks are parsed."""
import json
import re

TOOLS = [
    {"type": "function", "function": {
        "name": "add_reaction",
        "description": "React to a message with an emoji. Use the message number shown as [#N].",
        "parameters": {"type": "object", "properties": {
            "message_ref": {"type": "integer", "description": "The N from [#N]"},
            "emoji": {"type": "string", "description": "A unicode emoji, e.g. 😂 (several, e.g. 😂🔥, add one "
                                                       "reaction each)"}},
            "required": ["message_ref", "emoji"]}}},
    {"type": "function", "function": {
        "name": "leave_channel",
        "description": "Leave the current channel. Put any goodbye in your normal reply text.",
        "parameters": {"type": "object", "properties": {
            "reason": {"type": "string"}}, "required": []}}},
    {"type": "function", "function": {
        "name": "stay_silent",
        "description": "Rarely needed. Only when the latest message is clearly between other people and "
                       "has nothing to do with you. Never use it when someone talks to you or mentions you.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "get_current_time",
        "description": "Check the current date and time.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "list_notes",
        "description": "List the notes in your personal notebook.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "read_note",
        "description": "Read one of your notes.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string", "description": "Note name, e.g. people.md"}},
            "required": ["name"]}}},
    {"type": "function", "function": {
        "name": "write_note",
        "description": "Write something down in your personal notebook (Markdown) so you remember it later, "
                       "e.g. facts about people, plans or promises. Only you can read your notes.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string", "description": "Note name, e.g. people.md"},
            "content": {"type": "string", "description": "Markdown text to write"},
            "append": {"type": "boolean", "description": "true = add to the end (default), "
                                                          "false = replace the whole note"}},
            "required": ["name", "content"]}}},
    {"type": "function", "function": {
        "name": "delete_note",
        "description": "Delete one of your notes.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string"}}, "required": ["name"]}}},
]

PFP_TOOL = {"type": "function", "function": {
    "name": "view_profile_picture",
    "description": "Look at a person's Discord profile picture. Only when someone asks you to, or their "
                   "picture actually matters to the conversation. You will see it before replying.",
    "parameters": {"type": "object", "properties": {
        "name": {"type": "string", "description": "Their display name"}},
        "required": ["name"]}}}


WEB_TOOLS = [
    {"type": "function", "function": {
        "name": "web_search",
        "description": "Search the internet (DuckDuckGo) for current or factual information you don't know. "
                       "Returns titles, links and short snippets.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "What to search for"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "read_webpage",
        "description": "Read the text of a web page, e.g. a link from search results or one someone posted.",
        "parameters": {"type": "object", "properties": {
            "url": {"type": "string", "description": "Full http(s) link"}},
            "required": ["url"]}}},
]


def tool_list(allow_pfp, allow_web=False):
    return TOOLS + ([PFP_TOOL] if allow_pfp else []) + (WEB_TOOLS if allow_web else [])


def tool_text_instructions(allow_pfp, allow_web=False):
    lines = ["You can run commands by writing a block like:",
             '<tool_call>{"name": "add_reaction", "arguments": {"message_ref": 12, "emoji": "😂"}}</tool_call>',
             "Available commands:"]
    for t in tool_list(allow_pfp, allow_web):
        f = t["function"]
        args = ", ".join(f["parameters"]["properties"].keys())
        lines.append(f"- {f['name']}({args}): {f['description']}")
    return "\n".join(lines)


CUSTOM_EMOJI_RE = re.compile(r"(<a?:\w+:\d+>)")


def _is_ri(ch):
    return 0x1F1E6 <= ord(ch) <= 0x1F1FF   # regional indicator (two make a flag)


def _trailing_ri(s):
    n = 0
    while n < len(s) and _is_ri(s[-1 - n]):
        n += 1
    return n


def split_emojis(value, limit=5):
    """'😂🔥' or '😂, 🔥' -> ['😂', '🔥']. Keeps ZWJ sequences (👨‍👩‍👧), skin tones, flags,
    keycaps and custom <:name:id> emoji together."""
    if isinstance(value, (list, tuple)):
        return [e for v in value for e in split_emojis(v, limit)][:limit]
    out = []
    for part in CUSTOM_EMOJI_RE.split(str(value or "")):
        if CUSTOM_EMOJI_RE.fullmatch(part):
            out.append(part)
            continue
        cur = ""
        for ch in part:
            cp = ord(ch)
            if ch.isspace() or ch in ",;+":
                out.append(cur)
                cur = ""
                continue
            joins = cur and (cp in (0x200D, 0xFE0F, 0xFE0E, 0x20E3) or 0x1F3FB <= cp <= 0x1F3FF
                             or 0xE0020 <= cp <= 0xE007F or cur[-1] == "‍"
                             or (ch.isascii() and ch.isalnum() and cur[-1].isascii() and cur[-1].isalnum())
                             or (_is_ri(ch) and _trailing_ri(cur) % 2 == 1))
            if joins:
                cur += ch
            else:
                out.append(cur)
                cur = ch
        out.append(cur)
    return [e for e in out if e][:limit]


TOOL_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
THINK_RE = re.compile(r"<think>.*?</think>", re.S)
PREFIX_RE = re.compile(r"^\s*(\[#\d+[^\]]*\]\s*)?([^\n:\[\]]{1,40}\(character\)\s*:|[^\n:\[\]]{1,40}:)?\s*")


def _parse_args(a):
    if isinstance(a, dict):
        return a
    try:
        return json.loads(a or "{}")
    except (json.JSONDecodeError, TypeError):
        return {}


def parse_reply(message, char_name):
    """Return (text, [(tool_name, args, call_id), ...]) from a chat-completions message."""
    calls = []
    for tc in message.get("tool_calls") or []:
        fn = tc.get("function", {})
        calls.append((fn.get("name"), _parse_args(fn.get("arguments")), tc.get("id")))
    text = message.get("content") or ""
    if isinstance(text, list):
        text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
    text = THINK_RE.sub("", text)
    if "</think>" in text:  # opening tag was in the template
        text = text.split("</think>", 1)[1]
    for block in TOOL_BLOCK_RE.findall(text):
        try:
            d = json.loads(block)
            calls.append((d.get("name"), _parse_args(d.get("arguments", d.get("parameters"))), None))
        except json.JSONDecodeError:
            pass
    text = TOOL_BLOCK_RE.sub("", text).strip()
    # strip an echoed "[#12] Name:" / "Name:" prefix the model may imitate
    m = PREFIX_RE.match(text)
    if m and m.group(0).strip():
        head = m.group(0)
        name_part = (m.group(2) or "").replace("(character)", "").strip(" :").lower()
        if m.group(1) or name_part == char_name.lower():
            text = text[len(head):]
    return text.strip(), calls
