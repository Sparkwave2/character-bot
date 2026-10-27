"""MCP-style commands a character can issue, plus parsing of the model output.

Commands are exposed as OpenAI/MCP tool schemas (llama-server --jinja turns them
into Qwen's native <tool_call> format). If the model writes the tool call into
plain text instead, the same <tool_call>{json}</tool_call> blocks are parsed."""
import json
import re

def _tool(tool_name, description, optional=(), **params):
    """A tool schema. Every definition is sent with every prompt, so the wording is kept short;
    `params` maps argument -> (json type, description or None)."""
    props = {k: {"type": t, **({"description": d} if d else {})} for k, (t, d) in params.items()}
    schema = {"type": "object", "properties": props}
    required = [k for k in params if k not in optional]
    if required:
        schema["required"] = required
    return {"type": "function", "function": {"name": tool_name, "description": description, "parameters": schema}}


ALL_TOOLS = [
    _tool("add_reaction", "React to message #N with an emoji (several emojis = several reactions).",
          message_ref=("integer", "N from [#N]"), emoji=("string", None)),
    _tool("leave_channel", "Leave the channel (say goodbye in your reply).", optional=("reason",),
          reason=("string", None)),
    _tool("stay_silent", "Don't reply. Only when the message is between others and not about you.",),
    _tool("read_note", "Read one of your notes.", name=("string", None)),
    _tool("write_note", "Save something to remember (facts about people, plans, promises) in your notebook.",
          optional=("mode",), name=("string", "e.g. people.md"), text=("string", None),
          mode=("string", "append (default), replace or delete")),
    _tool("view_profile_picture", "See someone's profile picture, only if asked or it matters.",
          name=("string", "display name")),
    _tool("web_search", "Search the web for facts or news.", query=("string", None)),
    _tool("read_webpage", "Read a web page (or see an image link).", url=("string", None)),
    _tool("draw_image", "Draw a picture and post it; you'll see it.",
          prompt=("string", "English description: subject, look, setting, style")),
    _tool("post_gif", "Post a GIF matching a search.", query=("string", "e.g. happy dance")),
]
TOOL_NAMES = [t["function"]["name"] for t in ALL_TOOLS]
# what each switch on the Behavior page says, and whether it is on by default
TOOL_LABELS = {
    "add_reaction": ("React with emojis", True), "leave_channel": ("Leave on their own", True),
    "stay_silent": ("Stay silent", True), "read_note": ("Read notes", True),
    "write_note": ("Write and delete notes", True),
    "view_profile_picture": ("View profile pictures", True), "web_search": ("Search the web", True),
    "read_webpage": ("Read web pages", True), "draw_image": ("Draw pictures (Drawing page)", False),
    "post_gif": ("Post GIFs (needs a KLIPY key)", True),
}


def tool_list(enabled, exclude=()):
    """`enabled(name) -> bool` decides which commands the character gets."""
    return [t for t in ALL_TOOLS if enabled(t["function"]["name"]) and t["function"]["name"] not in exclude]


def tools_tokens(tools):
    """Rough prompt size of the tool definitions (the chat template puts them in the system prompt)."""
    return int(len(json.dumps(tools, ensure_ascii=False)) / 3.5) + 60 if tools else 0


def tool_text_instructions(tools):
    lines = ["You can run commands by writing a block like:",
             '<tool_call>{"name": "add_reaction", "arguments": {"message_ref": 12, "emoji": "😂"}}</tool_call>',
             "Available commands:"]
    for t in tools:
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
