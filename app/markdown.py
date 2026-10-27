"""Discord-flavoured markdown -> styled text segments for the local chat window.

parse(text) returns [(text, tags, url)] where tags is a frozenset of: b i u s code codeblock h1 h2 h3
sub quote spoiler, and url is set on links."""
import re

CODEBLOCK_RE = re.compile(r"```(?:([\w+#.-]{1,20})\n)?\n?(.*?)```", re.S)
HEADING_RE = re.compile(r"^(#{1,3}) (.+)$")
INLINE_RE = re.compile(
    r"\\(?P<esc>[*_~`|\\>#\-\[\]])"
    r"|`(?P<code>[^`\n]+)`"
    r"|\[(?P<ltext>[^\]\n]+)\]\((?P<lurl>https?://[^)\s]+)\)"
    r"|<?(?P<url>https?://[^\s<>]*[^\s<>.,!?;:'\")\]])>?"
    r"|\|\|(?P<spoiler>.+?)\|\|"
    r"|\*\*\*(?P<bi>.+?)\*\*\*"
    r"|\*\*(?P<b>.+?)\*\*"
    r"|__(?P<u>.+?)__"
    r"|\*(?P<i>[^\s*](?:.*?[^\s*])?)\*"
    r"|(?<![\w])_(?P<i2>[^\s_](?:.*?[^\s_])?)_(?![\w])"
    r"|~~(?P<s>.+?)~~")
NESTED = {"spoiler": {"spoiler"}, "bi": {"b", "i"}, "b": {"b"}, "u": {"u"}, "i": {"i"}, "i2": {"i"}, "s": {"s"}}


def inline(text, tags=frozenset()):
    out, pos = [], 0
    for m in INLINE_RE.finditer(text):
        if m.start() > pos:
            out.append((text[pos:m.start()], tags, None))
        kind = m.lastgroup
        if kind == "esc":
            out.append((m.group("esc"), tags, None))
        elif kind == "code":
            out.append((m.group("code"), tags | {"code"}, None))
        elif kind in ("ltext", "lurl"):
            out += [(t, tg, m.group("lurl")) for t, tg, _ in inline(m.group("ltext"), tags | {"link"})]
        elif kind == "url":
            out.append((m.group("url"), tags | {"link"}, m.group("url")))
        else:
            out += inline(m.group(kind), tags | NESTED[kind])
        pos = m.end()
    if pos < len(text):
        out.append((text[pos:], tags, None))
    return out


def _lines(text):
    out = []
    lines = text.split("\n")
    for n, line in enumerate(lines):
        tags = frozenset()
        h = HEADING_RE.match(line)
        if h:
            line, tags = h.group(2), frozenset({f"h{len(h.group(1))}"})
        elif line.startswith("-# "):
            line, tags = line[3:], frozenset({"sub"})
        elif line.startswith("> ") or line == ">":
            line, tags = line[2:], frozenset({"quote"})
            out.append(("▎ ", frozenset({"quote", "bar"}), None))
        m = re.match(r"^(\s*)[-*] (.*)$", line)
        if m:
            out.append((m.group(1) + "  • ", tags, None))
            line = m.group(2)
        out += inline(line, tags)
        if n < len(lines) - 1:
            out.append(("\n", frozenset(), None))
    return out


def parse(text):
    out, pos = [], 0
    for m in CODEBLOCK_RE.finditer(text or ""):
        before = text[pos:m.start()]
        if before:
            out += _lines(before.rstrip("\n") if before.endswith("\n") else before)
            out.append(("\n", frozenset(), None))
        # the line break belongs to the block, so its background fills the last line too
        out.append((m.group(2).rstrip("\n") + "\n", frozenset({"codeblock"}), None))
        pos = m.end()
        if pos < len(text) and text[pos] == "\n":
            pos += 1
    if pos < len(text or ""):
        out += _lines(text[pos:])
    return [s for s in out if s[0]]
