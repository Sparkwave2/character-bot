"""One character turn: call the model, run its commands, feed the results back
and let it continue until it writes its message (MCP-style tool loop)."""
import inspect
import json
import re
from dataclasses import dataclass, field

from .log import log
from .tools import parse_reply, split_emojis

MAX_ROUNDS = 6

# Small models like to narrate a command ("*reacts with 😂*", "let me look that up") instead of
# calling it. Each entry: command to suggest, commands that count as doing it, what gives it away.
class _Act(str):
    """`_ACT % words` matches those words inside an *action* or an (action)."""
    def __mod__(self, words):
        return (r"\*[^*\n]{0,60}\b(?:%s)\b[^*\n]{0,60}\*|\([^()\n]{0,60}\b(?:%s)\b[^()\n]{0,60}\)"
                % (words, words))


_ACT = _Act()
_WILL = r"\b(?:i'?ll|i will|let me|lemme|i'?m gonna|i'?m going to|gonna)\s+"
CLAIMS = [
    ("add_reaction", {"add_reaction"}, re.compile(
        _ACT % r"react\w*|adds? (?:a |an )?\S* ?reaction" + r"|\breact(?:s|ed)? with\b"
        + r"|" + _WILL + r"(?:react|add\s+(?:a|an|my)\s+(?:\S+\s+){0,3}?reactions?|use\s+a\s+tool\s+call)\b"
        r"|\b(?:i\s+)?(?:just\s+)?(?:reacted|added\s+(?:a|an|my)\s+(?:\S+\s+){0,3}?reaction)\b", re.I)),
    ("write_note", {"write_note"}, re.compile(
        _ACT % r"writes|jots|scribbles|takes (?:a )?notes?|notes (?:it|that|this) down" +
        r"|" + _WILL + r"(?:note\s+(?:that|it|this)|make\s+a\s+note|take\s+a\s+note)\b"
        r"|\b(?:i'?ll|let me|lemme|i'?m gonna|gonna)\s+(?:write|jot|note|put)\b[^.!?\n]{0,30}\b(?:down|notes?|notebook)\b"
        r"|\b(?:written|wrote|jotted|noted)\s+(?:that|it|this)\s+down\b"
        r"|\b(?:added|saved|put)\s+(?:that|it|this)\s+(?:to|in(?:to)?)\s+my\s+(?:notes?|notebook)\b"
        r"|\bi'?ll\s+(?:remember\s+(?:that|this|it)|keep\s+(?:that|this|it)\s+in\s+mind)\b", re.I)),
    ("web_search", {"web_search", "read_webpage"}, re.compile(
        _ACT % r"searches|googles|looks (?:it|that|this) up|browses|types" +
        r"|\b(?:let me|lemme|i'?ll|i'?m gonna|gonna)\s+(?:search|google|look\s+(?:it|that|this)\s+up|check\s+online)\b"
        r"|\bi\s+(?:just\s+)?(?:searched|googled|looked\s+(?:it|that|this)\s+up)\b", re.I)),
    ("get_current_time", {"get_current_time"}, re.compile(
        _ACT % r"checks? (?:the time|the clock|(?:his|her|their|my) (?:phone|watch))" +
        r"|\b(?:let me|lemme)\s+check\s+the\s+(?:time|clock|date)\b", re.I)),
    ("draw_image", {"draw_image"}, re.compile(
        _ACT % r"draws|sketches|doodles|paints|scribbles (?:a|an|something)" +
        r"|\bhere'?s\s+(?:my|the|a)\s+(?:drawing|sketch|doodle|painting)\b"
        r"|\b(?:let me|lemme|i'?ll|i'?m gonna|gonna)\s+(?:draw|sketch|doodle|paint)\b", re.I)),
    ("post_gif", {"post_gif"}, re.compile(
        _ACT % r"posts? (?:a )?gif|sends? (?:a )?gif" + r"|\[gif[^\]]*\]", re.I)),
    ("leave_channel", {"leave_channel"}, re.compile(
        _ACT % r"leaves|walks (?:out|away|off)|storms (?:off|out)|heads out|exits|disappears|vanishes", re.I)),
]


@dataclass
class TurnResult:
    text: str = ""
    silent: bool = False
    leave: bool = False
    actions: list = field(default_factory=list)   # [(name, args)] in call order


def unfulfilled_claims(text, used, available):
    """[(command, quoted phrase)] for commands the text says happened but weren't called."""
    text = (text or "").replace("’", "'")
    out = []
    for cmd, done_by, rx in CLAIMS:
        if cmd in available and not (done_by & used):
            m = rx.search(text)
            if m:
                out.append((cmd, m.group(0).strip()))
    return out


def describe_action(name, args):
    """How a command shows up afterwards in the character's own memory (None = not recorded)."""
    a = lambda k: str(args.get(k, "")).strip()[:80]
    return {
        "write_note": lambda: (f'deleted their note "{a("name")}"' if a("mode").lower().startswith("del")
                               else f'wrote in their note "{a("name")}"'),
        "read_note": lambda: f'read their note "{a("name")}"',
        "get_current_time": lambda: "checked the time",
        "web_search": lambda: f'searched the web for "{a("query")}"',
        "read_webpage": lambda: f"read the web page {a('url')}",
        "view_profile_picture": lambda: f"looked at {a('name')}'s profile picture",
        "draw_image": lambda: f'drew a picture: "{a("prompt")}"',
        "post_gif": lambda: f'posted a GIF ("{a("query")}")',
    }.get(name, lambda: None)()   # reactions and leaving are recorded as events already


JUST_REPLY = {"type": "function", "function": {
    "name": "just_reply", "description": "No command now; write your reply.",
    "parameters": {"type": "object", "properties": {}}}}
DECIDE = ("[system] Before you write your reply, decide whether one of your commands fits right now, "
          "for example {examples}. Call the ones you want (several are fine), or call just_reply if none "
          "fit. You write your message right after this.")
EXAMPLES = [("add_reaction", "reacting to a message"), ("write_note", "writing down something worth remembering"),
            ("read_note", "checking your notes"), ("get_current_time", "checking the time"),
            ("web_search", "looking something up"), ("draw_image", "drawing what was asked for"),
            ("post_gif", "posting a GIF"), ("leave_channel", "leaving when you're told to go")]
_decide_broken = False


async def _decide(llm, msgs, tools, char_name):
    """Small models rarely pick a command on their own while they're busy roleplaying. One extra
    call where a command is required (just_reply counts) makes the choice explicit. Returns the
    model's reply if it chose real commands (the question is then kept in `msgs`), else None."""
    global _decide_broken
    if _decide_broken:
        return None
    names = {t["function"]["name"] for t in tools}
    examples = ", ".join(e for n, e in EXAMPLES if n in names) or "using a command"
    question = {"role": "user", "content": DECIDE.format(examples=examples)}
    try:
        reply = await llm.chat(msgs + [question], tools + [JUST_REPLY], tool_choice="required")
    except Exception as e:
        _decide_broken = "rejected" in str(e)   # this server/template can't force tool calls
        log(f"[tools] command check skipped: {e}", level="warn")
        return None
    _text, calls = parse_reply(reply, char_name)
    if not any(n != "just_reply" for n, _, _ in calls):
        return None
    msgs.append(question)
    return reply


async def run_turn(llm, msgs, tools, char_name, on_react=None, on_pfp=None, handlers=None, review=None,
                   decide=False):
    """`on_react(args) -> str` performs a reaction and returns a result line.
    `on_pfp(name) -> list[content parts]` returns the picture (or a text part).
    `handlers` maps other command names to `fn(args) -> str` or an async one (notes, time, web).
    `review(text) -> (note or None, [content parts])` looks at the finished message once; a note
    (e.g. "this link is dead") sends it back for one rewrite.
    `decide` first asks the model which commands it wants (see _decide)."""
    handlers = handlers or {}
    available = {t["function"]["name"] for t in tools}
    res = TurnResult()
    texts, reacted, checked, fallback = [], set(), False, ""
    first = await _decide(llm, msgs, tools, char_name) if decide and tools else None
    for round_no in range(MAX_ROUNDS):
        if first is not None:
            reply, first = first, None
        else:
            reply = await llm.chat(msgs, tools)
        text, calls = parse_reply(reply, char_name)
        for name, args, _ in calls:
            if name != "just_reply":
                log(f"[tools] {char_name} used {name}({json.dumps(args, ensure_ascii=False)[:120]})")
        if text:
            texts.append(text)
        if not calls:
            # one look before posting: commands it only wrote about, and the links it wants to post
            if checked or round_no >= MAX_ROUNDS - 2 or not text:
                break
            checked = True
            claims = unfulfilled_claims(text, {n for n, _ in res.actions}, available)
            note, pictures = await review(text) if review else (None, [])
            if not claims and not note:
                break
            notes = ["[system] Your message has not been posted yet."]
            if claims:
                log(f"[tools] {char_name} wrote {claims[0][1]!r} without using {claims[0][0]}; asking again")
                said = "; ".join(f'"{p}" ({c})' for c, p in claims)
                notes.append(f"You wrote {said}, but writing it doesn't do anything: you didn't use the "
                             "command, so it didn't happen. If you meant it, use the command now.")
            if note:
                notes.append(note)
            notes.append("Then write your whole message again (it replaces the one above)"
                         + (", without describing commands you didn't use." if claims else "."))
            texts.pop()
            fallback = text
            msgs.append({"role": "assistant", "content": reply.get("content") or text})
            body = " ".join(notes)
            msgs.append({"role": "user", "content": [{"type": "text", "text": body}] + pictures
                         if pictures else body})
            continue
        res.actions += [(n, a) for n, a, _ in calls if n != "just_reply"]
        names = {n for n, _, _ in calls}
        if "stay_silent" in names:
            res.silent = True
            return res

        results, images = [], []
        for name, args, _ in calls:
            if name == "just_reply":
                results.append("OK.")
            elif name == "add_reaction":
                results.append(await _react_each(args, reacted, on_react))
            elif name == "view_profile_picture" and on_pfp:
                images += await on_pfp(str(args.get("name", "")))
                results.append("The picture is shown below.")
            elif name == "leave_channel":
                res.leave = True
                results.append("You will leave the channel right after this. "
                               "Write your goodbye message now if you want to say one.")
            elif name in handlers:
                try:
                    out = handlers[name](args)
                    out = await out if inspect.isawaitable(out) else out
                    if isinstance(out, list):   # content parts with a picture (e.g. a drawing)
                        images += out
                        out = "The result is shown below."
                    results.append(out)
                except Exception as e:
                    results.append(f"{name} failed: {e}")
            else:
                results.append(f"Unknown command {name}.")

        if res.leave and texts:
            break  # goodbye already written
        # nothing left to say after a reaction is fine; ask once for the actual reply
        _append_results(msgs, reply, calls, results)
        if images:
            msgs.append({"role": "user", "content": images + [
                {"type": "text", "text": "[system] Now continue with your reply."}]})
        tools = [t for t in tools if t["function"]["name"] != "view_profile_picture"] if images else tools
    res.text = "\n".join(texts).strip()
    if not res.text and not res.silent:
        res.text = fallback   # the retry wrote nothing; keep what it said first
    return res


async def _react_each(args, reacted, on_react):
    """Several emojis in one call ("😂🔥") become one reaction each."""
    emojis = split_emojis(args.get("emoji")) or [args.get("emoji")]
    out = []
    for e in emojis:
        key = (str(args.get("message_ref")), e)
        if key in reacted:
            out.append(f"Already reacted {e}.")
            continue
        reacted.add(key)
        out.append(await on_react({**args, "emoji": e}) if on_react else "Done.")
    return " ".join(out)


def _append_results(msgs, reply, calls, results):
    native = reply.get("tool_calls") or []
    if native:
        msgs.append({"role": "assistant", "content": reply.get("content") or "", "tool_calls": native})
        ids = [tc.get("id") or f"call_{i}" for i, tc in enumerate(native)]
        for i, r in enumerate(results[:len(ids)]):
            msgs.append({"role": "tool", "tool_call_id": ids[i], "content": r})
        if len(results) > len(ids):  # extra calls parsed from text
            msgs.append({"role": "user", "content": "[system] " + " ".join(results[len(ids):])})
    else:
        msgs.append({"role": "assistant", "content": reply.get("content") or ""})
        lines = [f"- {n}({json.dumps(a, ensure_ascii=False)}): {r}" for (n, a, _), r in zip(calls, results)]
        msgs.append({"role": "user", "content": "[system] Command results:\n" + "\n".join(lines)
                     + "\nNow write your message (or nothing if you have nothing to add)."})
