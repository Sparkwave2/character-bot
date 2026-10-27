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
_ACT = r"\*[^*\n]{0,60}\b(?:%s)\b[^*\n]{0,60}\*"          # inside an *action*
CLAIMS = [
    ("add_reaction", {"add_reaction"}, re.compile(
        _ACT % r"react(?:s|ed|ing)?" + r"|\breact(?:s|ed)? with\b", re.I)),
    ("write_note", {"write_note"}, re.compile(
        _ACT % r"writes|jots|scribbles|takes (?:a )?notes?|notes (?:it|that|this) down" +
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
        "write_note": lambda: f'wrote in their note "{a("name")}"',
        "delete_note": lambda: f'deleted their note "{a("name")}"',
        "read_note": lambda: f'read their note "{a("name")}"',
        "list_notes": lambda: "looked through their notes",
        "get_current_time": lambda: "checked the time",
        "web_search": lambda: f'searched the web for "{a("query")}"',
        "read_webpage": lambda: f"read the web page {a('url')}",
        "view_profile_picture": lambda: f"looked at {a('name')}'s profile picture",
        "draw_image": lambda: f'drew a picture: "{a("prompt")}"',
        "post_gif": lambda: f'posted a GIF ("{a("query")}")',
    }.get(name, lambda: None)()   # reactions and leaving are recorded as events already


async def run_turn(llm, msgs, tools, char_name, on_react=None, on_pfp=None, handlers=None):
    """`on_react(args) -> str` performs a reaction and returns a result line.
    `on_pfp(name) -> list[content parts]` returns the picture (or a text part).
    `handlers` maps other command names to `fn(args) -> str` or an async one (notes, time, web)."""
    handlers = handlers or {}
    available = {t["function"]["name"] for t in tools}
    res = TurnResult()
    texts, reacted, checked, fallback = [], set(), False, ""
    for round_no in range(MAX_ROUNDS):
        reply = await llm.chat(msgs, tools)
        text, calls = parse_reply(reply, char_name)
        if text:
            texts.append(text)
        if not calls:
            # one chance to actually do what it only wrote about
            claims = [] if checked or round_no >= MAX_ROUNDS - 2 or not text else \
                unfulfilled_claims(text, {n for n, _ in res.actions}, available)
            checked = True
            if not claims:
                break
            log(f"[tools] {char_name} wrote {claims[0][1]!r} without using {claims[0][0]}; asking again")
            texts.pop()
            fallback = text
            msgs.append({"role": "assistant", "content": reply.get("content") or text})
            said = "; ".join(f'"{p}" ({c})' for c, p in claims)
            msgs.append({"role": "user", "content":
                         f"[system] Your message has not been posted yet. You wrote {said}, but writing it "
                         "doesn't do anything: you didn't use the command, so it didn't happen. If you meant "
                         "it, use the command now. Then write your whole message again (it replaces the one "
                         "above), without describing commands you didn't use."})
            continue
        res.actions += [(n, a) for n, a, _ in calls]
        names = {n for n, _, _ in calls}
        if "stay_silent" in names:
            res.silent = True
            return res

        results, images = [], []
        for name, args, _ in calls:
            if name == "add_reaction":
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
