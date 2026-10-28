"""Everything about a character conversation that doesn't depend on where it happens: turn-taking, the
system prompt, the command loop and the commands' handlers. The Discord bot and the local chat window
both build on this; each supplies its own way to post, show typing and remove a character."""
import asyncio
import os
import random
import re
import time

from .agent import describe_action, form_names, run_turn
from .config import ROOT
from .draw import DrawError
from .gifs import GifError, search_gif
from .log import log
from .media import IMG_EXT_RE, encode_image
from .notes import Notebook, current_time
from .tools import THINK_RE, tool_list, tool_text_instructions, tools_tokens
from .web import fetch_link, links_in, preview, read_webpage, remove_link, web_search

FILES_DIR = os.path.join(ROOT, "files")
VOICE_EXCLUDED = ("add_reaction", "view_profile_picture", "draw_image", "post_gif", "create_file")
CODE_FENCE_RE = re.compile(r"^\s*```[^\n]*\n(.*?)\n?```\s*$", re.S)
ANY_FENCE_RE = re.compile(r"```[^\n]*\n(.*?)\n?```", re.S)
PROSE_FILES = ("md", "markdown", "txt", "html", "htm", "tex")   # ``` can be real content in these

# Sent with every prompt, so every sentence here costs context: keep it short.
SYSTEM_RULES = """You are {name}, chatting in a Discord server with real people and maybe other AI characters.

{prompt}

Chat format: messages look like "[#N] Name: text"; other AI characters are marked "(character)", messages from other channels "in #channel". You see display names{pfp} and posted images. Reactions show up as events; never reply to one.
{commands}{notebook}Write only {name}'s next message: no "[#N]" or name prefix, never speak for others. Keep it natural, usually 1-3 short paragraphs.

You are in #{channel} on "{guild}". Characters here: {present}.
{tool_text}"""


COMMAND_LINES = """Alongside your reply you can use commands; use them freely when they fit: {hints}. Only commands make things happen: writing "*reacts*" or "I'll note that" without the command does nothing, so never claim it.
"""
NOTEBOOK_LINE = """Your notebook lasts forever, even after chat history is forgotten.{write} {notes}
"""
# (command, when to use it)
COMMAND_HINTS = [
    ("add_reaction", "react to messages that deserve it"),
    ("get_current_time", "check the time instead of guessing it"),
    ("web_search", "search instead of guessing facts"),
    ("draw_image", "draw when asked"),
    ("post_gif", "post a GIF now and then"),
    ("create_file", "make a file when someone wants one"),
    ("transform", "transform into {forms} whenever you change form, also when someone changes you"),
    ("leave_channel", "leave_channel when you're told to go and agree"),
]
FILE_PROMPT = ("[system] Now write the file {name} ({about}). Your answer is saved as the file exactly as you "
               "write it, so write only its complete contents, between <file> and </file>: no commands, no "
               "explanation, no ``` fence.")
FILE_START, FILE_END = '<file name="{name}">\n', "</file>"


def file_contents(text, ext):
    """The file out of the model's answer: without the <file> markers, and without ``` fences or anything
    the model wrote around them (in code files only; Markdown may contain real fences)."""
    text = THINK_RE.sub("", text or "").split("</think>")[-1]
    text = re.sub(r"^\s*<file[^>]*>\n?", "", text).split(FILE_END)[0]
    m = CODE_FENCE_RE.match(text)
    if m:
        text = m.group(1)
    elif ext not in PROSE_FILES:
        blocks = ANY_FENCE_RE.findall(text)
        if blocks:
            text = max(blocks, key=len)
    return text.strip("\n") + "\n"


def safe_filename(name):
    """'My Story.MD' -> ('My_Story.md', 'md'); keeps it a plain name inside the folder."""
    base, ext = os.path.splitext(os.path.basename(str(name or "").strip()))
    base = re.sub(r"[^\w\-. ]+", "", base).strip(" .").replace(" ", "_")[:60] or "file"
    ext = re.sub(r"[^\w]+", "", ext).lower()
    return (f"{base}.{ext}" if ext else base), ext


def _with_param(tool, prop, schema):
    """A copy of `tool` with one argument's schema replaced (the shared definition stays as it is)."""
    params = dict(tool["function"]["parameters"])
    params["properties"] = {**params["properties"], prop: schema}
    return {**tool, "function": {**tool["function"], "parameters": params}}


def allowed_extensions(cfg):
    raw = cfg["files"].get("extensions") or ""
    return [e.strip().lstrip(".").lower() for e in re.split(r"[,\s]+", raw) if e.strip()]


class ChatLogic:
    """Mixed into a class that has `core`, `cfg`, `http_session`, `recent`, `phase`, `tasks`, `followup`,
    `present(channel_id)`, and implements `_send`, `post_file`, `_typing`, `remove_character`, `announce`."""
    log_tag = "chat"
    excluded_tools = ()          # commands that don't work in this place

    @property
    def behavior(self):
        return self.cfg["behavior"]

    @property
    def gen_lock(self):
        return self.core.gen_lock    # one character generates at a time, here and everywhere else

    def _witness(self, channel_id, entry):
        for info in self.present(channel_id).values():
            self.core.memory.add(info["name"], entry)
        rec = self.recent.setdefault(channel_id, [])
        rec.append(entry)
        del rec[:-30]

    # ---- links --------------------------------------------------------------
    async def _link_previews(self, content, limit=2):
        """A short look at the web pages linked in a message (images are handled as images)."""
        urls = [u for u in links_in(content) if not IMG_EXT_RE.search(u)][:limit]
        if not urls:
            return ""
        links = await asyncio.gather(*(fetch_link(self.http_session, u) for u in urls))
        lines = []
        for lk in links:
            log(f"[web] link {lk.url[:80]}: {lk.kind}{' (' + lk.reason + ')' if lk.reason else ''}")
            if lk.kind == "page":
                lines.append(await self._describe_page(lk))
            elif lk.kind != "image":
                lines.append(preview(lk))
        return "\n".join(x for x in lines if x)

    def _link_checker(self, char, tools):
        """(review, finalize) for a character's message: every link is opened before it's posted.
        Dead ones go back to the character once, image links are shown to it to confirm, and any
        dead link still left afterwards is taken out."""
        cache = {}
        can_search = any(t["function"]["name"] == "web_search" for t in tools)

        async def check(u):
            if u not in cache:
                cache[u] = await fetch_link(self.http_session, u, int(self.behavior["max_image_side"]))
            return cache[u]

        async def review(text):
            urls = links_in(text)[:6]
            if not urls:
                return None, []
            links = await asyncio.gather(*(check(u) for u in urls))
            dead = [lk for lk in links if lk.ok is False]
            pics = [lk for lk in links if lk.kind == "image"]
            notes, parts = [], []
            if dead:
                log(f"[web] {char.name} tried to post dead link(s): {', '.join(lk.url[:80] for lk in dead)}")
                notes.append("These links don't work, so they can't be posted: "
                             + "; ".join(f"{lk.url} ({lk.reason})" for lk in dead) + ". Leave them out"
                             + (", or use web_search to find a real one." if can_search else "."))
            if pics:
                log(f"[web] showing {char.name} the image link(s) it wants to post")
                notes.append("Below is what your image link(s) actually show. If a picture is what you meant, "
                             "keep its link; if not, change or remove it.")
                for lk in pics:
                    parts += [{"type": "text", "text": lk.url}, {"type": "image_url", "image_url": {"url": lk.image}}]
            return " ".join(notes) or None, parts

        async def finalize(text):
            for u in links_in(text):
                if (await check(u)).ok is False:
                    log(f"[web] removed a dead link from {char.name}'s message: {u[:100]}")
                    text = remove_link(text, u)
            return text

        return review, finalize

    async def _describe_page(self, link):
        """Have the loaded model sum up a linked page; falls back to the first lines of its text."""
        llm, lc = self.core.llm, self.cfg["llm"]
        if not llm.ready or not link.text.strip():
            return preview(link)
        room = max(2000, min(16000, (int(lc["ctx_size"]) - int(lc["max_tokens"]) - 600) * 3))
        msgs = [{"role": "system", "content":
                 "You describe web pages for people in a chat who haven't opened them. In 2-4 plain sentences, "
                 "say what the page is and what it actually contains (key facts, names, numbers, the gist of "
                 "an article). No preamble, no opinions."},
                {"role": "user", "content": f"URL: {link.url}\nTitle: {link.title}\n\n{link.text[:room]}"}]
        try:
            reply = await llm.chat(msgs)
        except Exception as e:
            log(f"[web] couldn't describe {link.url[:80]}: {e}", level="warn")
            return preview(link)
        text = THINK_RE.sub("", reply.get("content") or "")
        text = re.sub(r"\s+", " ", text.split("</think>")[-1]).strip()
        if not text:
            return preview(link)
        log(f"[web] described {link.url[:80]}: {text[:100]}")
        return f"[link {link.url}: {link.title or 'web page'} — {text[:700]}]"

    # ---- turn taking --------------------------------------------------------
    def schedule(self, channel, force=None):
        cid = channel.id
        if self.phase.get(cid) == "generating":
            self.followup[cid] = force or self.followup.get(cid) or True
            return
        t = self.tasks.get(cid)
        if t and not t.done():
            # inherit a forced speaker only from a turn that is still waiting to speak
            force = force or getattr(t, "forced", None)
            t.cancel()
        task = asyncio.create_task(self._run_turn(channel, force))
        task.forced = force
        self.tasks[cid] = task

    async def _run_turn(self, channel, force):
        cid = channel.id
        try:
            self.phase[cid] = "waiting"
            await asyncio.sleep(float(self.behavior["response_delay"]))
            asyncio.current_task().forced = None  # this turn is being used up now
            speaker = force or self.pick_responder(cid)
            if not speaker or self._spoke_last(cid, speaker):
                return
            self.phase[cid] = "generating"
            async with self.gen_lock:
                await self.respond(channel, speaker)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log(f"[{self.log_tag}] error while responding: {e!r}")
        finally:
            if self.tasks.get(cid) is asyncio.current_task():
                self.phase.pop(cid, None)
                self.tasks.pop(cid, None)
                nxt = self.followup.pop(cid, None)
                if nxt:
                    self.schedule(channel, force=nxt if isinstance(nxt, str) else None)

    def _spoke_last(self, cid, speaker):
        """A character never answers itself: skip if the newest entry is its own message."""
        rec = self.recent.get(cid) or []
        last = rec[-1] if rec else None
        return bool(last and last.kind == "msg" and last.char_name
                    and last.char_name.lower() == speaker.lower())

    def pick_responder(self, cid):
        present = [i["name"] for i in self.present(cid).values()]
        rec = [e for e in self.recent.get(cid, []) if e.kind == "msg"]
        if not present or not rec:
            return None
        last = rec[-1]
        candidates = [n for n in present if not (last.char_name and n.lower() == last.char_name.lower())]
        if not candidates:
            return None
        # replying directly to a character's message
        if last.reply_ref:
            parent = self.core.memory.by_ref.get(last.reply_ref)
            if parent and parent.char_name and parent.char_name in candidates:
                if not last.char_name or self._chain_ok(rec):
                    return parent.char_name
        mentioned = self._mentioned(last.text, candidates)
        if last.char_name:  # character-to-character
            if not self._chain_ok(rec):
                return None
            if mentioned:
                return mentioned
            if random.random() < float(self.behavior["bot_reply_chance"]):
                return random.choice(candidates)
            return None
        if mentioned:
            return mentioned
        if len(candidates) == 1:
            return candidates[0]
        for e in reversed(rec):  # whoever has been talking most recently keeps the thread
            if e.char_name and e.char_name in candidates:
                return e.char_name
        return random.choice(candidates)

    def _chain_ok(self, rec):
        chain = 0
        for e in reversed(rec):
            if not e.char_name:
                break
            chain += 1
        return chain < int(self.behavior["max_bot_chain"])

    @staticmethod
    def _mentioned(text, names):
        best, pos = None, None
        low = (text or "").lower()
        for n in names:
            m = re.search(r"\b" + re.escape(n.lower()) + r"\b", low)
            if m and (pos is None or m.start() < pos):
                best, pos = n, m.start()
        return best

    # ---- generation ---------------------------------------------------------
    def tool_on(self, name):
        """The command's switch on the Behavior page, plus what it needs to work."""
        if not self.cfg["tools"].get(name, True):
            return False
        if name == "post_gif":
            return bool(self.cfg["gifs"].get("klipy_key"))
        if name == "draw_image":
            return not self.core.draw.paused_by
        if name == "create_file":
            return bool(allowed_extensions(self.cfg))
        return True

    def tools_for(self, voice=False, char=None):
        """`char`'s commands; transform is only there for a character with forms (and lists them)."""
        # in voice chat nothing can be shown, so reactions, pictures, GIFs and files are left out
        tools = tool_list(self.tool_on, exclude=tuple(self.excluded_tools) + (VOICE_EXCLUDED if voice else ()))
        forms = [c for c in self.core.characters.forms_of(char)] if char else []
        if voice:   # in voice chat a form needs a voice of its own
            forms = [c for c in forms if getattr(self, "can_speak", lambda c: True)(c)]
        forms = [c.name for c in forms]
        out = []
        for t in tools:
            name = t["function"]["name"]
            if name == "create_file":   # the switchable list of file types goes in the schema
                t = _with_param(t, "filename", {"type": "string", "description": "with one of these endings: "
                                                + ", ".join(allowed_extensions(self.cfg))})
            elif name == "transform":
                if not forms:
                    continue
                t = _with_param(t, "into", {"type": "string", "enum": forms})
            out.append(t)
        return out

    # ---- context meter -------------------------------------------------------
    def note_usage(self, char_name):
        """Remember the real size (from llama-server) of the character's biggest call this turn."""
        if self.core.llm.peak_tokens:
            self.core.context_used[char_name.lower()] = (self.core.llm.peak_tokens, time.time())

    def context_estimate(self, channel, char):
        """Tokens the character's next prompt would take right now (estimate, nothing is trimmed)."""
        b, lc = self.behavior, self.cfg["llm"]
        tools = self.tools_for(char=char)
        system = self._system_prompt(channel, char, tools=tools)
        budget = int(lc["ctx_size"]) - int(lc["max_tokens"]) - 256 - tools_tokens(tools)
        msgs = self.core.memory.build_messages(char.name, system, channel.id, budget, int(b["image_token_cost"]),
                                               int(b["max_images_in_context"]), dry_run=True)
        return self.core.memory.count_tokens(msgs, int(b["image_token_cost"])) + tools_tokens(tools)

    def _system_prompt(self, channel, char, present=None, tools=None):
        tools = self.tools_for(char=char) if tools is None else tools
        names = {t["function"]["name"] for t in tools}
        native = self.cfg["llm"].get("use_native_tools", True)
        forms = " or ".join(form_names(tools))   # named, so it knows which form changes it back
        hints = [h.format(forms=forms) for cmd, h in COMMAND_HINTS if cmd in names]
        commands = COMMAND_LINES.format(hints=", ".join(hints) or "whenever useful") if names else ""
        notebook = ""
        if names & {"read_note", "write_note"}:
            write = " Write down what's worth remembering as soon as you learn it." if "write_note" in names else ""
            notebook = NOTEBOOK_LINE.format(write=write, notes=self.notebook(char).list())
        return SYSTEM_RULES.format(
            name=char.name, prompt=char.prompt.strip(), channel=channel.name, guild=channel.guild.name,
            present=", ".join(present or [i["name"] for i in self.present(channel.id).values()]),
            commands=commands, notebook=notebook,
            pfp=" (and profile pictures, via the command)" if "view_profile_picture" in names else "",
            tool_text="" if native or not tools else "\n" + tool_text_instructions(tools))

    async def respond(self, channel, speaker):
        info = self.present(channel.id).get(speaker.lower())
        char = self.core.characters.get(speaker)
        if not info or not char:
            return
        llm = self.core.llm
        if not llm.ready:
            why = f"suspended while {llm.suspended_by} is running" if llm.suspended_by else "not loaded"
            log(f"[{self.log_tag}] {speaker} can't reply: model {why}")
            return
        b, lc = self.behavior, self.cfg["llm"]
        budget = int(lc["ctx_size"]) - int(lc["max_tokens"]) - 256
        tools = self.tools_for(char=char)
        system = self._system_prompt(channel, char, tools=tools)
        budget -= tools_tokens(tools)
        msgs = self.core.memory.build_messages(char.name, system, channel.id, budget,
                                               int(b["image_token_cost"]), int(b["max_images_in_context"]))
        log(f"[{self.log_tag}] {char.name} is thinking in #{channel.name} ({len(msgs) - 1} context entries)")

        typing = asyncio.create_task(self._typing(channel, char)) if b.get("typing_indicator") else None
        review, finalize = self._link_checker(char, tools) if b.get("check_links", True) else (None, None)
        llm.peak_tokens = 0
        try:
            res = await run_turn(llm, msgs, tools, char.name,
                                 on_react=lambda a: self._react(channel, char.name, a),
                                 on_pfp=lambda name: self._profile_picture(channel, name),
                                 handlers=self._local_tools(char, channel, info, msgs), review=review,
                                 decide=bool(b.get("decide_commands", True) and lc.get("use_native_tools", True)))
            if finalize and res.text:
                res.text = await finalize(res.text)
        finally:
            if typing:
                typing.cancel()
            self.note_usage(char.name)

        self._remember_actions(char, channel.id, channel.name, res)
        if res.silent:
            log(f"[{self.log_tag}] {char.name} chose to stay silent")
        elif res.text:
            await self._send(channel, info, char, res.text)
        if res.leave:
            await self.remove_character(channel, char.name, f"{char.name} decided to leave #{channel.name}.")
            await self.announce(channel, f"👋 **{char.name}** left the channel.")
        elif res.transform:
            await self.transform(channel, char, res.transform)
        if self.core.memory.dirty:
            self.core.memory.save()

    def _remember_actions(self, char, channel_id, channel_name, res):
        """Commands leave no trace in the chat history, so the model would only ever see itself
        talking. Recording them (privately) shows it that it really uses its commands."""
        mem = self.core.memory
        for name, args in res.actions:
            what = describe_action(name, args)
            if what:
                mem.add(char.name, mem.new_entry(kind="event", channel_id=channel_id, channel_name=channel_name,
                                                 text=f"{char.name} {what}."))

    def notebook(self, char):
        """Linked forms are one being: they share their notes, like their memory."""
        return Notebook(char.key, [c.key for c in self.core.characters.group_of(char)])

    def form_here(self, channel_id, char):
        """The name of another form of `char` that is already in the channel, or None."""
        here = self.present(channel_id)
        return next((c.name for c in self.core.characters.group_of(char)
                     if c.name.lower() != char.name.lower() and c.name.lower() in here), None)

    async def transform(self, channel, char, into, by=None, announce=True):
        """`char` becomes its form `into` right where it is; the new form speaks next.
        `by` is the person who used /transform. Returns the notice, or None if it didn't happen."""
        new = self.core.characters.get(into)
        if not new or new.name.lower() in self.present(channel.id):
            return None
        if not await self.swap_form(channel, char, new):
            return None
        log(f"[{self.log_tag}] {by + ' turned ' if by else ''}{char.name}{'' if by else ' transformed'} "
            f"into {new.name} in #{channel.name}")
        ev = self.core.memory.new_entry(kind="event", channel_id=channel.id, channel_name=channel.name,
                                        text=f"{by} transformed {char.name} into {new.name}." if by else
                                        f"{char.name} transformed into {new.name}.")
        self._witness(channel.id, ev)
        notice = f"✨ **{char.name}** transformed into **{new.name}**."
        if announce:
            await self.announce(channel, notice)
        self.schedule(channel, force=new.name)
        return notice

    def transform_target(self, channel_id, who="", into=""):
        """Who /transform changes and into what: (character, form name), or (None, why not).
        `who` is a character here or left out (the only one here with forms); `into` is one of its forms or
        left out (its only form). `who` alone may also name the form, e.g. "/transform Mayo" changes
        Mayo (Scorpion) back."""
        store, here = self.core.characters, self.present(channel_id)
        chars = [c for c in (store.get(i["name"]) for i in here.values()) if c and store.forms_of(c)]
        if not here:
            return None, "Nobody is here."
        if who and who.lower() not in here:
            c = store.get(who)
            other = self.form_here(channel_id, c) if c else None
            if not other or into:
                return None, f"**{who}** isn't here."
            who, into = other, c.name
        if who:
            chars = [c for c in chars if c.name.lower() == who.lower()]
            if not chars:
                return None, f"**{here[who.lower()]['name']}** has no other forms (Characters page → Forms)."
        elif len(chars) != 1:
            return None, ("Nobody here has other forms (Characters page → Forms)." if not chars else
                          "Who? " + ", ".join(f"**{c.name}**" for c in chars) + " can transform.")
        char = chars[0]
        forms = [f.name for f in store.forms_of(char)]
        if into:
            form = next((f for f in forms if f.lower() == into.lower()), None)
            if not form:
                return None, f"**{char.name}** can only transform into " + ", ".join(f"**{f}**" for f in forms) + "."
            return char, form
        if len(forms) > 1:
            return None, f"Into what? **{char.name}** can become " + ", ".join(f"**{f}**" for f in forms) + "."
        return char, forms[0]

    @staticmethod
    def transform_args(arg):
        """'/transform Mayo into Mayo (Scorpion)' -> ('Mayo', 'Mayo (Scorpion)'); 'Mayo' -> ('Mayo', '')."""
        parts = re.split(r"\s+(?:into|->|>)\s+", arg.strip(), maxsplit=1, flags=re.I)
        return parts[0].strip(), (parts[1].strip() if len(parts) > 1 else "")

    async def post_gif(self, channel, info, char, url, title):
        """Discord shows a .gif link as the GIF itself."""
        await self._send(channel, info, char, url)

    # ---- command handlers ---------------------------------------------------
    def _local_tools(self, char, channel=None, info=None, msgs=None):
        """`channel`/`info` (where and as whom to post) are needed for commands that post something;
        `msgs` (the conversation so far) for create_file, which writes the file in a call of its own."""
        nb = self.notebook(char)
        drawn, made = [], []

        async def draw(args):
            prompt = str(args.get("prompt") or "").strip()
            if not prompt:
                return "Say what to draw."
            if drawn:
                return "You already drew a picture this turn."
            drawn.append(prompt)
            log(f"[draw] {char.name} is drawing: {prompt[:100]}")
            try:
                png = await self.core.draw.draw(self.http_session, prompt)
            except DrawError as e:
                log(f"[draw] {char.name} couldn't draw: {e}", level="warn")
                return f"Drawing failed: {e}. Tell them you couldn't draw it right now."
            await self.post_file(channel, info, char, "drawing.png", png)
            seen = encode_image(png, int(self.behavior["max_image_side"]))
            note = "[system] Your drawing was posted in the chat. This is what it looks like:"
            return [{"type": "text", "text": note}, {"type": "image_url", "image_url": {"url": seen}}] \
                if seen else "Your drawing was posted in the chat."

        async def create_file(args):
            name, ext = safe_filename(args.get("filename") or args.get("name"))
            allowed = allowed_extensions(self.cfg)
            if ext not in allowed:
                return f"Can't make .{ext or '?'} files. Use one of: {', '.join(allowed)}."
            if len(made) >= 2:
                return "That's enough files for one message."
            made.append(name)
            about = str(args.get("about") or args.get("description") or "what was asked for").strip()[:300]
            log(f"[files] {char.name} is writing {name}: {about[:100]}")
            # the answer is started for the model (a prefill), so it can only go on with the file itself
            ask = [{"role": "user", "content": FILE_PROMPT.format(name=name, about=about)},
                   {"role": "assistant", "content": FILE_START.format(name=name)}]
            reply = await self.core.llm.chat((msgs or [])[:] + ask, stop=[FILE_END],
                                             max_tokens=int(self.cfg["files"].get("max_tokens") or 4000))
            text = file_contents(reply.get("content"), ext)
            if not text.strip():
                return "The file came out empty; nothing was posted."
            folder = os.path.join(FILES_DIR, char.key)
            os.makedirs(folder, exist_ok=True)
            with open(os.path.join(folder, name), "w", encoding="utf-8", newline="") as f:
                f.write(text)
            await self.post_file(channel, info, char, name, text.encode("utf-8"))
            lines = text.count("\n")
            log(f"[files] {char.name} posted {name} ({lines} lines)")
            return f"Posted {name} ({lines} lines) in the chat; everyone can open it. Don't paste it again."

        async def gif(args):
            q = str(args.get("query") or "").strip()
            try:
                url, title = await search_gif(self.http_session, self.cfg["gifs"].get("klipy_key"), q)
            except GifError as e:
                log(f"[gifs] {char.name} couldn't post a GIF: {e}", level="warn")
                return f"No GIF posted: {e}."
            log(f"[gifs] {char.name} posted a GIF for {q!r}")
            await self.post_gif(channel, info, char, url, title)
            return f'Posted a GIF ("{title}").'

        def logged(fn, what):
            def run(args):
                out = fn(args)
                log(f"[notes] {char.name} {what}: {out.splitlines()[0][:80]}")
                return out
            return run

        async def search(args):
            log(f"[web] {char.name} searched for {args.get('query')!r}")
            return await web_search(args.get("query"))

        async def read_page(args):
            log(f"[web] {char.name} read {str(args.get('url'))[:100]}")
            return await read_webpage(self.http_session, args.get("url"))

        return {"get_current_time": current_time, "read_note": nb.read, "write_note": logged(nb.write, "wrote"),
                "web_search": search, "read_webpage": read_page,
                **({"draw_image": draw, "post_gif": gif, "create_file": create_file} if channel and info else {})}
