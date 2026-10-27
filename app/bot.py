"""Discord side: slash commands, webhook characters, turn-taking."""
import asyncio
import io
import random
import re
import time
from typing import Optional

import aiohttp
import discord
from discord import app_commands

from .config import load_state, save_state
from .log import log
from .draw import DrawError
from .gifs import GifError, search_gif
from .media import IMG_EXT_RE, encode_image, fetch_image, images_from_message
from .agent import describe_action, run_turn
from .notes import Notebook, current_time
from .tools import THINK_RE, tool_list, tool_text_instructions, tools_tokens
from .web import fetch_link, links_in, preview, read_webpage, remove_link, web_search

SYSTEM_RULES = """You are roleplaying as {name} in a Discord server, chatting alongside real people and possibly other AI characters.

{prompt}

How the chat works:
- Each message is shown as "[#N] Name: text". N is the message number. Messages from other AI characters are marked "(character)".
- Messages from a different channel than the current one are marked "in #channel". You remember them, but you are now speaking in #{channel}.
- You only see people's display names{pfp}. You can see images people post.
- Emoji reactions are shown as events. You never reply to a reaction itself, but you may react to messages yourself.
{commands}{notebook}- Write only {name}'s next message: no "[#N]" prefix, no name prefix, never speak for other people or characters.
- Keep it natural and conversational for Discord; usually 1-3 short paragraphs at most.

Current channel: #{channel} in the server "{guild}". Characters present here: {present}.
{tool_text}"""


COMMAND_LINES = """- Normally you reply with a message, and you can use commands ({kinds}) alongside it. Use them freely whenever they fit, as a real person would: {hints}. A command never replaces your reply.
- Only a command makes something happen. Writing "*reacts with 😂*", "I'll write that down" or "let me look it up" without using the command does nothing, so never claim you did something you didn't use a command for.
"""
NOTEBOOK_LINE = """- You have a personal notebook of Markdown notes that lasts forever, even after old chat history is forgotten. {write}{notes}
"""
# (command, what it's called in the list, when to use it)
COMMAND_HINTS = [
    ("add_reaction", "reacting", "react to messages that deserve it"),
    ("write_note", "notes", None), ("read_note", "notes", None), ("list_notes", "notes", None),
    ("get_current_time", "checking the time", "check the time instead of guessing it"),
    ("web_search", "searching the web",
     "search the web for news, facts or anything you're unsure about instead of making it up"),
    ("draw_image", "drawing", "draw a picture when someone asks for one or when showing beats telling"),
    ("post_gif", "posting GIFs", "post a GIF now and then when it fits the mood"),
    ("leave_channel", "leaving", "use leave_channel when you decide to leave or are told to go and agree"),
    ("stay_silent", "staying silent", None),
]

TYPED_CMD_RE = re.compile(r"^\s*/(summon|dismiss|gohome|forget|characters|who|context|join|leave)\b[ \t]*([^\n]*)",
                          re.I)


def meter_bar(used, size, width=16):
    filled = min(width, round(width * used / max(size, 1)))
    return "█" * filled + "░" * (width - filled)


class TypedInteraction:
    """Lets the slash-command handlers answer a command that was typed as a normal message.
    Replies go to the channel as a reply to that message (there is no "only you can see this")."""

    def __init__(self, message):
        self.message = message
        self.channel = message.channel
        self.channel_id = message.channel.id
        self.guild = message.guild
        self.user = message.author
        self.response = self
        self.followup = self

    async def send_message(self, text, ephemeral=False, view=None):
        await self.message.reply(text, mention_author=False, **({"view": view} if view else {}))

    async def send(self, text):
        await self.message.reply(text, mention_author=False)

    async def defer(self, thinking=False):
        await self.channel.typing()


class CharacterPicker(discord.ui.View):
    """Dropdowns of characters (25 per menu, Discord's limit) showing each one's description."""

    def __init__(self, bot, cmd, chars, user_id):
        super().__init__(timeout=180)
        self.bot, self.cmd, self.user_id = bot, cmd, user_id
        for i in range(0, min(len(chars), 125), 25):
            part = chars[i:i + 25]
            menu = discord.ui.Select(
                placeholder="Choose a character" if len(chars) <= 25 else f"{part[0].name} – {part[-1].name}",
                options=[discord.SelectOption(label=c.name[:100], value=c.name,
                                              description=(c.description or None) and c.description[:100])
                         for c in part])
            menu.callback = self._make_callback(menu)
            self.add_item(menu)

    def _make_callback(self, menu):
        async def chosen(interaction):
            if interaction.user.id != self.user_id:
                return await interaction.response.send_message("This list is for someone else.", ephemeral=True)
            self.stop()
            name = menu.values[0]
            log(f"[discord] {interaction.user.display_name} picked {name} for /{self.cmd}")
            if self.cmd == "join":
                await self.bot.cmd_join(interaction, name)
            else:
                await self.bot.cmd_summon(interaction, name)
        return chosen


class CharacterBot(discord.Client):
    def __init__(self, core):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.dm_messages = False      # DMs are disabled entirely
        intents.dm_reactions = False
        intents.dm_typing = False
        super().__init__(intents=intents)
        self.core = core
        self.cfg = core.cfg
        self.tree = app_commands.CommandTree(self)
        self.state = load_state()
        self.state.setdefault("summoned", {})   # channel id -> {char lower: {name, webhook_id, webhook_url}}
        self.http_session = None
        self.gen_lock = asyncio.Lock()          # only one character generates at a time
        self.tasks = {}                         # channel id -> asyncio.Task
        self.phase = {}                         # channel id -> "waiting" | "generating"
        self.followup = {}                      # channel id -> trigger to run after generation
        self.recent = {}                        # channel id -> list of recent Entry (turn-taking)
        self.names = {}                         # user id -> display name
        self.voice = {}                         # guild id -> voice.VoiceSession
        self.context_used = {}                  # char lower -> (tokens of its last reply's biggest call, time)
        self.on_change = lambda: None
        self._register_commands()

    # ---- helpers ------------------------------------------------------------
    @property
    def behavior(self):
        return self.cfg["behavior"]

    def allowed(self, channel_id):
        return str(channel_id) in {str(c) for c in self.cfg["discord"]["allowed_channels"]}

    def present(self, channel_id):
        return self.state["summoned"].get(str(channel_id), {})

    @staticmethod
    def ignored(message):
        """Messages starting with // are out-of-character and invisible to the characters."""
        return (message.content or "").lstrip().startswith("//")

    def webhook_owner(self, webhook_id):
        for cid, chars in self.state["summoned"].items():
            for info in chars.values():
                if info.get("webhook_id") == webhook_id:
                    return info["name"]
        return None

    def _save(self):
        save_state(self.state)
        try:
            self.on_change()
        except Exception:
            pass

    def list_channels(self):
        out = []
        for g in self.guilds:
            chans = [(str(c.id), c.name) for c in g.text_channels
                     if c.permissions_for(g.me).view_channel]
            out.append((g.name, chans))
        return out

    def summoned_overview(self):
        res = []
        for cid, chars in self.state["summoned"].items():
            ch = self.get_channel(int(cid))
            res.append((f"#{ch.name}" if ch else cid, [c["name"] for c in chars.values()]))
        return res

    # ---- lifecycle ----------------------------------------------------------
    async def setup_hook(self):
        self.http_session = aiohttp.ClientSession()
        try:
            await self.tree.sync()
        except discord.HTTPException as e:
            log(f"[discord] command sync failed: {e}")

    async def on_ready(self):
        log(f"[discord] connected as {self.user} in {len(self.guilds)} server(s)")
        # Commands are global only. Older versions also registered a per-server copy, which made
        # every command show up twice; remove those leftovers.
        for g in self.guilds:
            try:
                if await self.tree.fetch_commands(guild=g):
                    self.tree.clear_commands(guild=g)
                    await self.tree.sync(guild=g)
                    log(f"[discord] removed duplicate commands in {g.name}")
            except discord.HTTPException:
                pass
        self.on_change()

    async def close(self):
        for guild_id in list(self.voice):
            await self.voice.pop(guild_id).close()
        if self.http_session:
            await self.http_session.close()
        await super().close()

    # ---- slash commands -----------------------------------------------------
    def _register_commands(self):
        store = self.core.characters

        def choices(chars, current):
            # autocomplete entries are only a name; descriptions are shown by the picker (no argument)
            return [app_commands.Choice(name=c.name, value=c.name)
                    for c in chars if current.lower() in c.name.lower()][:25]

        async def char_autocomplete(interaction, current):
            return choices(store.characters.values(), current)

        async def voice_autocomplete(interaction, current):
            return choices([c for c in store.characters.values() if c.voice], current)

        async def present_autocomplete(interaction, current):
            names = [c["name"] for c in self.present(interaction.channel_id).values()]
            if len(names) > 1:
                names.append("all")
            return [app_commands.Choice(name=n, value=n) for n in names if current.lower() in n.lower()][:25]

        @self.tree.command(name="summon", description="Summon a character into this channel")
        @app_commands.guild_only()
        @app_commands.describe(character="Character to summon (leave empty to pick from a list)")
        @app_commands.autocomplete(character=char_autocomplete)
        async def summon(interaction: discord.Interaction, character: Optional[str] = None):
            if not character:
                return await self.pick_character(interaction, "summon")
            await self.cmd_summon(interaction, character)

        @self.tree.command(name="dismiss", description="Dismiss a character from this channel")
        @app_commands.guild_only()
        @app_commands.describe(character="Character to dismiss (or 'all')")
        @app_commands.autocomplete(character=present_autocomplete)
        async def dismiss(interaction: discord.Interaction, character: str):
            await self.cmd_dismiss(interaction, character)

        @self.tree.command(name="gohome", description="Dismiss every character in this channel")
        @app_commands.guild_only()
        async def gohome(interaction: discord.Interaction):
            await self.cmd_gohome(interaction)

        @self.tree.command(name="forget", description="Wipe the memory of every character in this channel")
        @app_commands.guild_only()
        async def forget(interaction: discord.Interaction):
            await self.cmd_forget(interaction)

        @self.tree.command(name="characters", description="List available and present characters")
        @app_commands.guild_only()
        async def characters(interaction: discord.Interaction):
            await self.cmd_characters(interaction)

        @self.tree.command(name="who", description="Show which characters are in this channel")
        @app_commands.guild_only()
        async def who(interaction: discord.Interaction):
            await self.cmd_who(interaction)

        @self.tree.command(name="context", description="Show how full each character's context is")
        @app_commands.guild_only()
        async def context(interaction: discord.Interaction):
            await self.cmd_context(interaction)

        @self.tree.command(name="join", description="Bring a character into your voice channel")
        @app_commands.guild_only()
        @app_commands.describe(character="Character to talk to (needs a voice; leave empty to pick from a list)")
        @app_commands.autocomplete(character=voice_autocomplete)
        async def join(interaction: discord.Interaction, character: Optional[str] = None):
            if not character:
                return await self.pick_character(interaction, "join")
            await self.cmd_join(interaction, character)

        @self.tree.command(name="leave", description="Make the character in voice chat leave")
        @app_commands.guild_only()
        async def leave(interaction: discord.Interaction):
            await self.cmd_leave(interaction)

    async def cmd_characters(self, interaction):
        here = [c["name"] for c in self.present(interaction.channel_id).values()]
        lines = [f"**Here:** {', '.join(here) or 'nobody'}"]
        session = self.voice.get(getattr(interaction.guild, "id", None))
        if session:
            lines.append(f"**In voice:** {session.char.name} in {session.channel.name}")
        lines.append("**Available** (🔊 = can join voice chat):")
        for c in self.core.characters.characters.values():
            lines.append(f"- **{c.name}**" + (f" — {c.description}" if c.description else "")
                         + (" 🔊" if c.voice else ""))
        await interaction.response.send_message("\n".join(lines)[:1990], ephemeral=True)

    async def cmd_who(self, interaction):
        lines = []
        for info in self.present(interaction.channel_id).values():
            c = self.core.characters.get(info["name"])
            desc = c.description if c and c.description else ""
            lines.append(f"- **{info['name']}**" + (f" — {desc}" if desc else ""))
        lines = [f"**In #{interaction.channel.name}:**"] + lines if lines else \
            [f"No characters are in #{interaction.channel.name}. Use `/summon` to bring one."]
        session = self.voice.get(getattr(interaction.guild, "id", None))
        if session:
            lines.append(f"🔊 **{session.char.name}** is in voice chat ({session.channel.mention}).")
        await interaction.response.send_message("\n".join(lines)[:1990])

    async def pick_character(self, interaction, cmd):
        """Dropdown of characters with their descriptions; choosing one runs /summon or /join."""
        chars = list(self.core.characters.characters.values())
        if cmd == "join":
            chars = [c for c in chars if c.voice]
        else:
            here = self.present(interaction.channel_id)
            chars = [c for c in chars if c.name.lower() not in here]
        if not chars:
            text = "No character has a voice yet." if cmd == "join" else "Every character is already here."
            return await interaction.response.send_message(text, ephemeral=True)
        view = CharacterPicker(self, cmd, chars, interaction.user.id)
        prompt = "Who should join voice chat?" if cmd == "join" else "Who should I summon?"
        await interaction.response.send_message(prompt, view=view, ephemeral=True)

    async def cmd_gohome(self, interaction):
        if not self.present(interaction.channel_id):
            return await interaction.response.send_message("Nobody is here.", ephemeral=True)
        await self.cmd_dismiss(interaction, "all")

    async def cmd_forget(self, interaction):
        names = [c["name"] for c in self.present(interaction.channel_id).values()]
        if not names:
            return await interaction.response.send_message("Nobody is here.", ephemeral=True)
        for n in names:
            self.core.memory.forget(n)
        self.core.memory.save()
        log(f"[discord] {interaction.user.display_name} wiped the memory of {', '.join(names)}")
        await interaction.response.send_message(
            "🧹 " + ", ".join(f"**{n}**" for n in names) + " forgot everything.")

    async def typed_command(self, message):
        """Handle a slash command typed out as plain text ("/summon Mayo"). True if it was one."""
        m = TYPED_CMD_RE.match(message.content or "")
        if not m or message.webhook_id or not isinstance(message.channel, discord.TextChannel):
            return False
        cmd, arg = m.group(1).lower(), (m.group(2) or "").strip()
        it = TypedInteraction(message)
        if cmd in ("summon", "join") and not arg:
            await self.pick_character(it, cmd)
            return True
        if cmd == "dismiss" and not arg:
            await message.reply("Usage: `/dismiss <character>`", mention_author=False)
            return True
        log(f"[discord] {message.author.display_name} typed /{cmd} {arg}".rstrip())
        handlers = {"summon": lambda: self.cmd_summon(it, arg), "dismiss": lambda: self.cmd_dismiss(it, arg),
                    "gohome": lambda: self.cmd_gohome(it), "forget": lambda: self.cmd_forget(it),
                    "characters": lambda: self.cmd_characters(it), "who": lambda: self.cmd_who(it),
                    "context": lambda: self.cmd_context(it),
                    "join": lambda: self.cmd_join(it, arg),
                    "leave": lambda: self.cmd_leave(it)}
        await handlers[cmd]()
        return True

    # ---- voice chat ---------------------------------------------------------
    async def cmd_join(self, interaction, name):
        guild = interaction.guild
        member = guild.get_member(interaction.user.id) if guild else None
        target = member.voice.channel if member and member.voice else None
        reply = interaction.response.send_message
        if not target:
            return await reply("Join a voice channel first, then ask again.", ephemeral=True)
        if not self.allowed(interaction.channel_id) and interaction.channel_id != target.id:
            return await reply("I'm not allowed to speak in this channel.", ephemeral=True)
        char = self.core.characters.get(name)
        if not char:
            return await reply(f"No character named **{name}**.", ephemeral=True)
        if not char.voice:
            return await reply(f"**{char.name}** doesn't have a voice, so they can't join voice chat.",
                               ephemeral=True)
        if not self.core.tts.key:
            return await reply("No ElevenLabs API key is set (Voice page in the app).", ephemeral=True)
        perms = target.permissions_for(guild.me)
        if not (perms.connect and perms.speak):
            return await reply(f"I need the **Connect** and **Speak** permissions in {target.mention}.",
                               ephemeral=True)
        current = self.voice.get(guild.id)
        if current and current.char.name == char.name and current.channel.id == target.id:
            return await reply(f"**{char.name}** is already in {target.mention}.", ephemeral=True)
        await interaction.response.defer(thinking=True)
        from discord.ext import voice_recv
        from .voice import VoiceSession
        try:
            await asyncio.to_thread(self.core.stt.load)   # downloads the Whisper model the first time
        except Exception as e:
            return await interaction.followup.send(f"Speech recognition couldn't start: {e}")
        if current:
            await self.leave_voice(guild, f"{current.char.name} left {current.channel_label} "
                                          f"to make room for {char.name}.",
                                   f"👋 **{current.char.name}** left voice chat to make room for **{char.name}**.")
        try:
            vc = await target.connect(cls=voice_recv.VoiceRecvClient, self_deaf=False)
        except Exception as e:
            log(f"[voice] couldn't connect to {target.name}: {e!r}", level="warn")
            return await interaction.followup.send(f"Couldn't join {target.mention}: {e}")
        session = self.voice[guild.id] = VoiceSession(self, char, vc, interaction.channel)
        mem = self.core.memory
        mem.add(char.name, mem.new_entry(kind="event", channel_id=target.id, channel_name=session.channel_label,
                                         text=f"{interaction.user.display_name} brought {char.name} into the "
                                              f"voice channel {target.name}."))
        log(f"[voice] {char.name} joined {target.name}")
        await interaction.followup.send(f"🔊 **{char.name}** joined {target.mention}.")

    async def cmd_leave(self, interaction):
        session = self.voice.get(getattr(interaction.guild, "id", None))
        if not session:
            return await interaction.response.send_message("Nobody is in voice chat.", ephemeral=True)
        await interaction.response.send_message(f"👋 **{session.char.name}** left voice chat.")
        await self.leave_voice(interaction.guild, f"{interaction.user.display_name} sent {session.char.name} "
                                                  f"out of {session.channel_label}.")

    async def leave_voice(self, guild, reason, announce=None):
        """`announce` is posted in the text channel /join was used in."""
        session = self.voice.pop(guild.id, None)
        if not session:
            return
        mem = self.core.memory
        mem.add(session.char.name, mem.new_entry(kind="event", channel_id=session.channel.id,
                                                 channel_name=session.channel_label, text=reason))
        await session.close()
        log(f"[voice] {session.char.name} left {session.channel.name}")
        if announce:
            await self.announce(session.text_channel, announce)

    async def on_voice_state_update(self, member, before, after):
        session = self.voice.get(member.guild.id)
        if not session:
            return
        if member.id == self.user.id:
            if after.channel is None and not session.closed:  # kicked or disconnected
                await self.leave_voice(member.guild, f"{session.char.name} was disconnected from voice chat.",
                                       f"👋 **{session.char.name}** was disconnected from voice chat.")
            return
        if before.channel and before.channel.id == session.channel.id and \
                not any(not m.bot for m in session.channel.members):
            await self.leave_voice(member.guild, f"{session.char.name} left {session.channel_label} "
                                                 f"because everyone else left.",
                                   f"👋 **{session.char.name}** left voice chat because everyone else left.")

    async def cmd_summon(self, interaction, name):
        ch = interaction.channel
        if not isinstance(ch, discord.TextChannel):
            return await interaction.response.send_message("Characters can only be summoned in text channels.", ephemeral=True)
        if not self.allowed(ch.id):
            return await interaction.response.send_message("I'm not allowed to speak in this channel.", ephemeral=True)
        char = self.core.characters.get(name)
        if not char:
            return await interaction.response.send_message(f"No character named **{name}**.", ephemeral=True)
        if char.name.lower() in self.present(ch.id):
            return await interaction.response.send_message(f"**{char.name}** is already here.", ephemeral=True)
        if not ch.permissions_for(ch.guild.me).manage_webhooks:
            return await interaction.response.send_message("I need the **Manage Webhooks** permission here.", ephemeral=True)
        await interaction.response.defer(thinking=True)
        try:
            wh = await self._create_webhook(ch, char)
        except discord.HTTPException as e:
            return await interaction.followup.send(f"Couldn't create webhook: {e}")
        self.state["summoned"].setdefault(str(ch.id), {})[char.name.lower()] = {
            "name": char.name, "webhook_id": wh.id, "webhook_url": wh.url}
        self._save()
        await interaction.followup.send(f"✨ **{char.name}** has been summoned.")
        await self._backfill(ch, char.name)
        ev = self.core.memory.new_entry(kind="event", channel_id=ch.id, channel_name=ch.name,
                                        text=f"{interaction.user.display_name} summoned {char.name} into #{ch.name}.")
        self._witness(ch.id, ev)
        log(f"[discord] {char.name} summoned into #{ch.name}")
        self.schedule(ch, force=char.name)

    async def _create_webhook(self, channel, char):
        avatar = char.avatar_bytes()
        name = re.sub(r"(?i)discord|clyde", "_", char.name)[:80] or "Character"
        return await channel.create_webhook(name=name, avatar=avatar, reason="Character summoned")

    async def cmd_dismiss(self, interaction, name):
        here = self.present(interaction.channel_id)
        targets = list(here.values()) if name.lower() == "all" else [here[name.lower()]] if name.lower() in here else []
        if not targets:
            return await interaction.response.send_message(f"**{name}** isn't here.", ephemeral=True)
        await interaction.response.send_message(
            "👋 Dismissed " + ", ".join(f"**{t['name']}**" for t in targets) + ".")
        for t in targets:
            await self.remove_character(interaction.channel, t["name"],
                                        f"{interaction.user.display_name} dismissed {t['name']} from #{interaction.channel.name}.")

    async def remove_character(self, channel, name, reason):
        here = self.present(channel.id)
        info = here.pop(name.lower(), None)
        if not info:
            return
        if not here:
            self.state["summoned"].pop(str(channel.id), None)
        self._save()
        ev = self.core.memory.new_entry(kind="event", channel_id=channel.id, channel_name=channel.name, text=reason)
        self.core.memory.add(name, ev)
        self._witness(channel.id, ev)
        try:
            wh = await self.fetch_webhook(info["webhook_id"])
            await wh.delete(reason="Character left")
        except discord.HTTPException:
            pass
        log(f"[discord] {name} left #{channel.name}")

    # ---- message intake -----------------------------------------------------
    async def _entry_from_message(self, message, with_images=True):
        mem = self.core.memory
        if message.id in mem.by_msg:
            return mem.by_msg[message.id]
        char_name = self.webhook_owner(message.webhook_id) if message.webhook_id else None
        reply_ref = None
        if message.reference and message.reference.message_id:
            parent = mem.by_msg.get(message.reference.message_id)
            reply_ref = parent.ref if parent else None
        images = []
        if with_images:
            b = self.behavior
            images = await images_from_message(self.http_session, message, b["max_image_side"],
                                               b["fetch_image_links"])
        text = message.clean_content or ""
        if not text and message.stickers:
            text = "[sticker: " + ", ".join(s.name for s in message.stickers) + "]"
        if with_images and self.behavior.get("voice_messages", True):
            heard = await self._hear_voice_messages(message)
            text = (text + "\n" + heard).strip() if heard else text
        if with_images and not char_name and self.behavior.get("link_previews", True):
            seen = await self._link_previews(message.content or "")
            text = (text + "\n" + seen).strip() if seen else text
        author = char_name or message.author.display_name
        self.names[message.author.id] = author
        return mem.new_entry(msg_id=message.id, channel_id=message.channel.id,
                             channel_name=message.channel.name, author=author,
                             author_id=message.author.id, char_name=char_name, text=text,
                             images=images, reply_ref=reply_ref,
                             ts=message.created_at.timestamp())

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

    async def _hear_voice_messages(self, message):
        """Discord voice messages (Ogg Opus attachments) -> "[voice message] what was said"."""
        out = []
        for a in message.attachments:
            if not a.is_voice_message() or (a.duration or 0) > 600:
                continue
            names = [i["name"] for i in self.present(message.channel.id).values()]
            try:
                data = await a.read()
                said = await asyncio.to_thread(self.core.stt.transcribe_file, data,
                                               f"Talking with {', '.join(names)}." if names else None)
            except Exception as e:
                log(f"[voice] couldn't transcribe {message.author.display_name}'s voice message: {e}", level="warn")
                out.append("[voice message that couldn't be understood]")
                continue
            log(f"[voice] {message.author.display_name}'s voice message: {said or '(nothing)'}")
            out.append(f"[voice message] {said}" if said else "[voice message with no speech]")
        return "\n".join(out)

    def _witness(self, channel_id, entry):
        for info in self.present(channel_id).values():
            self.core.memory.add(info["name"], entry)
        rec = self.recent.setdefault(channel_id, [])
        rec.append(entry)
        del rec[:-30]

    async def _backfill(self, channel, char_name):
        n = int(self.behavior["backfill_messages"])
        if n <= 0:
            return
        msgs = [m async for m in channel.history(limit=n)]
        msgs.reverse()
        for i, m in enumerate(msgs):
            if m.author.id == self.user.id or self.ignored(m) or \
                    m.type not in (discord.MessageType.default, discord.MessageType.reply):
                continue
            # only fetch images for the last few messages to keep summon fast
            e = await self._entry_from_message(m, with_images=i >= len(msgs) - 3)
            self.core.memory.add(char_name, e)
            rec = self.recent.setdefault(channel.id, [])
            if e not in rec:
                rec.append(e)
        self.recent[channel.id] = sorted(self.recent.get(channel.id, []), key=lambda x: x.ts)[-30:]

    async def on_message(self, message: discord.Message):
        if message.guild is None:          # DMs are ignored
            return
        if message.author.id == self.user.id:
            return
        if not self.allowed(message.channel.id):
            return
        if message.type not in (discord.MessageType.default, discord.MessageType.reply):
            return
        if self.ignored(message) or await self.typed_command(message):
            return
        if not self.present(message.channel.id):
            return
        entry = await self._entry_from_message(message)
        self._witness(message.channel.id, entry)
        self.schedule(message.channel)

    async def on_raw_message_edit(self, payload):
        e = self.core.memory.by_msg.get(payload.message_id)
        if e and "content" in payload.data:
            e.text = payload.data["content"]
            self.core.memory.dirty = True

    async def on_raw_message_delete(self, payload):
        self.core.memory.remove_message(payload.message_id)

    async def on_raw_reaction_add(self, payload):
        await self._reaction(payload, added=True)

    async def on_raw_reaction_remove(self, payload):
        await self._reaction(payload, added=False)

    async def _reaction(self, payload, added):
        if payload.guild_id is None or payload.user_id == self.user.id:
            return
        if not self.allowed(payload.channel_id) or not self.present(payload.channel_id):
            return
        target = self.core.memory.by_msg.get(payload.message_id)
        if not target:
            return
        who = payload.member.display_name if payload.member else self.names.get(payload.user_id, "Someone")
        ch = self.get_channel(payload.channel_id)
        verb = "reacted" if added else "removed their reaction"
        snippet = (target.text[:50] + "…") if len(target.text) > 50 else target.text
        ev = self.core.memory.new_entry(
            kind="event", channel_id=payload.channel_id, channel_name=getattr(ch, "name", "?"),
            text=f'{who} {verb} {payload.emoji} {"to" if added else "on"} #{target.ref} ({target.author}: "{snippet}")')
        self._witness(payload.channel_id, ev)
        # reactions are seen but never trigger a reply

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
            log(f"[discord] error while responding: {e!r}")
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
        return True

    def tools_for(self, voice=False):
        # in voice chat nothing can be shown, so reactions, pictures and GIFs are left out
        return tool_list(self.tool_on, exclude=("add_reaction", "view_profile_picture", "draw_image", "post_gif")
                         if voice else ())

    # ---- context meter -------------------------------------------------------
    def note_usage(self, char_name):
        """Remember the real size (from llama-server) of the character's biggest call this turn."""
        if self.core.llm.peak_tokens:
            self.context_used[char_name.lower()] = (self.core.llm.peak_tokens, time.time())

    def context_estimate(self, channel, char):
        """Tokens the character's next prompt would take right now (estimate, nothing is trimmed)."""
        b, lc = self.behavior, self.cfg["llm"]
        tools = self.tools_for()
        system = self._system_prompt(channel, char, tools=tools)
        budget = int(lc["ctx_size"]) - int(lc["max_tokens"]) - 256 - tools_tokens(tools)
        msgs = self.core.memory.build_messages(char.name, system, channel.id, budget, int(b["image_token_cost"]),
                                               int(b["max_images_in_context"]), dry_run=True)
        return self.core.memory.count_tokens(msgs, int(b["image_token_cost"])) + tools_tokens(tools)

    def context_report(self, channel_id=None):
        """[(character, channel name, next prompt estimate, last real usage or None, context size)]."""
        size = int(self.cfg["llm"]["ctx_size"])
        out = []
        for cid, chars in list(self.state["summoned"].items()):
            if channel_id is not None and str(cid) != str(channel_id):
                continue
            ch = self.get_channel(int(cid))
            for info in list(chars.values()):
                char = self.core.characters.get(info["name"])
                if not ch or not char:
                    continue
                last = self.context_used.get(char.name.lower())
                out.append((char.name, ch.name, self.context_estimate(ch, char), last[0] if last else None, size))
        return out

    async def cmd_context(self, interaction):
        rows = self.context_report(interaction.channel_id)
        if not rows:
            return await interaction.response.send_message("No characters are here.", ephemeral=True)
        lines = [f"**Context** (model limit {rows[0][4]:,} tokens, {int(self.cfg['llm']['max_tokens']):,} "
                 "of it kept free for the reply):"]
        for name, _ch, est, last, size in rows:
            lines.append(f"**{name}** `{meter_bar(est, size)}` ≈{est:,} / {size:,} ({est * 100 // size}%)"
                         + (f" · last reply used {last:,}" if last else ""))
        lines.append("-# Older messages are dropped automatically when it's full; notes are kept.")
        await interaction.response.send_message("\n".join(lines)[:1990])

    def _system_prompt(self, channel, char, present=None, tools=None):
        tools = self.tools_for() if tools is None else tools
        names = {t["function"]["name"] for t in tools}
        native = self.cfg["llm"].get("use_native_tools", True)
        kinds, hints = [], []
        for cmd, kind, hint in COMMAND_HINTS:
            if cmd in names:
                if kind not in kinds:
                    kinds.append(kind)
                if hint:
                    hints.append(hint)
        commands = COMMAND_LINES.format(kinds=", ".join(kinds), hints=", ".join(hints) or "whenever useful") \
            if names else ""
        notebook = ""
        if names & {"list_notes", "read_note", "write_note"}:
            write = "Write down things worth remembering (names, facts people tell you about themselves, " \
                    "promises, plans) as soon as you learn them. " if "write_note" in names else ""
            notebook = NOTEBOOK_LINE.format(write=write, notes=Notebook(char.key).list())
        return SYSTEM_RULES.format(
            name=char.name, prompt=char.prompt.strip(), channel=channel.name, guild=channel.guild.name,
            present=", ".join(present or [i["name"] for i in self.present(channel.id).values()]),
            commands=commands, notebook=notebook,
            pfp=" (and profile pictures, if you ask to view one)" if "view_profile_picture" in names else "",
            tool_text="" if native or not tools else "\n" + tool_text_instructions(tools))

    async def respond(self, channel, speaker):
        info = self.present(channel.id).get(speaker.lower())
        char = self.core.characters.get(speaker)
        if not info or not char:
            return
        llm = self.core.llm
        if not llm.ready:
            why = f"suspended while {llm.suspended_by} is running" if llm.suspended_by else "not loaded"
            log(f"[discord] {speaker} can't reply: model {why}")
            return
        b, lc = self.behavior, self.cfg["llm"]
        budget = int(lc["ctx_size"]) - int(lc["max_tokens"]) - 256
        tools = self.tools_for()
        system = self._system_prompt(channel, char, tools=tools)
        budget -= tools_tokens(tools)
        msgs = self.core.memory.build_messages(char.name, system, channel.id, budget,
                                               int(b["image_token_cost"]), int(b["max_images_in_context"]))
        log(f"[discord] {char.name} is thinking in #{channel.name} ({len(msgs) - 1} context entries)")

        typing = asyncio.create_task(self._typing(channel)) if b.get("typing_indicator") else None
        review, finalize = self._link_checker(char, tools) if b.get("check_links", True) else (None, None)
        llm.peak_tokens = 0
        try:
            res = await run_turn(llm, msgs, tools, char.name,
                                 on_react=lambda a: self._react(channel, char.name, a),
                                 on_pfp=lambda name: self._profile_picture(channel, name),
                                 handlers=self._local_tools(char, channel, info), review=review,
                                 decide=bool(b.get("decide_commands", True) and lc.get("use_native_tools", True)))
            if finalize and res.text:
                res.text = await finalize(res.text)
        finally:
            if typing:
                typing.cancel()
            self.note_usage(char.name)

        self._remember_actions(char, channel.id, channel.name, res)
        if res.silent:
            log(f"[discord] {char.name} chose to stay silent")
        elif res.text:
            await self._send(channel, info, char, res.text)
        if res.leave:
            await self.remove_character(channel, char.name, f"{char.name} decided to leave #{channel.name}.")
            await self.announce(channel, f"👋 **{char.name}** left the channel.")
        if self.core.memory.dirty:
            self.core.memory.save()

    async def _send(self, channel, info, char, text="", file=None):
        wh = discord.Webhook.from_url(info["webhook_url"], client=self)
        chunks = [text[i:i + 1990] for i in range(0, len(text), 1990)] or [""]
        mentions = discord.AllowedMentions(everyone=False, roles=False, users=True)
        for i, chunk in enumerate(chunks):
            extra = {"file": file} if file and i == len(chunks) - 1 else {}
            try:
                await wh.send(chunk, username=char.name, allowed_mentions=mentions, wait=True, **extra)
            except discord.NotFound:  # webhook was deleted by someone; recreate it
                new = await self._create_webhook(channel, char)
                info.update(webhook_id=new.id, webhook_url=new.url)
                self._save()
                wh = new
                if file:
                    file.reset()
                await wh.send(chunk, username=char.name, allowed_mentions=mentions, wait=True, **extra)

    @staticmethod
    async def announce(channel, text):
        """A short notice from the bot itself (characters don't see the bot's own messages)."""
        if channel is None:
            return
        try:
            await channel.send(text, allowed_mentions=discord.AllowedMentions.none())
        except discord.HTTPException as e:
            log(f"[discord] couldn't post in #{getattr(channel, 'name', '?')}: {e}")

    def _remember_actions(self, char, channel_id, channel_name, res):
        """Commands leave no trace in the chat history, so the model would only ever see itself
        talking. Recording them (privately) shows it that it really uses its commands."""
        mem = self.core.memory
        for name, args in res.actions:
            what = describe_action(name, args)
            if what:
                mem.add(char.name, mem.new_entry(kind="event", channel_id=channel_id, channel_name=channel_name,
                                                 text=f"{char.name} {what}."))

    @staticmethod
    async def _typing(channel):
        """Discord shows "typing…" for ~10 s per signal and only clears it early when the bot account
        itself posts, which webhook replies don't count as. Refresh rarely (not every 5 s like
        discord.py) so the indicator fades as soon as possible after the reply."""
        try:
            while True:
                await channel._state.http.send_typing(channel.id)
                await asyncio.sleep(9)
        except (discord.HTTPException, asyncio.CancelledError):
            pass

    def _local_tools(self, char, channel=None, info=None):
        """`channel`/`info` (the webhook) are needed for commands that post something."""
        nb = Notebook(char.key)
        drawn = []

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
            await self._send(channel, info, char, file=discord.File(io.BytesIO(png), "drawing.png"))
            seen = encode_image(png, int(self.behavior["max_image_side"]))
            note = "[system] Your drawing was posted in the chat. This is what it looks like:"
            return [{"type": "text", "text": note}, {"type": "image_url", "image_url": {"url": seen}}] \
                if seen else "Your drawing was posted in the chat."

        async def gif(args):
            q = str(args.get("query") or "").strip()
            try:
                url, title = await search_gif(self.http_session, self.cfg["gifs"].get("klipy_key"), q)
            except GifError as e:
                log(f"[gifs] {char.name} couldn't post a GIF: {e}", level="warn")
                return f"No GIF posted: {e}."
            log(f"[gifs] {char.name} posted a GIF for {q!r}")
            await self._send(channel, info, char, url)
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

        return {"get_current_time": current_time, "list_notes": nb.list, "read_note": nb.read,
                "write_note": logged(nb.write, "wrote"), "delete_note": logged(nb.delete, "deleted"),
                "web_search": search, "read_webpage": read_page,
                **({"draw_image": draw, "post_gif": gif} if channel and info else {})}

    async def _react(self, channel, char_name, args):
        """Returns a result line for the tool loop."""
        try:
            ref = int(str(args.get("message_ref", "")).lstrip("#"))
        except ValueError:
            return "Invalid message number."
        emoji = str(args.get("emoji", "")).strip()
        target = self.core.memory.by_ref.get(ref)
        if not target or not emoji or not target.msg_id:
            return f"Message #{ref} not found."
        ch = self.get_channel(target.channel_id) or channel
        try:
            msg = ch.get_partial_message(target.msg_id)
            await msg.add_reaction(emoji)
        except discord.HTTPException as e:
            log(f"[discord] {char_name} couldn't react with {emoji}: {e}")
            return f"Couldn't react with {emoji} (use a standard unicode emoji)."
        log(f"[discord] {char_name} reacted {emoji} to #{ref}")
        ev = self.core.memory.new_entry(kind="event", channel_id=ch.id, channel_name=ch.name,
                                        text=f"{char_name} reacted {emoji} to #{ref}")
        self._witness(ch.id, ev)
        return f"Reacted {emoji} to #{ref}."

    async def _profile_picture(self, channel, name):
        uid = next((i for i, n in self.names.items() if n.lower() == name.lower().lstrip("@")), None)
        if uid is None:
            return [{"type": "text", "text": f"[system] No one called {name} was found."}]
        try:
            member = channel.guild.get_member(uid) or await channel.guild.fetch_member(uid)
        except discord.HTTPException:
            member = None
        img = None
        if member:
            img = await fetch_image(self.http_session, member.display_avatar.with_size(256).url,
                                    int(self.behavior["max_image_side"]))
        if not img:
            return [{"type": "text", "text": f"[system] Couldn't load {name}'s profile picture."}]
        return [{"type": "text", "text": f"[system] This is {name}'s profile picture:"},
                {"type": "image_url", "image_url": {"url": img}}]
