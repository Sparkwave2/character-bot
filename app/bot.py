"""Discord side: slash commands, webhook characters, message intake (the conversation itself is chat.py)."""
import asyncio
import io
import re
from typing import Optional

import aiohttp
import discord
from discord import app_commands

from .chat import ChatLogic
from .config import load_state, save_state
from .log import log
from .media import fetch_image, images_from_message

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


class CharacterBot(ChatLogic, discord.Client):
    log_tag = "discord"

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
        self.tasks = {}                         # channel id -> asyncio.Task
        self.phase = {}                         # channel id -> "waiting" | "generating"
        self.followup = {}                      # channel id -> trigger to run after generation
        self.recent = {}                        # channel id -> list of recent Entry (turn-taking)
        self.names = {}                         # user id -> display name
        self.voice = {}                         # guild id -> voice.VoiceSession
        self.on_change = lambda: None
        self._register_commands()

    # ---- helpers ------------------------------------------------------------
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
            return choices([c for c in store.characters.values() if self.can_speak(c)], current)

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
                         + (" 🔊" if self.can_speak(c) else ""))
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
            chars = [c for c in chars if self.can_speak(c)]
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
        """Sends everyone home first (like /gohome), then wipes their memory, so the leave message
        doesn't stay behind as the first thing they remember."""
        names = [c["name"] for c in self.present(interaction.channel_id).values()]
        if not names:
            return await interaction.response.send_message("Nobody is here.", ephemeral=True)
        await interaction.response.send_message(
            "🧹 " + ", ".join(f"**{n}**" for n in names) + " went home and forgot everything.")
        task = self.tasks.pop(interaction.channel_id, None)
        if task:
            task.cancel()
        for n in names:
            await self.remove_character(interaction.channel, n, f"{n} was sent home from #{interaction.channel.name}.")
        for n in names:
            self.core.memory.forget(n)
        self.recent.pop(interaction.channel_id, None)
        self.core.memory.save()
        log(f"[discord] {interaction.user.display_name} sent {', '.join(names)} home and wiped their memory")

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
        if not self.can_speak(char):
            return await reply(f"**{char.name}** doesn't have a voice, so they can't join voice chat.",
                               ephemeral=True)
        if not self.local_voice and not self.core.tts.key:
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
        if self.local_voice:
            try:
                await asyncio.to_thread(self.core.local_tts.load)   # downloads Kokoro the first time
            except Exception as e:
                return await interaction.followup.send(f"The local voice couldn't start: {e}")
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
        files = [a.filename for a in message.attachments
                 if not a.is_voice_message() and not (a.content_type or "").startswith("image/")]
        if files:
            text = (text + "\n" + " ".join(f"[attached file: {f}]" for f in files)).strip()
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

    @property
    def local_voice(self):
        return bool(self.cfg["voice"].get("local_tts"))

    def can_speak(self, char):
        """With the local voice everyone can talk (there's a default voice); ElevenLabs needs a voice per character."""
        return self.local_voice or bool(char.voice)

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
                last = self.core.context_used.get(char.name.lower())
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

    @staticmethod
    async def _typing(channel, char=None):
        """Discord shows "typing…" for ~10 s per signal and only clears it early when the bot account
        itself posts, which webhook replies don't count as. Refresh rarely (not every 5 s like
        discord.py) so the indicator fades as soon as possible after the reply."""
        try:
            while True:
                await channel._state.http.send_typing(channel.id)
                await asyncio.sleep(9)
        except (discord.HTTPException, asyncio.CancelledError):
            pass

    async def post_file(self, channel, info, char, name, data):
        await self._send(channel, info, char, file=discord.File(io.BytesIO(data), name))

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
