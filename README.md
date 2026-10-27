# Character Bot

A local AI character bot for Discord. A Qwen vision model (GGUF) runs through llama.cpp, and characters speak in Discord through webhooks, each with its own name and profile picture. Everything is configured from a dark-mode desktop GUI.

## Setup

1. **Python 3.10+.** Double-click `run.bat`. On first run it creates `.venv` and installs the requirements.
2. **Model.** You need a Qwen-VL GGUF *and* its matching `mmproj-*.gguf` vision projector, for example from the `ggml-org/Qwen2.5-VL-7B-Instruct-GGUF` or `Qwen3-VL` repos on Hugging Face. Select both files under **Model**, then click **Load**. You don't need to install llama.cpp yourself (see below).
3. **Discord bot.** Do this in https://discord.com/developers/applications:
   - Create an application, then a bot, and copy its token into **Discord → Bot token**.
   - Under *Bot → Privileged Gateway Intents*, enable **MESSAGE CONTENT INTENT**.
   - Click **Save & connect**, then **Copy invite link**, and open that link to add the bot to your server. The link already requests the Manage Webhooks, Send Messages, Add Reactions, Read Message History, Connect and Speak permissions. A bot you invited before voice chat existed may still need Connect and Speak.
   - Tick the channels the bot may speak in and click **Save selection**. The bot does nothing in channels you leave unticked, and it ignores DMs.
4. **Voice chat (optional).** Put your ElevenLabs API key on the **Voice** page and give characters a voice (see below). Speech recognition uses Whisper, which runs locally; the model (~500 MB for `small`) downloads into the **Model folder** on the first `/join`. Whisper uses the GPU when the CUDA libraries are installed:

   ```bash
   .venv\Scripts\python -m pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"
   ```

   Without them it runs on the CPU, which is slower.
5. **Drawing (optional).** On the **Drawing** page, set the folder of your AUTOMATIC1111 or Forge WebUI (the one containing `webui.bat`), then use **Pick…** to choose a model. Checkpoints of 5.5 GB or more are marked SDXL; that's a guess based on file size. Then turn on **Draw pictures** on the Behavior page. The first drawing starts the WebUI in the background (API only, no window, port 7861) and closes it when the app closes. A WebUI you start yourself with `--api` works too.
   - **GPU memory.** Plain A1111 can't share the GPU well with llama-server, so the default launch option is `--medvram-sdxl`, and the model is unloaded after a few idle minutes. [Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge) is a fork of A1111 that manages GPU memory on its own, offloading parts of the model when VRAM is tight. It uses the same API and folder layout, so you can switch by pointing the folder setting at a Forge install. While a game from the Games page is running, drawing is paused and the WebUI is stopped.
6. **GIFs (optional).** Get a free API key at [partner.klipy.com](https://partner.klipy.com) and put it under **Behavior → Commands**.

## Built-in engine (llama.cpp)

llama.cpp is managed automatically. The first time you click **Load** (or **Install** under **Model → Engine**), the app downloads the right prebuilt `llama-server` from the official ggml-org GitHub releases into `llama.cpp/`:

- **Auto** picks CUDA if `nvidia-smi` finds an NVIDIA GPU, matching the CUDA build to your driver. Otherwise it picks Vulkan if a Vulkan driver is present, and CPU if not. macOS uses the Metal build. You can force CUDA, Vulkan, CPU or ROCm instead.
- On Windows, the CUDA build also needs the CUDA runtime zip (~400 MB). That download only happens once; later updates reuse it and fetch only the ~160 MB main build.
- With **Update on startup** on, the app checks for a newer build each time it starts and installs it into a new folder while the current one keeps running. It then restarts the model on the new build and deletes the old one.
- If you'd rather use your own build, set **Advanced → Custom llama-server**. While that's set, the managed install is ignored.

## Characters

Each character is a JSON file in `characters/`:

```json
{
  "name": "Luna",
  "description": "A sleepy witch who knows too much",
  "prompt": "You are Luna, ...",
  "avatar": "luna.png",
  "voice": "21m00Tcm4TlvDq8ikWAM"
}
```

- `avatar` is an image path relative to the JSON file, or a `data:image/png;base64,...` URI.
- `description` is optional. It's shown in `/characters` and in the autocomplete lists.
- `voice` is an ElevenLabs voice ID, or the name of a voice in your ElevenLabs account. A character without a voice can't join voice chat.

You can create and edit characters in the **Characters** page, which copies the chosen picture next to the JSON. **Pick…** next to the voice field lists the voices in your ElevenLabs account, labelled by type. Free ElevenLabs plans can use Built-in and Designed voices (and your own clones) through the API, but not Voice Library voices; those are greyed out and marked "Library (paid plan)".

## In Discord

- `/summon <character>`: creates a webhook with the character's name and avatar, reads recent channel history and greets the channel.
- `/dismiss <character|all>`: removes the character and deletes its webhook.
- `/gohome`: dismisses every character in the channel (same as `/dismiss all`).
- `/forget`: wipes the chat memory of every character in the channel. Their notes are kept.
- `/who`: shows which characters are in the channel (and who is in voice chat), visible to everyone.
- `/context`: shows how full each character's context is in this channel. You get a bar and the estimated size of its next prompt out of the model's limit, plus the real size of its last reply, as reported by llama-server. The **Dashboard** has the same meter for every summoned character, updated every few seconds.
- `/characters`: lists who is in the channel and who is available, with their descriptions. 🔊 marks characters that can join voice chat.
- `/summon` or `/join` without a character opens a dropdown that shows each character's description. Only the person who asked can use it. While you type a name, the autocomplete list shows names only, since Discord's autocomplete has no room for descriptions.
- `/join <character>`: the character joins the voice channel you're in and talks with everyone there. Only one character can be in voice per server; joining another one replaces it.
- `/leave`: the character in voice chat leaves. It also leaves on its own when everyone else has left.
- Typing any of these as a normal message (e.g. `/summon Mayo`) works too, in allowed channels. The bot answers with a normal reply instead of a private one, and characters don't see the command message.

Characters can also act through MCP-style tool calls (OpenAI tool schemas, which llama-server's `--jinja` turns into Qwen's native `<tool_call>` format):

| Tool | Effect |
|---|---|
| `add_reaction(message_ref, emoji)` | React to message `#N`. Several emojis (`😂🔥`) become separate reactions |
| `leave_channel(reason)` | The character leaves on its own (after sending any goodbye text), and the bot posts a 👋 notice |
| `stay_silent()` | Skip replying |
| `view_profile_picture(name)` | Look at someone's avatar before replying (can be turned off under Behavior) |
| `web_search(query)`, `read_webpage(url)` | Search DuckDuckGo and read a page as text |
| `read_note(name)`, `write_note(name, text, mode)` | The character's personal Markdown notebook in `notes/<character>/`. `mode` is append (default), replace or delete; the list of notes is already in the prompt |
| `draw_image(prompt)` | Draw a picture with Stable Diffusion and post it; the character sees the result (off by default, see Drawing) |
| `post_gif(query)` | Post the best-matching GIF from KLIPY (needs a KLIPY key) |

Every command has its own switch under **Behavior → Commands**. The system prompt only mentions commands that are switched on. In voice chat, reactions, profile pictures, drawings and GIFs are left out.

Before each reply in text chat, the model is asked whether a command fits right now. It must call at least one command, or `just_reply`. This costs one extra model call per reply and can be turned off under **Behavior → Ask about commands before each reply**. If the server can't force a tool call, the step switches itself off. Every command a character runs is logged as `[tools] Name used command(...)`.

Small models often narrate a command instead of running it ("*reacts with 😂*", "I'll write that down", "let me look it up"). When a reply says that without the matching command, the character is told once that nothing happened and asked to use the command and rewrite its message; the log shows `[tools] … asking again`. Commands a character used (notes, searches, checking the time) are also kept in its own memory, so it sees in its history that it really uses them.

## How it behaves

- **One speaker at a time.** A global lock means only one character generates at a time. The program picks who replies in this order: whoever was replied to, then whoever was named, then whoever spoke most recently, then a random character.
- **Characters talk to each other.** Messages from characters are visible to the other characters. Characters can answer each other, up to the *Max character chain* limit, so they can't loop forever.
- **Memory carries across channels.** Each character has one memory covering every channel it has been in. Messages from other channels are tagged `in #channel`. Memory is saved to `memory.json` and survives restarts.
- **Small fixed overhead.** The rules and command definitions are sent with every prompt, so they're kept short: about 1,100 tokens with every command on. The current date and time are added at the end of each prompt, so characters always know it without a command. Putting it at the end keeps llama-server's prompt cache valid.
- **Bigger context, same VRAM budget.** The default context is 32,768 tokens with an 8-bit KV cache (Model → *Context memory type* `q8_0`), which uses half the memory of `f16`. For Qwen3-VL-30B-A3B that's about 49 KB per token, about 1.6 GB at 32k.
- **Context scrolls automatically.** When building a prompt, the newest messages are kept until the context budget (`ctx_size − max_tokens`) is full. Older messages are dropped permanently. Only the most recent *N* images are sent as images; older ones become `[image]`.
- **`//` messages are ignored.** A message starting with `//` is out of character: characters never see it and it never triggers a reply.
- **Notes outlive the context.** Each character has a notebook in `notes/<character>/*.md` (at most 50 notes of 8000 characters each). Its note list is in the system prompt, and the files can be edited by hand.
- **Typing indicator.** Discord can't cancel a typing indicator early, and webhook replies don't clear it, so it fades on its own a few seconds after the reply. Turn it off under Behavior if that bothers you.
- **Reactions** are recorded as events the characters can see, but a reaction never triggers a reply.
- **Identity.** Characters only see people's display names, plus profile pictures if they ask through the tool. No other profile details are passed to the model.
- **Voice messages.** Discord voice messages are transcribed with the same local Whisper model as voice chat, so characters see `[voice message] what was said`. This can be turned off under Behavior.
- **Links people post.** Up to two linked pages per message are opened. The loaded model writes a short description, which characters see as `[link …: title — what the page says]`. Characters can use `read_webpage` to read the whole page. Dead links are marked as dead.
- **Links characters post.** Before a message goes out, every link in it is opened:
  - A dead link (404, a site that doesn't exist, or a "page not found" page) is sent back so the character can fix it. If the link is still in the rewrite, it's removed from the message.
  - An image link is shown to the character as a picture, so it can confirm that's what it meant.
  - Links that can't be verified (403, rate limits, timeouts) are let through.
  - Both of these can be turned off under Behavior.
- **Images.** Characters see uploaded images, stickers, embed images and pasted image links. Links without an image extension are fetched only if the server answers with an `image/*` content type.
- **Voice chat.** Each speaker's audio is collected until they pause (*Pause that ends speech* on the Voice page). Whisper then transcribes it, the character answers, and ElevenLabs speaks the answer. What's said in voice goes into the character's memory, tagged `in #General (voice)`, so the character remembers it in text chat too. Emojis, links and `*actions*` are left out of the spoken reply. Discord's voice encryption (DAVE) is handled for both sending and receiving.
- **Games.** Add process names (for example `eldenring.exe`) on the **Games** page. While any of them is running, llama-server is stopped, which frees the VRAM. It reloads automatically when the game closes. While the model is unloaded, characters stay quiet.

## Files

| File | Purpose |
|---|---|
| `config.json` | All settings (written by the GUI) |
| `state.json` | Which characters are summoned where, and their webhooks |
| `memory.json` | Character memories (text only) |
| `notes/` | Each character's notes |
| `llama.cpp/` | Managed llama.cpp build and its `installed.json` manifest |
| `models/whisper/` | Whisper speech-recognition model (unless you set another folder) |
| `app/` | Source: `bot.py` (Discord), `voice.py` (voice chat), `stt.py` (Whisper), `tts.py` (ElevenLabs), `web.py` (search), `draw.py` (Stable Diffusion WebUI), `gifs.py` (KLIPY), `engine.py` (llama.cpp installer), `llm.py` (llama-server process), `memory.py` (context), `games.py`, `gui.py` |
