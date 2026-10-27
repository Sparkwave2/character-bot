# Character Bot

A local AI character bot for Discord. A Qwen vision model (GGUF) runs through llama.cpp, and characters speak in Discord through webhooks, each with its own name and profile picture. Everything is configured from a dark-mode desktop GUI.

## Setup

1. **Python 3.10+.** Double-click `run.bat`. On first run it creates `.venv` and installs the requirements.
2. **Model.** The easiest way is **Model → Choose a model…** (see below): it lists vision models that work with the app, marks which ones fit your graphics card, and downloads the one you pick. You can also bring your own: a vision GGUF *and* its matching `mmproj-*.gguf` projector, selected under **Model**. Then click **Load**. You don't need to install llama.cpp yourself (see below).
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

   **Local voice (optional).** Instead of ElevenLabs, characters can speak with [Kokoro](https://github.com/thewh1teagle/kokoro-onnx), a small speech model that runs on this PC. It's free, has no quota, and starts talking in about 0.2 s on the GPU. Install it like this; `--no-deps` stops pip from replacing the GPU build of onnxruntime with the CPU one:

   ```bash
   .venv\Scripts\python -m pip install onnxruntime-gpu espeakng-loader phonemizer nvidia-cuda-runtime-cu12 nvidia-cufft-cu12 nvidia-curand-cu12
   ```

   ```bash
   .venv\Scripts\python -m pip install --no-deps kokoro-onnx
   ```

   Then turn on **Voice → Use the local voice**. The model (~350 MB) downloads into the **Model folder** on the first `/join`. Choose a voice per character with the **Local** Pick… button on the Characters page; characters without one use the default voice. With the local voice on, every character can join voice chat. Without the GPU packages Kokoro runs on the CPU, at about 3× real time.
5. **Drawing (optional).** On the **Drawing** page, set the folder of your AUTOMATIC1111 or Forge WebUI (the one containing `webui.bat`), then use **Pick…** to choose a model. Checkpoints of 5.5 GB or more are marked SDXL; that's a guess based on file size. Then turn on **Draw pictures** on the Behavior page. The first drawing starts the WebUI in the background (API only, no window, port 7861) and closes it when the app closes. A WebUI you start yourself with `--api` works too.
   - **GPU memory.** Plain A1111 can't share the GPU well with llama-server, so the default launch option is `--medvram-sdxl`, and the model is unloaded after a few idle minutes. [Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge) is a fork of A1111 that manages GPU memory on its own, offloading parts of the model when VRAM is tight. It uses the same API and folder layout, so you can switch by pointing the folder setting at a Forge install. While a game from the Games page is running, drawing is paused and the WebUI is stopped.
6. **GIFs (optional).** Get a free API key at [partner.klipy.com](https://partner.klipy.com) and put it under **Behavior → Commands**.

## Built-in engine (llama.cpp)

llama.cpp is managed automatically. The first time you click **Load** (or **Install** under **Model → Engine**), the app downloads the right prebuilt `llama-server` from the official ggml-org GitHub releases into `llama.cpp/`:

- **Auto** picks CUDA if `nvidia-smi` finds an NVIDIA GPU, matching the CUDA build to your driver. Otherwise it picks Vulkan if a Vulkan driver is present, and CPU if not. macOS uses the Metal build. You can force CUDA, Vulkan, CPU or ROCm instead.
- On Windows, the CUDA build also needs the CUDA runtime zip (~400 MB). That download only happens once; later updates reuse it and fetch only the ~160 MB main build.
- With **Update on startup** on, the app checks for a newer build each time it starts and installs it into a new folder while the current one keeps running. It then restarts the model on the new build and deletes the old one.
- If you'd rather use your own build, set **Advanced → Custom llama-server**. While that's set, the managed install is ignored.

## Choosing a model

**Model → Choose a model…** opens a built-in list of vision models (Qwen3-VL 2B–32B, Gemma 3, Mistral Small 3.2, plus abliterated versions), each tagged:

- `uncensored`: refusals removed, so characters stay in their role.
- `fast`: replies quickly even on small GPUs.
- `smart`: follows long character prompts and commands well.
- `MoE`: big-model knowledge at small-model speed.
- `official`: from the original authors.

Each entry shows how much VRAM it needs with your current context size and context memory type (the model file, its projector, the context, and about 0.8 GB of working memory), and whether that **fits**, is **tight** or is **too big** for your card. The card's VRAM is read with `nvidia-smi`; for other GPUs, type it in. The best model that fits is marked ★ **Recommended**. **Download & use** fetches the model and its projector from Hugging Face into **Download folder** (one subfolder per repository) and makes it the model that loads; a loaded model is restarted on it. Stopped downloads continue where they left off. The progress shows under **Model files**.

## Local chat

The **Chat** page is a channel inside the app (`#local-chat`), for talking to characters without Discord. Characters behave as they do on Discord, with the same prompt, memory (it shows up as `#local-chat` in their history), notebook and turn-taking. Characters can talk to each other there too.

- **+ Summon** adds a character; ✕ next to a name dismisses it. **Send everyone home**, **Forget everything** and **Clear the chat** are next to them. Typing `/summon <name>`, `/dismiss <name|all>`, `/gohome`, `/forget`, `/who` or `/clear` works too. Clearing only empties the window; memories stay.
- Messages use Discord markdown: `**bold**`, `*italic*`, `__underline__`, `~~strike~~`, `` `code` ``, code blocks, `> quotes`, `# headings`, `-# small text`, lists, `||spoilers||` (click to reveal) and links.
- **+** attaches pictures or files. Ctrl+V pastes a picture, or files copied in Explorer. Pictures show as previews; click one for full size. Characters see the pictures, and they read small text files (up to 6,000 characters). Clicking a file opens it in a viewer with **Save as…** and **Open with…**.
- Commands that need Discord are left out: reactions, profile pictures and GIFs. Drawing and `create_file` work, and their pictures and files appear in the chat.
- There's no voice in the local chat. Voice chat is Discord-only.
- `//text` is out of character, as on Discord. Under **You**, set the name characters see (default: your Windows user name). Click the circle next to it to choose your picture and crop it, the same way as a character's. The chat, its attachments and your picture (`you.png`) are saved in `localchat/`.
- Pictures in the window (chat previews, avatars, the character list) are loaded in the background, so a long chat full of images opens without freezing the app. They appear a moment later.

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
- `voice` is an ElevenLabs voice ID, or the name of a voice in your ElevenLabs account. A character without a voice can't join voice chat, unless the local voice is on.
- `local_voice` (optional) is a Kokoro voice such as `af_bella` or `jf_alpha`, used when the local voice is on.

You can create and edit characters in the **Characters** page, which copies the chosen picture next to the JSON. **Pick…** next to the voice field lists the voices in your ElevenLabs account, labelled by type. Free ElevenLabs plans can use Built-in and Designed voices (and your own clones) through the API, but not Voice Library voices; those are greyed out and marked "Library (paid plan)".

## In Discord

- `/summon <character>`: creates a webhook with the character's name and avatar, reads recent channel history and greets the channel.
- `/dismiss <character|all>`: removes the character and deletes its webhook.
- `/gohome`: dismisses every character in the channel (same as `/dismiss all`).
- `/forget`: sends every character in the channel home (like `/gohome`), then wipes their chat memory. This way, their leave message isn't the first thing they remember afterwards. Their notes are kept.
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
| `get_current_time()` | Check the date and time |
| `view_profile_picture(name)` | Look at someone's avatar before replying (can be turned off under Behavior) |
| `web_search(query)`, `read_webpage(url)` | Search DuckDuckGo and read a page as text |
| `read_note(name)`, `write_note(name, text, mode)` | The character's personal Markdown notebook in `notes/<character>/`. `mode` is append (default), replace or delete; the list of notes is already in the prompt |
| `draw_image(prompt)` | Draw a picture with Stable Diffusion and post it; the character sees the result (off by default, see Drawing) |
| `post_gif(query)` | Post the best-matching GIF from KLIPY (needs a KLIPY key) |
| `create_file(filename, about)` | Make a file and post it as an attachment: code, a story, a list, CSV data and so on. The character only names the file and says what goes in it; the contents are then written in a separate model call (up to **Longest file**, 4,000 tokens by default), so files can be much longer than a chat message. Only the endings listed under **Behavior → File types characters can create** are allowed, and the character is told which. A copy is kept in `files/<character>/`. This is separate from the notebook, which is the character's private memory |

Every command has its own switch under **Behavior → Commands**. The system prompt only mentions commands that are switched on. In voice chat, reactions, profile pictures, drawings, GIFs and files are left out.

Before each reply in text chat, the model is asked whether a command fits right now. It must call at least one command, or `just_reply`. This costs one extra model call per reply and can be turned off under **Behavior → Ask about commands before each reply**. If the server can't force a tool call, the step switches itself off. Every command a character runs is logged as `[tools] Name used command(...)`.

Small models often narrate a command instead of running it ("*reacts with 😂*", "I'll write that down", "let me look it up"). When a reply says that without the matching command, the character is told once that nothing happened and asked to use the command and rewrite its message; the log shows `[tools] … asking again`. Commands a character used (notes, searches, checking the time) are also kept in its own memory, so it sees in its history that it really uses them.

## How it behaves

- **One speaker at a time.** A global lock means only one character generates at a time. The program picks who replies in this order: whoever was replied to, then whoever was named, then whoever spoke most recently, then a random character.
- **Characters talk to each other.** Messages from characters are visible to the other characters. Characters can answer each other, up to the *Max character chain* limit, so they can't loop forever.
- **Memory carries across channels.** Each character has one memory covering every channel it has been in. Messages from other channels are tagged `in #channel`. Memory is saved to `memory.json` and survives restarts.
- **Small fixed overhead.** The rules and command definitions are sent with every prompt, so they're kept short: about 1,100 tokens with every command on.
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
- **Voice chat.** Each speaker's audio is collected until they pause (*Pause that ends speech* on the Voice page). Whisper then transcribes it, the character answers right away (text chat's reply delay doesn't apply), and ElevenLabs or the local voice speaks the answer. The speech is streamed, so playback starts while the rest is still being made. Each reply logs where the time went, for example `[voice] Mayo started talking 2.1 s after the speaker stopped: pause detection 0.8, Whisper 0.3, model 0.7 (…), local voice 0.2`. If someone talks over the character for more than 0.4 s, it stops at once and drops the rest of its reply. Only the part it actually said is kept in its memory, marked `[interrupted]`, and it answers the new message instead. An answer that was still being written when someone spoke again is dropped too. This can be turned off on the Voice page. What's said in voice goes into the character's memory, tagged `in #General (voice)`, so the character remembers it in text chat too. Emojis, links and `*actions*` are left out of the spoken reply. Discord's voice encryption (DAVE) is handled for both sending and receiving.
- **Games.** Add process names (for example `eldenring.exe`) on the **Games** page. While any of them is running, llama-server is stopped, which frees the VRAM. It reloads automatically when the game closes. While the model is unloaded, characters stay quiet.

## Files

| File | Purpose |
|---|---|
| `config.json` | All settings (written by the GUI) |
| `logs/llm-verbose.log` | With **Model → Advanced → Verbose model log** on: every prompt exactly as the model reads it (llama-server renders the chat template, including the command definitions), followed by its raw reply and tool calls. Images appear as the template's image markers. Over 50 MB it moves to `llm-verbose.old.log`. **Open log** opens it |
| `state.json` | Which characters are summoned where, and their webhooks |
| `memory.json` | Character memories (text only) |
| `notes/` | Each character's notes |
| `files/` | Copies of the files characters made with `create_file` |
| `localchat/` | The local chat's messages (`chat.json`) and attachments |
| `llama.cpp/` | Managed llama.cpp build and its `installed.json` manifest |
| `models/whisper/` | Whisper speech-recognition model (unless you set another folder) |
| `app/` | Source: `bot.py` (Discord), `voice.py` (voice chat), `stt.py` (Whisper), `tts.py` (ElevenLabs and Kokoro), `web.py` (search), `draw.py` (Stable Diffusion WebUI), `gifs.py` (KLIPY), `engine.py` (llama.cpp installer), `llm.py` (llama-server process), `memory.py` (context), `games.py`, `gui.py` |
