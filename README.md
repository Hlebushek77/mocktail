# Mocktail Plus

[![Release](https://img.shields.io/github/v/release/blackixxce12/mocktail?include_prereleases&sort=semver)](https://github.com/blackixxce12/mocktail/releases)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Fork of komaruworld/mocktail](https://img.shields.io/badge/fork%20of-komaruworld%2Fmocktail-555)](https://github.com/komaruworld/mocktail)

**Mocktail Plus** is an unofficial fork of
[Mocktail](https://github.com/komaruworld/mocktail)
([mocktail.bigrat.space](https://mocktail.bigrat.space/)), which runs the
Android `x86_64` Roblox client on Linux. Plus is upstream Mocktail with a
settings window that opens before Roblox, fixes for signing in, buying Robux
and typing in text boxes, and NVIDIA and memory work on top. Everything else
is upstream's.

- Mocktail Plus is not affiliated with Roblox Corporation, VinegarHQ or the
  authors of Mocktail. Please report problems you only see in Plus to this
  repository, not to upstream.
- It does not distribute the Roblox client. Mocktail downloads it on the
  first start.
- It is licensed under the [Apache License 2.0](LICENSE), like upstream.
- The program is still called Mocktail: the command is `mocktail`, the app ID
  is `space.bigrat.mocktail`, and it uses the same files as upstream's
  native packages and AppImage.

![The settings window's Accounts page](assets/screenshots/plus-settings-accounts.png)

## How it works

```
Roblox APK -> signature and ABI checks -> Bionic + JNI -> SDL3 + Vulkan/OpenGL
```

Mocktail provides the Android ABI and JNI pieces the client expects, then
connects them to SDL3 and Vulkan or OpenGL on the Linux side. The APK is
downloaded on first launch (`updates.source`; APKPure is currently the only
download source) and is checked before any native code is loaded. It is
not bundled with Mocktail. The last working copy is kept in case an update
fails: with `updates.automatic` on, a new Roblox version passes isolated test
runs (two when Mocktail's compatibility list does not cover it yet) before it
replaces the current one.

## What Plus changes

Plus is based on upstream `main` at
[`273209b`](https://github.com/komaruworld/mocktail/commit/273209b)
("Fix angle platform selection"), 39 commits after the 1.0.4 release, so it
already has upstream's fixes since 1.0.4, such as the chat text fixes. The
list below summarizes what Plus changes; `git log --stat 273209b..main`
shows every change, file by file.

**Settings window**

- A GTK 4 / libadwaita window (`mocktail_launcher_ui`) opens before Roblox
  starts, with pages for graphics, display, performance, audio, accounts,
  integrations, network and updates, advanced settings and system
  information. See [The settings window](#the-settings-window).
- Every setting has a hint that says what it does on this computer and what
  is recommended here, an **ENV** badge while an environment variable
  overrides it, and an **Overrides Roblox setting** badge while it takes over
  one of Roblox's own in-game settings.
- Changes are written to `config.yaml` only when you press Save or Play. The
  editor changes only the lines it has to, so comments and layout stay, and
  it keeps copies of the file before its first save, before Restore Backup
  and before Reset All Settings.
- A Fast Flags editor for `fflags.json` that points out the flags Mocktail
  manages itself.
- Variables set by a desktop shortcut or a terminal can be moved into the
  settings and removed from your copy of the desktop entry.
- New `config.yaml` sections `display`, `account`, `engine` and `launcher`
  (see [Configuration](#configuration)), the command-line options
  `--launcher` and `--play` (`--no-launcher`), and the desktop entry actions
  "Play now" and "Mocktail Settings".

**Sign-in**

- Roblox's questions about Google Play Integrity are answered as
  "unavailable". Mocktail makes up no token and no verdict.
- When Roblox's own sign-in still asks for a device check that only a
  certified device can pass (Google Play Integrity, Apple Private Access
  Token or a console's platform token), Mocktail opens roblox.com's sign-in
  page instead and restarts once you have signed in there, so Roblox starts
  signed in.
- Stale sessions: clearing the sign-in window's Roblox cookies now really
  deletes them, and a session from the website window is checked with Roblox
  before it is saved. A session Roblox refuses is not saved.
- Several Roblox accounts can be saved and switched in the settings window,
  and you can start as a guest.
- Roblox's web windows (sign-in, verification, the Robux page) open as child
  windows of the game, so the compositor knows they belong to the game
  instead of placing them as separate applications. If a compositor tiles
  one beside a fullscreen game anyway, the game goes back to fullscreen once
  it closes.

**Robux purchases**

- Mocktail has no Google Play Billing, so buying Robux used to do nothing.
  Plus opens `https://www.roblox.com/upgrades/robux` in your default web
  browser instead, both from Roblox's Robux page and from purchase prompts
  inside experiences. See [Buying Robux](#buying-robux).

**Chat and text boxes**

- Each call into Roblox's TextBox code now runs in its own JNI local frame,
  as on Android, so the references it leaves behind are freed when it
  returns, and the text box position is polled at most once every 50 ms
  while a text box has focus.

**NVIDIA and graphics**

- On a Wayland session, NVIDIA cards with direct Vulkan can run on native
  Wayland automatically when the driver and the compositor support explicit
  sync. Upstream always uses XWayland there. See [NVIDIA](#nvidia).
- `display.server` chooses Wayland or X11 (XWayland) for the game window.
- Roblox may load its shader pack on several threads on NVIDIA cards, as it
  does on Intel and AMD. Upstream always made it use one thread there;
  `engine.nvidia_shader_mt: false` brings that back.
- `engine.gpu` chooses the graphics card on computers with more than one,
  including laptops with AMD integrated graphics and an NVIDIA card, where
  upstream always used the NVIDIA card.
- `engine.graphics_quality` forces a graphics quality level you choose, or
  leaves the level to Roblox's own setting (the default). A chosen level
  takes effect since 1.0.4-plus.2; before, Mocktail sent it under a name
  Roblox ignores.
- Since 1.0.4-plus.2 the performance preset no longer sends 33 client
  settings that Roblox 2.738 never reads. Roblox looks a setting up only
  under the prefix it registered it with (`F` or `DF`), and some of the
  names do not exist in it at all. Among them were the quality level 3 the
  settings window said the preset forced, a texture mip bias and low-end
  level of detail. Roblox never applied them, so the game behaves as before;
  the preset now sends 40 settings.
- With direct Vulkan, vertical sync "On" now stays synchronized with Mesa's
  drivers (AMD, Intel, NVK) when the frame rate limit is unlimited.
- The session log names the graphics card the game renders on, and how much
  ETC2 texture decoding a session did.

**Textures and stutter** (1.0.4-plus.2)

- The stutter that texture conversion caused while moving or turning the
  camera on NVIDIA cards is gone in our test. Roblox streams textures in
  the ETC2 format, which these cards cannot read, and Mocktail used to
  convert every one of them on the processor, on the game's render thread,
  with a staging buffer created and destroyed for every texture copy. Now
  the graphics card converts them, and the processor path that remains as
  the fallback is about seven times faster. See
  [ETC2 textures](#etc2-textures).
- **Compressed textures (BC)** (`engine.etc2_to_bc`), on by default: on
  cards without ETC2 the textures are converted to BC and stay compressed
  instead of being unpacked. In the test the game used about 800 MiB of
  video memory instead of about 1900 MiB.
- `MOCKTAIL_FRAME_STATS=1` logs frame pacing and texture work,
  `scripts/frame_stats_summary.py` summarizes such a log, and
  `MOCKTAIL_VK_CALL_STATS=1` counts the game's Vulkan calls. See
  [Measuring stutter](#measuring-stutter).

**Memory**

- The processor path of the texture conversion gives its scratch memory
  back after a loading burst: anything over 64 MiB, once no texture has
  been uploaded for 5 seconds. 1.0.4-plus.1 gave back anything over 32 MiB
  after every upload batch, which in one test took the game's anonymous
  memory from 1817 MiB to 1455 MiB at the same frame rate, but made each
  batch of a loading burst allocate the memory again.
- The description of `performance.memory_limit_mb` now says what it does: a
  watchdog stops Mocktail with exit status 137 once the game's resident
  memory plus swap reaches the limit; swap stays enabled.

**Translations**

- The settings window is in English and Russian and follows the system
  language. The desktop entry's description and its actions are in Russian
  too.

## Install

Mocktail Plus is published only as a package for CachyOS and Arch Linux on
the [Releases page](https://github.com/blackixxce12/mocktail/releases). There
is no Flatpak, AppImage, AUR, DNF or APT package of Plus.

Requirements:

- CachyOS or Arch Linux on `x86_64`. The package is built for the x86-64-v3
  level, so the CPU must support it (AVX2, BMI2, FMA and more; Intel Core
  from Haswell on and AMD Zen are new enough, but some Pentium, Celeron and
  Atom CPUs are not). `/lib/ld-linux-x86-64.so.2 --help` lists
  `x86-64-v3 (supported, searched)` on a CPU that has it.
- A Vulkan driver for the default `direct-vulkan` graphics backend, or
  OpenGL ES 3.0 for `graphics.backend: opengl`.

Download `mocktail-plus-1.0.4.plus.2-1-x86_64.pkg.tar.zst` from the
`v1.0.4-plus.2` release and install it:

```bash
sudo pacman -U ./mocktail-plus-1.0.4.plus.2-1-x86_64.pkg.tar.zst
```

The package conflicts with `mocktail`, `mocktail-bin`, `mocktail-git` and
`mocktail-local`, so it cannot be installed beside them: pacman offers to
remove the one that is installed. Start Mocktail from your application menu
or with `mocktail`. The desktop entry also opens Roblox's `roblox:` and
`roblox-player:` website links.

### Build the package yourself

- `packaging/plus/PKGBUILD` builds the tagged release: run `makepkg` in
  `packaging/plus`.
- `packaging/plus/build-release.sh --out DIR` builds the release package
  the way it is published, for the x86-64-v3 level, into `DIR/pkg`. Run it
  with `--help` for its options.

### Going back to upstream Mocktail

Plus and upstream's native packages and AppImage use the same settings and
data directories (upstream's Flatpak keeps its own under
`~/.var/app/space.bigrat.mocktail`). Upstream
Mocktail ignores the new `config.yaml` sections, so the file keeps working.
The settings window moves the saved Roblox session into its account store
(`~/.local/share/mocktail/auth/accounts/`), which upstream does not read, so
after switching back you may have to sign in again.

### Upstream Mocktail

Upstream Mocktail has its own install channels: Flathub, a nightly Flatpak,
the AUR, DNF and APT repositories, and AppImage, DEB and RPM downloads. They
install upstream Mocktail without the Plus changes; its AUR packages conflict
with `mocktail-plus`. See the
[upstream README](https://github.com/komaruworld/mocktail#readme) and
[mocktail.bigrat.space](https://mocktail.bigrat.space/) for the current
instructions.

<details>
<summary>Upstream's install commands (upstream Mocktail, not Plus)</summary>

Flathub, or the nightly Flatpak built from upstream's `main`:

```bash
flatpak install flathub space.bigrat.mocktail
flatpak install --user https://mocktail.bigrat.space/mocktail.flatpakref
```

AUR (`mocktail`, `mocktail-bin` or `mocktail-git`):

```bash
paru -S mocktail-git
# or
yay -S mocktail-git
```

Fedora 44:

```bash
sudo curl -fsSL https://mocktail.bigrat.space/rpm/mocktail.repo \
  -o /etc/yum.repos.d/mocktail.repo
sudo dnf install mocktail
# or
sudo dnf install mocktail-nightly
```

Ubuntu 26.04:

```bash
sudo install -d -m 0755 /etc/apt/keyrings
sudo curl -fsSL https://mocktail.bigrat.space/mocktail-packages.gpg \
  -o /etc/apt/keyrings/mocktail.gpg
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/mocktail.gpg] https://mocktail.bigrat.space/apt mocktail main" | \
  sudo tee /etc/apt/sources.list.d/mocktail.list >/dev/null
sudo apt update
sudo apt install mocktail
# or
sudo apt install mocktail-nightly
```

</details>

## The settings window

**When it opens.** On a normal start, from the application menu or with
`mocktail`, the settings window opens first. Press **Play** to start Roblox;
closing the window quits without starting it. The window is skipped:

- for website joins (`roblox:` and `roblox-player:` links, `--launch-uri`),
  which start with the account selected in the window;
- with `mocktail --play` (or `--no-launcher`), or the desktop entry's
  **Play now** action;
- when **Show this window on start** (Advanced page, `launcher.show_on_start`)
  is off. `mocktail --launcher` or the desktop entry's **Mocktail Settings**
  action then opens it anyway;
- for headless runs, without a display, for the update checks' test runs,
  and for the restart that follows a website sign-in.

If the window cannot start or crashes, Roblox starts as if Play was pressed.

**Pages.**

| Page | Settings |
|---|---|
| Graphics | Graphics backend, Graphics card (`engine.gpu`), Graphics quality (`engine.graphics_quality`), Compressed textures (BC) (`engine.etc2_to_bc`), Multithreaded shader loading (`engine.nvidia_shader_mt`, under "More renderer options"), Frame rate limit, Vertical sync |
| Display | Window size, Start in (`display.start_mode`), Native resolution on scaled displays, Display server (`display.server`), Theme, Window title |
| Performance | Multithreaded rendering, Physics workers, GameMode, Memory limit |
| Audio | Output device, Microphone |
| Accounts | Saved accounts, Play as guest, Sign in on the Roblox website, Sign in inside Roblox, Sign-in method (`account.sign_in`) |
| Integrations | Discord Rich Presence, Fleasion |
| Network & Updates | Update Roblox automatically, Download source, Proxy, Custom CA bundle |
| Advanced | Device profile, What Mocktail decides and what Roblox decides, Fast Flags, Show this window on start, Environment variables, Files |
| About | Mocktail and Roblox versions, this computer, logs and diagnostic info |

**Hints.** Under its title, each row says what the current value does on
this computer and, when the value is not the recommended one, what is
recommended here. The info button at the end of the row (a light bulb in the
default icon theme) opens a longer explanation. **Ctrl+F** searches titles,
subtitles, `config.yaml` keys and variable names.

**Saving.** Changes stay in the window until you press **Save** (Ctrl+S).
**Play** (Ctrl+Enter) saves first. Closing the window with unsaved changes
asks whether to save them; Ctrl+Q closes it without playing. If `config.yaml`
has an error, the window says so (with the line, when the loader names one),
keeps the settings read-only and offers to open the file or reload it, to
restore the copy it kept before its first save, if there is one, and, for an
empty file, to use the defaults; Roblox cannot start until the file loads.

**Accounts.** Several Roblox accounts can be saved. Play and website joins
start with the selected one, and switching needs no new sign-in. The account
button at the bottom left switches accounts, starts as a guest, or adds and
manages accounts. Sessions are kept in files only your user can read and are
never shown in the window, in logs or on the command line.

**ENV badges.** Environment variables still win over `config.yaml` (see
[Configuration](#configuration)). When a variable from a desktop shortcut or
a terminal overrides a setting, its row shows an **ENV** badge and the value
this launch uses, and a banner says how many settings are overridden.
**Details** lists the variables and where they come from. **Move into
Settings** copies their values into the settings and leaves the variables
out of this launch. **Clean Up Shortcut** removes them from your own copy of
the desktop entry in `~/.local/share/applications`, or removes the copy when
it only added them, and keeps a backup next to it.

**Overrides Roblox setting.** Some settings replace one of Roblox's own
in-game settings, for example a graphics quality level that you force in
the settings window. Such rows carry an **Overrides Roblox setting**
badge, and the Advanced page lists them under **What Mocktail decides and
what Roblox decides**, together with the settings left to Roblox and the
client settings Mocktail sets at every start.

The screenshot at the top was taken with 1.0.4-plus.1 during the window's
self-test (`mocktail_launcher_ui --selftest`) in a nested Hyprland session,
with a fresh `config.yaml` and a scratch home directory; the Accounts page
has not changed since. The self-test sets `MOCKTAIL_GRAPHICS_BACKEND` to
show the ENV badge and the banner.

## Configuration

Mocktail reads `$XDG_CONFIG_HOME/mocktail/config.yaml` (usually
`~/.config/mocktail/config.yaml`; `MOCKTAIL_CONFIG_ROOT` replaces the
directory). It is created on the first start, with mode 0600 and a comment
for every setting. The settings window edits the same file, and you can
still edit it by hand; the window notices when the file changes while it is
open. [config/mocktail.example.yaml](config/mocktail.example.yaml) shows the
whole file.

Environment variables still override `config.yaml`. Plus adds these keys, in
new top-level sections; for each of them a variable that is set, even to an
empty value, hides the value from the file (empty means the default):

| Key | Values (default first) | Variable | What it does |
|---|---|---|---|
| `display.server` | `auto`, `wayland`, `x11` | `MOCKTAIL_DISPLAY_SERVER` | Display server for the game window. `auto` prefers Wayland, with the [NVIDIA rule](#nvidia). A server the session does not offer falls back to `auto`. `SDL_VIDEODRIVER`, `SDL_VIDEO_DRIVER` or a `MOCKTAIL_FORCE_*` switch in your environment takes precedence. |
| `display.start_mode` | `remember`, `windowed`, `maximized`, `fullscreen` | `MOCKTAIL_WINDOW_START_MODE` | Window state at start. `remember` restores the mode the last session ended in. |
| `account.sign_in` | `native`, `browser` | `MOCKTAIL_NATIVE_LOGIN` (`0` means `browser`, anything else `native`) | Where you sign in when Roblox starts without a working saved session: Roblox's own welcome screen, or Mocktail's roblox.com window. |
| `engine.graphics_quality` | `default`, `manual`, `1` to `21` | `MOCKTAIL_GRAPHICS_QUALITY` (also `auto` or `0` for `manual`) | Roblox's graphics quality level. `default` and `manual` leave it to Roblox's own Graphics Quality setting; a number forces that level (`DFIntDebugFRMQualityLevelOverride`), whatever the performance settings, and Roblox's in-game slider then has no effect. Before 1.0.4-plus.2 the level was sent under a name Roblox 2.738 ignores, so it never took effect. |
| `engine.gpu` | `auto`, `discrete`, `integrated` | `MOCKTAIL_GPU` | Graphics card for direct Vulkan on computers with more than one. See [Hybrid laptops](#hybrid-laptops). |
| `engine.nvidia_shader_mt` | `true`, `false` | `MOCKTAIL_NVIDIA_SHADER_MT` (`1`/`0`, `true`/`false`, `on`/`off`) | With direct Vulkan, lets Roblox load its shader pack on several threads on NVIDIA cards. `false` makes it use one thread there, as upstream does. |
| `engine.etc2_to_bc` | `true`, `false` | `MOCKTAIL_ETC2_TO_BC` (`1`/`0`, `true`/`false`, `on`/`off`) | Compressed textures (BC). With direct Vulkan on a card that cannot read ETC2 (NVIDIA), Roblox's ETC2 textures are converted to BC and stay compressed. `false` unpacks them to full size, as before 1.0.4-plus.2. A card that reads ETC2 itself is not affected. See [ETC2 textures](#etc2-textures). |
| `launcher.show_on_start` | `true`, `false` | `MOCKTAIL_LAUNCHER_SHOW_ON_START` (`1`/`0`, `true`/`false`, `on`/`off`) | Show the settings window before Roblox starts. |

A `config.yaml` from upstream Mocktail works unchanged: a missing section
means the defaults, and the settings window adds the section, with its
comments, the first time you change one of its settings.

## Signing in

The Accounts page offers three ways in:

- **Sign in on the Roblox website** opens roblox.com's sign-in page in a
  Mocktail window and saves the account.
- **Sign in inside Roblox** starts Roblox signed out, so you can use its own
  welcome screen, for example Quick Log In with a device where you are
  already signed in. The account is added to the saved accounts the next
  time the window opens.
- **Sign-in method** (`account.sign_in`) decides what happens when Roblox
  starts without a working session: Roblox's welcome screen (`native`, the
  default) or Mocktail's roblox.com window first (`browser`).

Signing in with a user name and password on Roblox's own screen may make
Roblox ask for a Google Play Integrity device check, which nothing on Linux
can pass. Plus answers Roblox's integrity questions as "unavailable", and
when Roblox asks for such a check anyway, Plus opens the roblox.com sign-in
page in a "Sign in to Roblox" window instead. Once you have signed in there,
Mocktail restarts by itself and Roblox starts signed in. The settings window
is skipped for that restart.

## Buying Robux

Mocktail has no Google Play Billing. Plus hands purchases to your web
browser:

- On Roblox's Robux page, choosing a package opens
  `https://www.roblox.com/upgrades/robux` in your default browser, and the
  page says "Opened the Robux page in your web browser. Finish your purchase
  there."
- A purchase prompt inside an experience opens the same page, shows a
  desktop notification, and tells the game that nothing was bought. After
  buying Robux in the browser, try the purchase in the experience again.

The browser uses its own roblox.com sign-in; Mocktail passes no session to
it, so sign in there with the same account.

## NVIDIA

### Display server

On a Wayland session that also has XWayland, upstream Mocktail always runs
NVIDIA cards with direct Vulkan through XWayland. Plus uses native Wayland
there when `display.server` is `auto` and all of these hold, checked in this
order:

1. `MOCKTAIL_PREFER_WAYLAND` is not `0` or `false`, and the surface-commit
   guard is on (it is unless `MOCKTAIL_WAYLAND_COMMIT_GUARD` is `0`, `false`,
   `off` or `no`). The guard keeps SDL's own surface commits from landing in
   the middle of NVIDIA's explicit-sync present, which would be a fatal
   Wayland protocol error.
2. The NVIDIA driver is version 555 or newer (explicit sync in its Vulkan
   driver), and `__NV_DISABLE_EXPLICIT_SYNC` is unset or `0`.
3. There is no Intel or AMD graphics card beside the NVIDIA one (presenting
   through PRIME is untested).
4. The compositor offers explicit sync (`wp_linux_drm_syncobj_manager_v1`).
5. The compositor is Hyprland, or the game does not wait for the display:
   `graphics.vsync: off`, or `auto` with `frame_rate_limit: unlimited`. With
   vertical sync, NVIDIA's Wayland driver can stall the game's rendering
   while its window is hidden (seen in GNOME).

Otherwise the game uses XWayland. A session without XWayland always uses
Wayland. The Display page's **Display server** row says what Automatic picks
on your computer and why, and so does the session log, for example:

```
[window] NVIDIA 615.71.09 direct Vulkan on Wayland: the compositor offers explicit sync (wp_linux_drm_syncobj_manager_v1) and is Hyprland; using the native Wayland WSI with the surface-commit guard; set display.server: x11 or SDL_VIDEODRIVER=x11 to override
```

That was the choice on an RTX 3060 Laptop GPU with driver 615.71.09 on
Hyprland. `display.server: x11` or `wayland` overrides the rule.

### Multithreaded shader loading

With direct Vulkan, upstream Mocktail makes Roblox load its shader pack on a
single thread on every NVIDIA card. Plus leaves that to Roblox's own
setting, which loads it on several threads, as on Intel and AMD. If shader loading fails on your
NVIDIA card, set `engine.nvidia_shader_mt: false` (Graphics page, More
renderer options, **Multithreaded shader loading**) or
`MOCKTAIL_NVIDIA_SHADER_MT=0`.

### ETC2 textures

Roblox's Android client sends most of its textures in the ETC2 format (EAC
for one- and two-channel ones). NVIDIA's Vulkan driver cannot read ETC2, so
Mocktail tells Roblox that ETC2 is supported anyway and converts these
textures as Roblox uploads them. A card whose driver reads ETC2 itself is
left alone. The session log says what happens on yours, for example:

```
[vulkan] ETC2/EAC emulation enabled: compressed uploads become BC1/BC3/BC4/BC5 host images (GPU transcode)
[vulkan] ETC2 GPU decode ready on family 0 (build 20 ms, selftest ok, BC transcode too)
```

Since 1.0.4-plus.2 the conversion works like this:

- **On the graphics card.** A compute shader converts the uploads in the
  same submission as Roblox's own copies, right before them. When a session
  starts, Mocktail runs it on test blocks of every ETC2 and EAC format and
  uses it only if the result matches the processor's byte for byte
  (`selftest ok` above).
- **Compressed textures (BC)**, on by default (`engine.etc2_to_bc`, the
  Graphics page's **Compressed textures (BC)** row): the textures become
  BC1, BC3, BC4 or BC5, which NVIDIA cards read natively, and stay
  compressed, so they take as much video memory as Roblox counts for them.
  Unpacked (RGBA8, and 16-bit formats for EAC), as earlier versions did and
  as the setting's off position still does, they take 4 to 8 times as
  much.
- **On the processor** whenever the graphics card cannot take an upload:
  with `MOCKTAIL_ETC2_GPU=0`, when the check fails, until the shader is
  ready at the start of a session, for the small textures that are
  enlarged without BC (see below), and in a few rarer cases that the
  `cpu_route` counts of the session totals name. This is the same
  conversion as before, rewritten: it reads Roblox's upload memory with
  streaming loads, runs on a pool of worker threads with a faster decoder,
  and reuses its staging memory instead of creating a buffer for every
  copy.

**Why Mocktail stuttered.** Roblox streams textures while you move and turn
the camera, and uploads many of them again and again. In one four-minute
session in Anime Vanguards it uploaded 33,162 textures, 1.8 GiB compressed
and 12.6 GiB unpacked. 1.0.4-plus.1 converted every upload on the processor
inside Roblox's `vkQueueSubmit`, that is on the game's render thread, and
created and destroyed a staging buffer for every texture copy. That session
spent 23.4 s converting and another 25.0 s creating and destroying those
buffers; single frames took up to 0.4 s while moving, and one conversion
held the game for 0.7 s while the place loaded. The new processor path
converted about as many uploads (33,181, 2.4 GiB compressed) in 3.2 s in
all, with no conversion longer than 69 ms. With BC on the graphics card, the
processor converted 18 of 24,040 uploads, and the graphics card spent
146 ms of its own time on the rest in the whole session.

The test: Anime Vanguards, Roblox 2.738.1397, an RTX 3060 Laptop GPU (6 GB)
with driver 615.71.09 on Hyprland, two runs per row on a random server
each, turning the camera for about 30 s and then moving for about 40 s.
"Stalls" adds up the conversions that held the render thread for 20 ms or
more.

| | Stalls, turning | Stalls, moving | Longest frame, moving | Frames over 100 ms, moving | Video memory of the game, average / peak |
|---|---|---|---|---|---|
| 1.0.4-plus.1's processor path | 0.35–2.0 s | 2.3–4.0 s | 366–384 ms | 24–39 | 1580–1680 / 2190–2210 MiB |
| New processor path, BC off | 0.02–0.08 s | 0.15–0.21 s | 101–118 ms | 1–2 | 1570–1750 / 2130–2160 MiB |
| Graphics card, BC off | none | none | 44–101 ms | 0–1 | 1800–1930 / 2410–2420 MiB |
| Graphics card, BC on (the default) | none | none | 62–64 ms | none | 770–840 / 910–1010 MiB |
| Processor, BC on (one run) | 0.12 s | none | 46 ms | none | 742 / 845 MiB |

The average frame rate while moving was 76–106 fps in every row and
depended more on the server than on the mode.

**Compressed textures (BC).** In the test the game needed less than half
the video memory with it, and converting on the graphics card caused no
stall. Roblox also uploaded less: 514–655 MiB of compressed textures per
run against 1.0–3.5 GiB with BC off. The cost is some quality in fine
detail, because converting one compressed format into another loses a
little: sharp edges in text and icons or smooth gradients can look
blockier. In screenshots from the test the interface icons looked the
same, and small avatar textures looked smoother with BC (see the next
paragraph). When the processor has to convert to BC (`MOCKTAIL_ETC2_GPU=0`,
or a session log without the `ETC2 GPU decode ready` line), loading can
stutter briefly: in the test the longest conversion took 196 ms while the
place loaded, and turning the camera brought three stalls of at most
62 ms. If textures look wrong to you, turn the setting off in the settings
window, with `engine.etc2_to_bc: false` or with `MOCKTAIL_ETC2_TO_BC=0`.

**Small textures are enlarged fourfold without BC.** With BC off, Mocktail
enlarges colour textures of up to 64×64 pixels four times in each
direction by repeating pixels (`MOCKTAIL_SMALL_TEXTURE_UPSCALE`, default
4), as earlier versions did, and converts them on the processor. Turning
that off (`MOCKTAIL_SMALL_TEXTURE_UPSCALE=1`) saves some video memory, but
in the same test Roblox then uploaded 132,893 textures, 15.5 GiB
compressed, in about four minutes, and the game ran at 41 fps while moving
instead of 80–90. So the enlargement stays. BC keeps small textures at
their own size, and Roblox's uploads stayed low there.

**Why ETC2 stays advertised.** `MOCKTAIL_ADVERTISE_ETC2=0` tells Roblox
that the card has no ETC2. Roblox then takes a much slower path of its own:

| Same place and server | ETC2 (1.0.4-plus.1) | `MOCKTAIL_ADVERTISE_ETC2=0` |
|---|---|---|
| Main menu | 113–123 fps | 53–56 fps |
| In game | 158–164 fps | about 62 fps |
| CPU use | about 2.3 cores | about 3.7 cores |
| Video memory of the game | about 1165 MiB | about 660–700 MiB |

That test ran Roblox 2.738.1397 in Greenville RP on the same RTX 3060 Laptop
GPU, on the same server for each pair of runs; with Mocktail's performance
preset off it was 123 fps with ETC2 against 38 fps without. While working on
1.0.4-plus.2 we counted the game's Vulkan calls (`MOCKTAIL_VK_CALL_STATS=1`)
in Anime Vanguards while the character stood still for 20 seconds. Without
ETC2, Roblox created about 156,000 textures in that time (about 7,800 a
second), all but one of them uncompressed RGBA8 rather than BC, made about
928,000 image copies and created about 929,000 buffers, and the game ran at
33 fps. With ETC2 it created 88 textures and made 8,966 image copies at 105
fps. Roblox's render thread was 85–87 % busy with work spread over
`libroblox` itself, and Mocktail's Vulkan code took about 1.4 % of the
profile. Three suspects on Mocktail's side were ruled out:

- The format answers: until 1.0.4-plus.2, Mocktail still reported the ETC2
  formats as usable while `MOCKTAIL_ADVERTISE_ETC2=0` hid ETC2 support. Now
  both say no, and it was still 33 fps.
- The performance preset's texture-compression flags: leaving them out
  (`MOCKTAIL_PRESET_TEXTURE_COMPRESSION=off`) gave 37 fps. Three of the
  five were among the settings Roblox never read (see
  [What Plus changes](#what-plus-changes)).
- The texture budget: leaving it to Roblox (`MOCKTAIL_TEXTURE_BUDGET_MB=roblox`)
  gave 40 fps.

The slow path is inside Roblox, so Mocktail does not try to change it.
Compressed textures (BC) gives the video memory savings of that path while
Roblox keeps its fast ETC2 one.

**Texture budget.** The performance preset caps Roblox's texture budget at
128 MB (`FIntRenderTextureTotalBudgetMB`). A tight budget makes Roblox
upload textures again when the camera turns: with 1.0.4-plus.1's processor
path, leaving the budget to Roblox cut the stalls while turning from 7.3 s
to 0.4 s in one pair of runs (and while moving from 7.1 s to 3.8 s).
1.0.4-plus.2 keeps 128 MB, because the graphics card now absorbs those
uploads without stalls and BC reduces them. `MOCKTAIL_TEXTURE_BUDGET_MB`
sets another budget for your own tests; `fflags.json` cannot, because
Roblox does not start when it sets a preset setting to another value.

**Variables.** `engine.etc2_to_bc` is the only one of these in
`config.yaml` and the settings window; the others are for tests and
troubleshooting.

| Variable | Values (default first) | What it does |
|---|---|---|
| `MOCKTAIL_ETC2_TO_BC` | `1`, `0` (also `true`/`false`, `on`/`off`) | Compressed textures (BC), the variable for `engine.etc2_to_bc`. |
| `MOCKTAIL_ETC2_GPU` | `1`, `0` (also `true`/`false`, `on`/`off`) | `0` keeps the conversion on the processor, for comparisons or if the graphics card path misbehaves. |
| `MOCKTAIL_ADVERTISE_ETC2` | `1`, `0` | `0` tells Roblox the card has no ETC2 (much slower, see above). |
| `MOCKTAIL_DISABLE_ETC2_EMULATION` | unset, `1` | Turns the ETC2 support off completely, as in Mocktail 1.0.4: Roblox sees a card without ETC2. |
| `MOCKTAIL_SMALL_TEXTURE_UPSCALE` | `4`, `1` to `8` | How many times colour textures of up to 64×64 pixels are enlarged in each direction with BC off; `1` turns it off. BC mode never enlarges. |
| `MOCKTAIL_TEXTURE_BUDGET_MB` | unset (128), `16` to `4095`, `roblox` | The texture budget while the performance preset is on; `roblox` leaves the budget to Roblox. Any other value stops the start with an error that names the variable. |
| `MOCKTAIL_PRESET_TEXTURE_COMPRESSION` | `on`, `off` | `off` leaves the preset's texture-compression flags (`FFlagDebugTerrainVTCompressedTextures`, `FFlagVulkanCompressedTextureAliasing`) out. |
| `MOCKTAIL_FRAME_STATS` | unset, `1` | Frame pacing and texture work in the session log; see [Measuring stutter](#measuring-stutter). |
| `MOCKTAIL_VK_CALL_STATS` | unset, `1` | Counts the game's Vulkan calls; see [Measuring stutter](#measuring-stutter). |
| `MOCKTAIL_ETC2_WORKERS` | half the CPU threads, at most 8; `1` to `64` | Worker threads of the processor path. |
| `MOCKTAIL_ETC2_STAGING_MEMORY` | `cached`, `coherent` | Memory for the processor path's staging: cached system memory when the driver offers it, or uncached write-combined memory, the kind Roblox uploads from on the test card. |
| `MOCKTAIL_ETC2_GPU_ARENA_MB` | automatic, a number of MiB | Video memory the graphics card path may hold for converted textures before it uses system memory. Automatic is 256 MiB, or 1/24 of the card's video memory if that is less. |
| `MOCKTAIL_ETC2_TRACE` | unset, a file | Writes one CSV line per converted upload. |
| `MOCKTAIL_ETC2_TRACE_BLOCKS` | unset, a directory | Saves up to 64 MiB of Roblox's compressed uploads per session, for quality tests. |

## Hybrid laptops

`engine.gpu` (Graphics page, **Graphics card**) picks the card that direct
Vulkan renders on when the computer has more than one:

- `auto` prefers the discrete card, unless `DRI_PRIME` or
  `__NV_PRIME_RENDER_OFFLOAD` is `0`, `off` or `igpu`;
- `discrete` and `integrated` pick that kind of card; when no such card has
  a Vulkan driver, another card stands in;
- a computer with one card always uses it, and a driver list you set in
  `VK_DRIVER_FILES` or `VK_ICD_FILENAMES` wins over all of this.

Plus finds the cards under `/sys/class/drm`, tells integrated from discrete
by their PCI position, and pins the chosen card's Vulkan driver. The session
log names the card in a line like
`[runtime] vulkan GPU=NVIDIA 10de:2520 discrete at 0000:01:00.0 ICD=...`.
On a laptop with Intel or AMD graphics beside an NVIDIA card, the
[display server rule](#display-server) keeps the game on XWayland while it
renders on the NVIDIA card; on the integrated card it prefers Wayland as
usual.

## FFlag overrides

Put a JSON object in `$XDG_CONFIG_HOME/mocktail/fflags.json` (usually
`~/.config/mocktail/fflags.json`), or use the Fast Flags editor on the
Advanced page. Mocktail applies it on the next launch.

PC profiles use the legacy Charts page by default to avoid the blank screen
caused by `Color3` errors in the newer SDUI page. An explicit
`FFlagLuaAppChartsAppPage` override takes precedence over this default.

```json
{
  "FFlagExample": "True",
  "DFIntExample": "120"
}
```

## Troubleshooting

- Logs: `~/.local/state/mocktail/logs/latest.log` (a custom
  `$XDG_STATE_HOME` replaces `~/.local/state`). The settings window opens the
  folder from its main menu and from the About page, which can also copy
  diagnostic info. Attach `latest.log` to a bug report.
- Mocktail does not currently work with `hardened_malloc`, which may make it
  fail to start or crash. Run Mocktail without `hardened_malloc`.
- Stutter: see [Measuring stutter](#measuring-stutter) below, and attach
  the summary to the report.
- More answers are in the [FAQ](FAQ.md).

### Measuring stutter

Start a session with `MOCKTAIL_FRAME_STATS=1`, for example
`MOCKTAIL_FRAME_STATS=1 mocktail --play`. The session log then gets these
lines:

- Every 2 seconds, how evenly frames arrived. The average frame rate hides
  the long frames you feel as stutter: `max` is the longest frame, and
  `over33`, `over50` and `over100` count the frames longer than 33, 50 and
  100 ms.

  ```
  [perf] frame_stats frames=196 fps=97.7 median=6.1ms p99=122.3ms max=329.1ms over33=6 over50=5 over100=2
  ```

- Each time the processor path of the [ETC2 conversion](#etc2-textures)
  holds the render thread for 20 ms or more, with its phases (preparing,
  reading Roblox's upload memory, converting, writing the result), its
  threads and the memory types it read from:

  ```
  [vulkan] ETC2 decode stall 68 ms: 4051 uploads, 50.6 MiB compressed, 367.2 MiB written (prepare 0, copy 11, decode 53, write 3 ms) [threads 8, src t3 50.6 MiB]
  ```

- Each graphics card batch whose recording took 1 ms or more, or whose
  work on the graphics card took 4 ms or more. It does not hold the render
  thread for the `gpu` time:

  ```
  [vulkan] ETC2 GPU submit: 2232 uploads, 48.2 MiB compressed, 289.5 MiB written, record 3.72 ms, gpu 7.66 ms (copy-in 3.85, decode 3.82; lag 1)
  ```

- Every 2 seconds with ETC2 texture activity, a `[perf] etc2_window` line
  with that window's counters: processor and graphics card time
  (`cpu_ms`, `gpu_ms`), uploads by path (`gpu=`, `fallback=` for the
  processor), staging allocations, texture images created and destroyed,
  and more.
- The session totals, which Mocktail logs even without
  `MOCKTAIL_FRAME_STATS` when the game closes its Vulkan device, and now
  and then while the processor path converts. With the graphics card path,
  or with `MOCKTAIL_FRAME_STATS`, details follow after the `|`, among them
  why uploads took the processor path (`cpu_route`):

  ```
  [vulkan] ETC2 decode totals: images=6187 (upscaled 0) uploads=18 in 1 submits compressed=0.0 MiB host=0.0 MiB (1.0x) decode=0 ms longest=0 ms | gpu uploads=24022 submits=969 compressed=655.3 MiB ... | cpu_route gpu_disabled=0 gpu_not_ready=18 ...
  ```

`scripts/frame_stats_summary.py` from the source repository adds these
lines up: frame rates, the longest frames, the stalls and their phases, the
graphics card batches, the windows' counters and the totals. It prints only
numbers and Mocktail's own lines, never other log text, so its output can
go into a bug report. To compare parts of a session, such as standing still
and running, note the log's line count
(`wc -l ~/.local/state/mocktail/logs/latest.log`) when each part starts and
ends, and name the parts:

```bash
python3 scripts/frame_stats_summary.py ~/.local/state/mocktail/logs/latest.log \
  --phase idle=410:530 --phase move=530:1700
```

To see what the conversion paths do on your computer, play the same place
with `MOCKTAIL_ETC2_GPU=0` (processor only) or `MOCKTAIL_ETC2_TO_BC=0`
(unpacked textures) and compare the summaries.

`MOCKTAIL_VK_CALL_STATS=1` adds a `[perf] vk_calls` line every 2 seconds:
the game's Vulkan calls per frame (draws, pipeline and descriptor binds,
copies), the window's totals (image copies and blits, buffers and images
created, the images by format: ETC2, BC, RGBA8 and others) and the time the
game spent waiting on fences, semaphores, image acquisition, presentation,
submission and idle waits. Without the variable Mocktail does not wrap
these calls at all.

```
[perf] vk_calls frames=193 span_ms=2006 per_frame: submit=1.0 draw=12.9 drawIdx=10.8 ... | window: copyImg=210 blit=168 ... createImg=381 (etc2 3 bc1-3 14 bc4-6 0 bc7 0 astc 0 rgba8 291 other 73) ... | blocked fence=... sem=... acquire=... present=... submit=... idle=... ms
```

## Building from source

Linux `x86_64` is supported. Upstream also builds for Linux `aarch64` (with
Roblox runtime validation still experimental) and runs in a Linux userspace
under FreeBSD 15.1's Linuxulator, tested with an `x86_64` Fedora 44
userspace; see the [FreeBSD Guide](packaging/freebsd/README.md). The Plus
changes have been built and tested only on `x86_64` Linux.

Building requires CMake 3.20+, Git, pkg-config, LLD, binutils, a C++17
compiler, SDL 3.4+, SDL3_ttf, Vulkan, EGL, libplacebo, fontconfig, libcurl,
OpenSSL, libelf, libyaml, minizip, Capstone 5, utf8proc, nlohmann/json, GTK4,
libadwaita 1.6+ and WebKitGTK 6.0. The settings window also needs
libadwaita 1.9+ and `msgfmt` from gettext. Without them, configure with
`-DMOCKTAIL_BUILD_LAUNCHER_UI=OFF`; Mocktail then starts Roblox directly.

<details>
<summary>Ubuntu 26.04+</summary>

```bash
sudo apt update
sudo apt install build-essential cmake git ninja-build pkg-config lld \
  libsdl3-dev libsdl3-ttf-dev libcurl4-openssl-dev libssl-dev \
  nlohmann-json3-dev libyaml-dev libelf-dev libminizip-dev \
  libcapstone-dev libgtk-4-dev libadwaita-1-dev libwebkitgtk-6.0-dev \
  libutf8proc-dev libfontconfig1-dev libegl-dev libvulkan-dev \
  libplacebo-dev libpng-dev zlib1g-dev gettext
```
</details>

<details>
<summary>Arch Linux</summary>

```bash
sudo pacman -S --needed base-devel cmake git ninja pkgconf lld sdl3 sdl3_ttf \
  curl openssl nlohmann-json libyaml libelf minizip capstone gtk4 \
  libadwaita webkitgtk-6.0 libutf8proc fontconfig libglvnd \
  libplacebo vulkan-headers vulkan-icd-loader zlib gettext
```
</details>

<details>
<summary>Fedora 44+</summary>

```bash
sudo dnf install gcc-c++ cmake git ninja-build pkgconf-pkg-config lld \
  SDL3-devel SDL3_ttf-devel libcurl-devel openssl-devel \
  nlohmann-json-devel libyaml-devel elfutils-libelf-devel minizip-ng-compat-devel \
  capstone-devel gtk4-devel libadwaita-devel webkitgtk6.0-devel \
  utf8proc-devel fontconfig-devel libglvnd-devel vulkan-headers \
  vulkan-loader-devel libplacebo-devel zlib-ng-compat-devel gettext
```
</details>

```bash
git clone --recurse-submodules https://github.com/blackixxce12/mocktail.git
cd mocktail
make build
./build/mocktail
```

## Development

- `main` is upstream Mocktail's `main` at `273209b` with the Plus commits on
  top; `git log 273209b..main` lists them. The files for upstream's install
  channels (Flatpak, AUR, Arch, DEB, RPM, AppImage) and its CI workflows stay
  in the tree for upstream; Plus release packages are built with
  `packaging/plus`.
- `make test` builds and runs the unit tests. The settings window has a
  self-test, `mocktail_launcher_ui --selftest OUT_DIR`, that opens the real
  window on a scratch configuration, goes through every page, saves and
  checks the result, renders each page to a PNG and writes `report.json`.
- Most Plus changes were written with
  [Claude Code](https://claude.com/claude-code) (AI-assisted). Every such
  commit carries the trailer
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## License and credits

[Apache License 2.0](LICENSE). Third-party components keep their own
licenses.

Mocktail is by komaruworld and its contributors; Mocktail Plus only adds to
their work. Upstream's code, website and community are at
[github.com/komaruworld/mocktail](https://github.com/komaruworld/mocktail),
[mocktail.bigrat.space](https://mocktail.bigrat.space/) and its
[Discord server](https://discord.gg/rhgpfcmFSD). To support upstream's
authors, see the Support section of the
[upstream README](https://github.com/komaruworld/mocktail#support).

<details>
<summary>Roblox in Mocktail (screenshots from upstream)</summary>

![Roblox home in Mocktail](assets/screenshots/flatpak-home.png)

![Roblox gameplay in Mocktail](assets/screenshots/flatpak-gameplay-tower.png)

![Roblox experience in Mocktail](assets/screenshots/flatpak-gameplay-lobby.png)

</details>
