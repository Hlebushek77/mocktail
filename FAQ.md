# FAQ

Answers for Mocktail Plus. Paths are those of the Plus package; a custom
`$XDG_CONFIG_HOME` or `$XDG_STATE_HOME` replaces `~/.config` or
`~/.local/state`.

- [How do I enable Discord RPC?](#how-do-i-enable-discord-rpc)
- [Where can I find logs for a bug report?](#where-can-i-find-logs-for-a-bug-report)
- [How do I start Roblox without the settings window?](#how-do-i-start-roblox-without-the-settings-window)
- [Signing in ends with "Load generic challenge failed". What now?](#signing-in-ends-with-load-generic-challenge-failed-what-now)
- [How do I buy Robux?](#how-do-i-buy-robux)
- [Chat text is invisible while I type](#chat-text-is-invisible-while-i-type)
- [Why does the game use XWayland, or native Wayland, on my NVIDIA card?](#why-does-the-game-use-xwayland-or-native-wayland-on-my-nvidia-card)
- [The game stutters when I move or turn the camera](#the-game-stutters-when-i-move-or-turn-the-camera)
- [What does Compressed textures (BC) do, and when should I turn it off?](#what-does-compressed-textures-bc-do-and-when-should-i-turn-it-off)
- [Why is ETC2 kept, and what does MOCKTAIL_ADVERTISE_ETC2=0 do?](#why-is-etc2-kept-and-what-does-mocktail_advertise_etc20-do)
- [Which of my settings override Roblox's own in-game settings?](#which-of-my-settings-override-robloxs-own-in-game-settings)
- [Can I run multiple Roblox instances?](#can-i-run-multiple-roblox-instances)
- [Can I play in VR?](#can-i-play-in-vr)

## How do I enable Discord RPC?

RPC is disabled by default. In the settings window, open **Integrations**,
turn on **Discord Rich Presence** and press **Save**.

Or edit `~/.config/mocktail/config.yaml` and set `enabled` to `true` under
`integrations.discord_rpc`. If the block is missing, add it under the
existing `integrations` section:

```yaml
integrations:
  discord_rpc:
    enabled: true
```

Open Discord Desktop and start Mocktail. Turn it off the same way.

## Where can I find logs for a bug report?

Mocktail saves logs automatically. Reproduce the issue, close Mocktail, and
attach `~/.local/state/mocktail/logs/latest.log`. The settings window opens
the logs folder from its main menu and from the **About** page, which can
also copy diagnostic info for the report.

## How do I start Roblox without the settings window?

Use the desktop entry's **Play now** action or `mocktail --play`, or turn
off **Show this window on start** on the **Advanced** page
(`launcher.show_on_start: false`). Website joins never show the window. To
open it again, use the **Mocktail Settings** action or `mocktail --launcher`.

## Signing in ends with "Load generic challenge failed". What now?

Roblox may answer a sign-in with a user name and password in its Android app
with a Google Play Integrity device check (challenge type `deviceintegrity`).
Nothing on Linux can pass it: Mocktail has no certified Android device to
show. In upstream Mocktail the check's page waits about a minute and then
shows "Load generic challenge failed".

Plus answers Roblox's integrity questions as "unavailable" (it makes up no
token), and when Roblox asks for the check anyway, it opens roblox.com's
sign-in page in a "Sign in to Roblox" window instead. Sign in there; Mocktail
then restarts by itself and Roblox starts signed in. The session log shows
such a check as `[webview] Roblox asked for deviceintegrity verification,
which Linux cannot provide; opening website sign-in instead`.

Other ways in, on the settings window's **Accounts** page:

- **Sign in on the Roblox website** signs in on roblox.com before Roblox
  starts and saves the account.
- **Sign in inside Roblox** starts Roblox signed out so you can use Quick
  Log In with a device where you are already signed in.
- **Sign-in method: Website window** (`account.sign_in: browser`) opens
  Mocktail's roblox.com window when Roblox starts without a working session
  (website joins excepted).

## How do I buy Robux?

In your web browser. Mocktail has no Google Play Billing, so it cannot buy
inside the app. Choosing a Robux package on Roblox's Robux page opens
`https://www.roblox.com/upgrades/robux` in your default browser, and the page
says "Opened the Robux page in your web browser. Finish your purchase
there." A purchase prompt inside an experience opens the same page, shows a
desktop notification and tells the game nothing was bought; after buying
Robux in the browser, try the purchase in the experience again.

The browser uses its own roblox.com sign-in, so sign in there with the same
account. If no browser can be opened, the page says "Couldn't open your web
browser. Buy Robux at roblox.com/upgrades/robux."

## Chat text is invisible while I type

Fixed. Text typed into text boxes such as the chat bar could stay
invisible in the Mocktail 1.0.4 release. Upstream fixed this on `main` after
1.0.4 (commits `0548ebd`, `1c11336` and `b37497d`), and Plus is built from
that `main`, so chat text shows while you type. Plus also runs each call into
Roblox's TextBox code in its own JNI local frame, as Android does, and polls
the text box position at most once every 50 ms while it has focus. If text
still does not show, attach `latest.log` to a bug report.

## Why does the game use XWayland, or native Wayland, on my NVIDIA card?

NVIDIA's Wayland presentation is reliable only with explicit sync. Upstream
Mocktail always runs NVIDIA cards with direct Vulkan through XWayland on a
Wayland session that has XWayland. Plus uses native Wayland automatically
when the driver is 555 or newer with explicit sync left on
(`__NV_DISABLE_EXPLICIT_SYNC` unset or `0`), there is no Intel or AMD card
beside the NVIDIA one, the compositor offers explicit sync, and either the
compositor is Hyprland or the game does not wait for the display (vertical
sync off, or Automatic with an unlimited frame rate). Otherwise it keeps
XWayland.
The full list is in the README's [NVIDIA](README.md#nvidia) section.

To see what was chosen and why, look at the Display page's **Display
server** row, or for a `[window] NVIDIA` line in `latest.log`. To choose
yourself, set **Display server** to Wayland or X11 (`display.server: wayland`
or `x11`).

## The game stutters when I move or turn the camera

On NVIDIA cards, Mocktail's texture conversion caused most of it until
1.0.4-plus.2. Roblox streams textures in the ETC2 format while you move,
NVIDIA cards cannot read ETC2, and Mocktail converted each upload on the
processor, on the game's render thread. In a test in Anime Vanguards on an
RTX 3060 Laptop GPU that added up to 2.3–4.0 s of stalls in 40 s of
walking, with single frames of up to 0.4 s.

Since 1.0.4-plus.2 the graphics card converts the textures, and in the same
test no conversion stalled the game any more; the longest frame while
walking was 44–101 ms. The processor path that remains as the fallback is
about seven times faster than before (0.15–0.21 s of stalls in the same
40 s). The README's [ETC2 textures](README.md#etc2-textures) section has
the numbers.

The session log shows which path converts:
`[vulkan] ETC2 GPU decode ready ...` means the graphics card does. If you
still see stutter, measure it as described under
[Measuring stutter](README.md#measuring-stutter) and attach the summary
and `latest.log` to a bug report. The summary also counts long frames that
no conversion stall explains (`unexplained long frames`); those have other
causes, such as a busy server.

## What does Compressed textures (BC) do, and when should I turn it off?

It is the Graphics page's **Compressed textures (BC)** row
(`engine.etc2_to_bc`, `MOCKTAIL_ETC2_TO_BC`), on by default since
1.0.4-plus.2. On a card that cannot read ETC2 (NVIDIA), Mocktail converts
Roblox's ETC2 textures to BC, a compressed format the card reads natively,
instead of unpacking them to full size. In a test on an RTX 3060 Laptop GPU
the game then used about 800 MiB of video memory on average instead of
about 1900 MiB (at most about 1000 instead of 2400 MiB), with no stalls.
On a card that reads ETC2 itself it changes nothing.

The cost is some quality in fine detail: sharp edges in text and icons or
smooth gradients can look blockier. Small textures (up to 64×64 pixels)
also stay at their own size instead of being enlarged fourfold, so they
look smoother but less crisp. If the graphics card cannot convert (the
session log has no `[vulkan] ETC2 GPU decode ready` line), the processor
does it, and loading can stutter briefly: up to about 0.2 s in the test.

Turn it off if textures look wrong to you. Mocktail then unpacks them, as
earlier versions did; the change applies the next time Roblox starts.

## Why is ETC2 kept, and what does MOCKTAIL_ADVERTISE_ETC2=0 do?

NVIDIA's Vulkan driver has no ETC2 textures, so Mocktail tells Roblox ETC2
is supported and converts those textures itself.
`MOCKTAIL_ADVERTISE_ETC2=0` stops advertising ETC2. That uses less video
memory, but Roblox then takes a much slower path of its own:

| Same place and server | ETC2 (1.0.4-plus.1) | `MOCKTAIL_ADVERTISE_ETC2=0` |
|---|---|---|
| Main menu | 113–123 fps | 53–56 fps |
| In game | 158–164 fps | about 62 fps |
| CPU use | about 2.3 cores | about 3.7 cores |
| Video memory of the game | about 1165 MiB | about 660–700 MiB |

(Roblox 2.738.1397, Greenville RP, RTX 3060 Laptop GPU, driver 615.71.09,
Hyprland.) The textures looked the same. While working on 1.0.4-plus.2 we
counted the game's Vulkan calls in another place while the character stood
still: in 20 seconds without ETC2, Roblox created about 156,000 uncompressed
RGBA8 textures, 928,000 image copies and 929,000 buffers and ran at 33 fps;
with ETC2 it created 88 textures and ran at 105 fps. Nothing on Mocktail's
side explained it: with consistent format answers it still ran at 33 fps,
without the performance preset's texture-compression flags at 37 and with
the texture budget left to Roblox at 40. The slow path is inside Roblox's
own code, so ETC2 stays advertised. To save video memory, keep
[Compressed textures (BC)](#what-does-compressed-textures-bc-do-and-when-should-i-turn-it-off)
on instead: it gives the low video memory while Roblox keeps its fast ETC2
path. The README's [ETC2 textures](README.md#etc2-textures) section has the
details.

## Which of my settings override Roblox's own in-game settings?

The settings window shows it. A row whose value replaces one of Roblox's own
settings has an **Overrides Roblox setting** badge, and its info button says
what is overridden and how to give it back. The **Advanced** page lists all
of them under **What Mocktail decides and what Roblox decides** (the
Performance page links there). Depending on your settings these are:

- Roblox's **Graphics Quality** slider, when `engine.graphics_quality`
  forces a level (see the [README](README.md#configuration)).
- Anti-aliasing and texture memory: the performance preset
  (`performance.physics_worker_mode: throughput`, the default, or `auto`
  with `multithreaded_rendering`) turns MSAA off and caps the texture budget
  at 128 MB.
- Roblox's **Maximum Frame Rate**, when `graphics.frame_rate_limit` sets a
  target.
- Roblox's **Theme**, when `appearance.theme` is not `roblox`.
- Roblox's **Fullscreen** setting, when `display.start_mode` is not
  `remember`.
- Voice chat, when the microphone is disabled.
- The device and interface, through the device profile (the PC profile
  shows Roblox's desktop interface).

The same group shows what is left to Roblox (volume, camera, chat and
anything not overridden) and the client settings Mocktail sets at every
start, such as Roblox's crash uploads being off.

Until 1.0.4-plus.2 the window also listed a forced quality level 3,
low-end level of detail, simpler shadows and fewer sound channels. The
performance preset did send settings for those, but Roblox 2.738 never read
them: it looks a client setting up only under the prefix it registered it
with (`F` or `DF`), and some of the names do not exist in it at all. The
preset no longer sends these 33 settings, so the game behaves as before,
and a quality level you choose in `engine.graphics_quality` now really
applies.

## Can I run multiple Roblox instances?

No. Mocktail runs one Roblox client at a time, and Plus does not change
that; it can save several accounts, but plays one at a time. Multi-instance
launching would mean working around Roblox's restrictions on running several
clients. The
[Roblox Terms of Use](https://en.help.roblox.com/hc/en-us/articles/115004647846-Roblox-Terms-of-Use)
prohibit bypassing technical protections, and violating the terms can get
your account suspended.

Bloxstrap brought the option back in
[v2.9.0](https://github.com/bloxstraplabs/bloxstrap/releases/tag/v2.9.0)
with a "use at your own risk" warning, then removed it in
[v2.10.0](https://github.com/bloxstraplabs/bloxstrap/releases/tag/v2.10.0)
after Roblox added measures against running multiple clients.

## Can I play in VR?

VR is experimental and lives on upstream's `vr` branch, which does not have
the Plus changes. Install the [build dependencies](README.md#building-from-source),
then build that branch:

```bash
git clone --recurse-submodules --branch vr https://github.com/komaruworld/mocktail.git mocktail-vr
cd mocktail-vr
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DMOCKTAIL_ENABLE_VR=ON
cmake --build build -j4
```

Start WiVRn or SteamVR/ALVR, connect your headset, then run:

```bash
./build/mocktail -vr
```
