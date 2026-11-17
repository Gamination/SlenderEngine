# SlenderEngine

**Big models, lean hardware.**

Run **Qwen3.8-Flash-Next IQ3_S locally on a Windows PC with 32 GB of RAM**. SlenderEngine is a speed fork of [Strata](https://github.com/Niko1221/Strata) that tunes how the model uses your GPU, RAM and SSD, with an optional second SSD for faster reads. The downloaded model files are not changed. Setup writes the model's experts into one extra file, which the engine streams from the SSD.

**Start here:** [Speeds](#speeds-on-this-pc) · [Requirements](#what-you-need) · [Setup](#easy-setup) · [Qwen3.6-35B-A3B](#qwen36-35b-a3b) · [Troubleshooting](#troubleshooting) · [Measurements](docs/fork/PERFORMANCE.md)

## Speeds on this PC

i7-14700KF, 32 GB DDR5-6000, RTX 4070 Ti 12 GB, Windows 11. Both models at 262K context with an 8-bit cache and thinking on.

| | Qwen3.8-Flash-Next GSQ-RCO IQ3_S | Qwen3.6-35B-A3B UD-Q4_K_M |
|---|---:|---:|
| Short answers | 43-44 tokens/s | 134 tokens/s |
| An answer after a 50K-token prompt | 37-43 tokens/s | about 98 tokens/s |
| Reading a 50K-token prompt | about 2,100 tokens/s; 2,860-2,880 with this PC's launcher, which adds the prompt switches below | about 4,500 tokens/s |
| Model files | 84 GB, plus a 50 GB experts file setup writes | 23 GB |
| Experts kept in | SSD, with a 10 GiB RAM cache | RAM (18.7 GB) |
| Setup | `FAST-SETUP.bat` | `FAST-SETUP-35B.bat` |

Flash-Next: 2026-11-13, 3 runs each, with the DFlash2 drafter. Qwen3.6: 2026-11-15, one run each. Ranges are wide because speed depends on how many drafted tokens are accepted. Your speed will differ. The details are in each model's section below.

The Flash-Next prompt switches are opt-in: `.\FAST-SETUP.bat --fast-prompts` turns them on. They change results only at rounding level ([their quality checks](docs/fork/PERFORMANCE.md)):

| Switch | What it does |
|---|---|
| `STRATA_PF_FUSED=1` | Fused int8 kernels for the prompt's experts |
| `STRATA_GDN_TC=1` | The GDN recurrence as a tensor-core scan |
| `--prefill auto:32768`, `STRATA_PREFILL_AUTO_MAX=32768`, `STRATA_PF_HC_REGION=1`, `STRATA_EMB_REUSE_ACCOUNT=1`, `STRATA_PREFILL_RING=256` | Larger prompt chunks: up to 16,896 tokens a pass instead of 12,544 |

This PC's launcher also sets other switches, so `--fast-prompts` alone may read a little slower than 2,860-2,880. It has not been measured on its own.

## What you need

| Part | Requirement for this setup |
|---|---|
| System | Windows 11; Windows 10 has not been tried. `FAST-SETUP.bat` is for Windows with an NVIDIA GPU. |
| RAM | 32 GB. Close other large apps before running the model. |
| GPU | NVIDIA with 12 GB of VRAM, as measured here on an RTX 4070 Ti. An 8 GB RTX 4060 laptop has run it too, with extra switches ([its report](docs/fork/LAPTOP-RTX4060.md)). Other cards have not been tried. |
| Storage | About 140 GB free on an SSD: the 84 GB model files, a 50 GB experts file setup writes from them, and room for the build. A second physical SSD is optional; it needs 50 GB for a copy of the experts file plus 5 GB spare. |
| Downloads | You download the model files yourself (step 1). Setup itself downloads only build tools that are missing and, if you have no drafter, the 5 GB MTP draft layer. |

Install [Git for Windows](https://git-scm.com/download/win) and your NVIDIA graphics driver first. Setup installs Python, CMake and Ninja if they are missing, and also Visual Studio 2022 Build Tools and the CUDA 13.0 toolkit. Windows may ask for administrator permission for those.

This guide builds the fork from source. For AMD, Mac, Linux or the upstream desktop app, use the [original Strata guides](https://github.com/Niko1221/Strata).

## Easy setup

### 1. Download the model

From [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/tree/main/IQ3_S), download the two files in the `IQ3_S` folder (54.8 GB and 28.8 GB) into one folder on the SSD. On a 100 Mbit/s line that takes about 2 hours (an estimate from the size, not timed).

Optional, and setup picks them up if they are there:

- **Pictures:** `mmproj-Qwen3.8-Flash-Next-BF16.gguf` (0.9 GB) from the same repository's top folder, saved beside the two files or in the folder above them. Without it, images are off.
- **The DFlash2 drafter:** `config.json` and `model.safetensors` (1.2 GB) from [incoai/Qwen3.8-Flash-Next-DFlash2](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2), in a folder of their own. It is the drafter the speeds above were measured with. Without it, setup downloads the MTP draft layer (5 GB) instead.

### 2. Download this fork and run setup

Open a folder on the SSD. Right-click inside it and choose **Open in Terminal** (PowerShell; on Windows 10, type `powershell` in the File Explorer address bar instead), then paste:

```powershell
git clone --branch speed-fork --single-branch https://github.com/rafid-dev/SlenderEngine.git
cd SlenderEngine
.\FAST-SETUP.bat
```

Only two files are for you: **`FAST-SETUP.bat`** (setup) and **`run-fast.bat`** (start, written by setup). `START-HERE.bat`, `SETUP.bat`, `UPDATE.bat` and `BENCH.bat` are upstream Strata's tools. Setup uses them itself, but they don't set up this fork's tuning.

### 3. Answer setup's questions

| Question | Answer |
|---|---|
| Where are the model files? | Paste or drag the folder with the two files into the window, then press Enter. You can also drag the folder onto `FAST-SETUP.bat` instead of starting it. |
| Where is the DFlash2 drafter? | Paste its folder, or **press Enter** to download the MTP draft layer. Not asked when the drafter is already in your Hugging Face cache. |
| How much RAM for the cache? | **Press Enter for 10 GiB** on a 32 GB PC. This is the model's RAM cache, not its total memory use. |
| Skip experts? | **Press Enter for No.** Skipping leaves out the least-weighted experts the model picks for each token: faster, but answers change. |
| Which second SSD? | **Press Enter to use one SSD**, or choose a listed second physical SSD. Setup copies the 50 GB experts file there and checks the copy. |

Between the drafter and RAM questions, setup builds the engine and writes the 50 GB experts file into `Strata-data` beside the source folder. That takes a while; keep the window open even if it seems stuck on one step. Setup may warn that IQ3_S needs 62 GB of RAM: that is upstream Strata's figure, and this fork runs it on 32 GB.

Setup creates **`run-fast.bat`** and **`strata-fast.json`**. The full path has **not yet been tested from a clean clone**.

### 4. Start chatting

Double-click **`run-fast.bat`**, or run this in the same terminal:

```powershell
.\run-fast.bat
```

Keep its window open. The browser opens at [http://127.0.0.1:8080](http://127.0.0.1:8080); wait for the model to finish loading, then send a message.

To stop, press **Ctrl+C** in the server window. Next time, just run `run-fast.bat` again. To change your RAM or SSD choices, stop the server and rerun `FAST-SETUP.bat`.

### Updating

Stop the server, then in the source folder:

```powershell
git pull
.\FAST-SETUP.bat --rebuild
```

This gets the newest code, rebuilds the engine with the model folder you gave before, and tunes it again. Upstream's `UPDATE.bat` keeps an engine you built yourself, but it hasn't been tested with this fork.

## Speed and memory

With expert skipping off and thinking enabled, the 2026-11-13 tests measured:

| | Speed |
|---|---:|
| Short answers | 43-44 tokens/s |
| An answer after a 50K-token prompt | 37-43 tokens/s |
| A follow-up question | 34-40 tokens/s |
| Reading a 50K-token prompt | about 2,100 tokens/s with the fork's defaults; 2,860-2,880 with this PC's own launcher |

These results are from one PC: i7-14700KF, 32 GB DDR5-6000, RTX 4070 Ti 12 GB, two Samsung 990 PRO SSDs, Windows 11, IQ3_S and 262K context, 3 runs each. A small draft model guesses the next few tokens and the main model checks them; speed depends on how many guesses are right, so the ranges are wide. The faster prompt reading uses opt-in settings that aren't defaults yet; they and their quality checks are on the measurements page, along with older results. Your speed will differ.

At the default 10 GiB cache, 2.7-2.8 GB of RAM stayed free in those tests and nothing was written to the page file. On the GPU, the engine sizes its expert cache to fill the card and keeps about 0.7 GB free. Giving it more RAM cache can leave too little memory for Windows and slow it down: 14 GiB was faster here but paged heavily. Start at 10 GiB and close other large apps.

See [performance, memory measurements and updates](docs/fork/PERFORMANCE.md) for the test conditions and older comparisons. Answer quality has not been measured with a task suite.

## Qwen3.6-35B-A3B

The same fork also runs **[Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)** (35B parameters, 3B active a token) at its full 262K context on the same 32 GB PC with a 12 GB card. Its experts stay in RAM as a mapped file instead of streaming from the SSD, so it is much faster than Flash-Next, and smaller.

First download `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` (22.7 GB) from [unsloth/Qwen3.6-35B-A3B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF). It must be this repository's file, which has the MTP block; the one in Unsloth's repository without MTP is a different size and is refused. For pictures, put `mmproj-F16.gguf` (0.9 GB) from the same repository beside it. The drive needs about 20 GB more for an expert cache file the engine writes next to the model on its first start.

Then, in the SlenderEngine folder from [step 2](#2-download-this-fork-and-run-setup):

```powershell
.\FAST-SETUP-35B.bat
.\run-qwen36-fast.bat
```

Setup asks where the file is (paste or drag it, or its folder, into the window). A copy already in your Hugging Face cache is used without asking. Setup builds the engine for your card, then writes `run-qwen36-fast.bat` with the settings chosen for your card, RAM and CPU. It downloads nothing of the model. After a `git pull`, `.\FAST-SETUP-35B.bat --rebuild` rebuilds with the same file. The browser page and API are the same as Flash-Next's, on the same port, so run one model at a time.

Measured on the PC above, thinking on, 262K context, 8-bit cache. The first column is the launcher's setting since 2026-11-15 (switches that speed up the MTP step, the CPU experts and the MTP block's prompt pass without changing the model's output, one run each); the others are earlier settings:

| | With MTP, 2026-11-15 (the launcher's setting) | With MTP, 2026-11-14 | With MTP, 2026-11-13 | Without MTP |
|---|---:|---:|---:|---:|
| Short answers | 134 tokens/s | 119-121 tokens/s | 106-119 tokens/s | 94-96 tokens/s |
| An answer after a 50K-token prompt | about 98 tokens/s | 91 tokens/s | 78-88 tokens/s | 75-83 tokens/s |
| Reading a 50K-token prompt | about 4,500 tokens/s | about 4,200 tokens/s | 4,100-4,300 tokens/s | 4,400-4,560 tokens/s |
| A follow-up question | reads only its new tokens (about 0.2 s after a 50K prompt) | the same | the same | the same |

The experts take about 18.7 GB of RAM as a mapped file. With a browser and a few apps open, it ran at full speed. A program above about 4 GB beside it made Windows page, and the speed fell by a third or more. Setup warns when your RAM or VRAM is short; a 16 GB PC is expected to page heavily. The switches, quality checks and limits are in [QWEN36-12GB.md](docs/fork/QWEN36-12GB.md). Like the Flash-Next setup, this path has **not yet been tested from a clean clone**.

## Troubleshooting

| Problem | What to do |
|---|---|
| `git` is not recognized | Install Git for Windows, then open a new terminal. |
| PowerShell cannot find the batch file | Make sure you are inside `SlenderEngine` and include `.\`, as in `.\FAST-SETUP.bat`. |
| `FAST-SETUP.bat` is missing | That folder is upstream Strata, not this fork. Download this fork with the command in step 2. |
| "… is X GB, but the published file is Y GB" | The model download did not finish, or the file was changed. Download that file again. |
| "No model files in …" | Point setup to the folder that holds the two `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-0000N-of-00002.gguf` files, or to one of them. |
| Python could not be installed | Install [64-bit Python 3.12](https://www.python.org/downloads/windows/), enable **Add python.exe to PATH**, and rerun setup. |
| Build tools or CUDA could not be installed | See the manual installation steps below, then rerun setup. |
| Tuning stops with "… is a stock engine, or this fork's from before it applied its own defaults" | The engine wasn't built from this fork's code. Run `.\FAST-SETUP.bat --rebuild`. |
| The chat page does not open | Keep the server window open, check it for errors, and open [the local page](http://127.0.0.1:8080) yourself. |
| Port 8080 is already in use | Stop the other server, or run `.\run-fast.bat --port 8081` and use [port 8081](http://127.0.0.1:8081). |
| The PC slows down or runs out of memory | Close other large apps and rerun setup with a smaller RAM cache. More cache is not always faster. |

<details>
<summary>Manual build tools and engine rebuild</summary>

If the automatic tool installation fails:

1. Download [Build Tools for Visual Studio 2022](https://aka.ms/vs/17/release/vs_buildtools.exe). In the installer, select **Desktop development with C++**, including the MSVC compiler and Windows SDK.
2. Install **CUDA Toolkit 13.0** from the [NVIDIA archive](https://developer.nvidia.com/cuda-toolkit-archive). The toolkit includes the compiler; a graphics driver alone is not enough. This matches the version setup tries to install.
3. Open a new terminal in the source folder and run `.\FAST-SETUP.bat` again (or `.\FAST-SETUP.bat --rebuild` if it got as far as the model questions before).

Setup installs CMake and Ninja if needed. More compiler details: [source installation](docs/SOURCE_INSTALL.md#from-source-windows).

</details>

<details>
<summary>Optional settings, second SSD and app connections</summary>

Run these from PowerShell in the source folder **after the first install**:

| Command | What it does |
|---|---|
| `.\FAST-SETUP.bat --ram-gib 12` | Sets a 12 GiB cache without asking for its size. |
| `.\FAST-SETUP.bat --skip 7` | Turns expert skipping on without asking: faster, but answers change. `--skip 0` turns it back off. |
| `.\FAST-SETUP.bat --no-mirror` | Turns off second-SSD reads. The existing copy stays on disk. |
| `.\FAST-SETUP.bat --mirror-dir E:\StrataMirror` | Uses a folder on a different physical SSD. Replace `E:` with your drive. |
| `.\FAST-SETUP.bat --verify` | Checks an existing SSD copy by hash. Run this after updating model files; the engine only checks file sizes. |
| `.\FAST-SETUP.bat --fast-prompts` | Turns on the opt-in prompt switches listed under [Speeds](#speeds-on-this-pc). |
| `.\FAST-SETUP.bat "D:\Models\IQ3_S"` | Sets the model up again from that folder (rebuilds, then tunes). |
| `.\FAST-SETUP.bat --drafter-dir "D:\Models\DFlash2"` | Uses the DFlash2 drafter in that folder without asking (first run or with a model folder). |
| `.\FAST-SETUP.bat --rebuild` | Rebuilds the engine from this folder's code with the model folder you gave before, then tunes. |
| `.\FAST-SETUP.bat --dry-run --yes --no-mirror` | Previews tuning without writing files. On a first run it shows the setup command instead of running it. |
| `.\FAST-SETUP.bat "D:\Models\IQ3_S" --yes` | Installs with no questions: the MTP drafter unless `--drafter-dir` is given, tuning defaults, and a second drive if there is one. |

A second SSD must be a different physical drive, not another partition of the same SSD. The copy of the experts file (`experts.bin`, 50 GB) is checked by hash when it is made. You can keep using one SSD without it.

For apps using the local API:

| API | Base URL |
|---|---|
| OpenAI-compatible | `http://127.0.0.1:8080/v1` |
| Anthropic-compatible | `http://127.0.0.1:8080` |

Use your configured API key if enabled. See [API details](docs/DETAILS.md#using-it) for authentication and requests.

The DFlash2 drafter is [incoai's](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2), Apache-2.0 according to its model page (checked 2026-11-15). Setup packs it into `Strata-data\dflash2` and tuning uses it from there. It is optional: without it the MTP draft layer drafts.

</details>

<details>
<summary>Example: a black hole demo made with the local model</summary>

![A real-time black hole written by this model, running locally on this fork](docs/fork/media/blackhole.gif)

Written by Qwen3.8-Flash-Next IQ3_S through OpenCode on the PC described above. Recorded frame by frame at 4K, so playback is smoother than the live page.

</details>

## More details

- [FORK.md](FORK.md): engine changes, switches and measurements.
- [Qwen3.6-35B-A3B on a 12 GB card](docs/fork/QWEN36-12GB.md): its configuration, measurements and quality checks.
- [The web app](docs/WEB_APP.md): the chat page at the server's address, where its data is kept, and its security.
- [Sharing over Tailscale](docs/TAILSCALE.md): use the page from your phone or another PC, HTTPS and private, with `TAILSCALE-SHARE.bat`.
- [Worklog](docs/fork/WORKLOG.md): recorded tests, including failed experiments.
- [The prompt room](docs/fork/PROMPT-ROOM.md): how long prompts read in fewer passes, and how it was found and checked.
- [RTX 4060 8 GB laptop report](docs/fork/LAPTOP-RTX4060.md): optional switches contributed by [Gamination](https://github.com/Gamination).

## Credits

SlenderEngine is [Strata](https://github.com/Niko1221/Strata), by Niko1221 and the Strata contributors, with speed changes for 32 GB PCs on top. The engine, the server, the setup and almost all of the code are theirs.

- **Where it started.** This repository began on 2026-11-03 as a GitHub fork of Niko1221/Strata, from commit `c62d5e3` (engine 0.1.55), under the name Strata-32GB. On 2026-11-06 it moved to upstream `v0.2.11` (engine 0.1.57). It has since left GitHub's fork network and was renamed SlenderEngine; the full git history, including every upstream commit, is kept.
- **Still Strata underneath.** The program is still `strata`, its switches are still `STRATA_*` and its config files are still `strata-*.json`, so Strata's documentation applies. [FORK.md](FORK.md) lists what this fork changes, and [compare with upstream v0.2.11](https://github.com/rafid-dev/SlenderEngine/compare/c3b3f50...speed-fork) shows every changed line.
- **Upstream's credits** for the model, llama.cpp / ggml and the projects Strata learned from are in [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md#credits).
- **Contributors to this fork:** [Gamination](https://github.com/Gamination), the 8 GB card switches and the RTX 4060 laptop measurements.
- **The models:** [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) and [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) by the Qwen team; the Qwen3.6 GGUFs with the MTP block by [Unsloth](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF).

Code is [MIT licensed](LICENSE), keeping Niko1221's copyright alongside this fork's; model weights and optional components have their own licenses. SlenderEngine is not made or endorsed by the Strata project. Problems with this fork belong in [this repository's issues](https://github.com/rafid-dev/SlenderEngine/issues), not upstream's.
