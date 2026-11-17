#!/usr/bin/env python3
"""One-step setup of Qwen3.8-Flash-Next GSQ-RCO IQ3_S on a 32 GB Windows PC with an NVIDIA card. FAST-SETUP.bat runs this.

The first run asks where the model files are (they are never downloaded here), builds the engine with setup.py, then tunes
for a 32 GB PC and offers dual SSD streaming. Later runs only tune. It writes strata-fast.json and run-fast.bat beside
setup's own config; the tuning does not change setup's files. Nothing is written with --dry-run.

  PATH                the model files: the folder with the two .gguf files, or one of them (dragging the folder onto
                      FAST-SETUP.bat passes it here). Given again later, it sets the model up again from that folder
  --drafter-dir PATH  the DFlash2 drafter you have (config.json + model.safetensors, or its .pack). Without it the
                      first run asks; Enter downloads the MTP draft layer (about 5 GB) instead
  --rebuild           build the engine again from this folder's code, with the model folder setup used before
  --fast-prompts      the opt-in prompt switches (fused prompt experts, tensor-core GDN scan, larger chunks): faster
                      long prompts, results changed at rounding level (docs/fork/PERFORMANCE.md)
  --dry-run           show what would happen and stop
  --yes               no questions: the settings, and the best second drive unless --no-mirror
  --ram-gib N         GiB of RAM for the experts, without asking (default on a 32 GB PC: 10)
  --skip N            the expert skip, without asking: 0 (the default, every expert computed) or 7 (faster, answers change)
  --mirror-dir PATH   the folder for the copy of experts.bin (it must be on another physical drive)
  --no-mirror         one drive only
  --verify            compare an existing copy with the original by hash, too
  --config PATH       the config to start from (default strata-iq3_s.json, else strata-iq3_xxs.json)
  --out-dir PATH      where to write the config and launcher (default this folder)
"""
import argparse
import ctypes
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import threading
from pathlib import Path

HOME = Path(__file__).resolve().parent.parent
GIB = 1024 ** 3
MIRROR_SPARE = 5 * GIB
PATCHED_MARK = b"STRATA_FORK_DEFAULTS"   # only this fork's engine holds this name, from the build that applies its defaults itself

# The model files this setup tunes (ISTA-DASLab's GSQ-RCO quants), with the sizes setup.py pins (ISTA_FILES), so a download
# that did not finish is caught here, before a build. IQ3_S is the one measured; IQ3_XXS is the smaller one the tuning also reads.
MODEL_PAGE = "https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/tree/main/IQ3_S"
MODEL_RE = re.compile(r"^Qwen3\.8-Flash-Next-GSQ-RCO-(IQ3_S|IQ3_XXS)-(\d{5})-of-(\d{5})\.gguf$", re.I)
MODEL_SIZES = {
    "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf": 54817524224,
    "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf": 28800138432,
    "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf": 47039860096,
    "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf": 28800138432,
}
MMPROJ = ("mmproj-Qwen3.8-Flash-Next-BF16.gguf", 907543008)      # the picture encoder, in the same repository's top folder
DRAFTER_PAGE = "https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2"


def say(text=""):
    print(text, flush=True)


def total_ram_gib():
    class Status(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong), ("avail", ctypes.c_ulonglong),
                    ("page", ctypes.c_ulonglong), ("avail_page", ctypes.c_ulonglong), ("virt", ctypes.c_ulonglong),
                    ("avail_virt", ctypes.c_ulonglong), ("ext", ctypes.c_ulonglong)]
    s = Status()
    s.length = ctypes.sizeof(Status)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
    return s.total / GIB


def cpu_cores():
    """Per physical core: (efficiency class, its logical processors), in Windows' order."""
    k = ctypes.windll.kernel32
    size = ctypes.c_ulong(0)
    k.GetLogicalProcessorInformationEx(0, None, ctypes.byref(size))
    buf = ctypes.create_string_buffer(size.value)
    if not k.GetLogicalProcessorInformationEx(0, buf, ctypes.byref(size)):
        return []
    cores, p = [], 0
    while p < size.value:
        _rel, step = struct.unpack_from("<II", buf.raw, p)
        eff = buf.raw[p + 9]
        mask = struct.unpack_from("<Q", buf.raw, p + 32)[0]
        cores.append((eff, [b for b in range(64) if mask >> b & 1]))
        p += step
    return cores


def cpu_plan(cores):
    """The layout measured on an i7-14700KF on any hybrid CPU: workers on the P-cores' first threads, the host on the first core,
    the readers on E-cores. E-core workers made every expert slower there, and readers on P-core hyperthreads twice as slow.
    A CPU with one kind of core keeps the engine's own placement."""
    top = max((c[0] for c in cores), default=0)
    p_cores = [c[1] for c in cores if c[0] == top]
    e_threads = [t for c in cores if c[0] != top for t in c[1]]
    if not e_threads or not p_cores:
        return {"hybrid": False, "workers": [], "readers": [], "p": len(p_cores), "e": 0}
    workers = [c[0] for c in p_cores if 0 not in c]
    readers = e_threads[6:10] if len(e_threads) >= 10 else e_threads[-4:] if len(e_threads) >= 4 else []
    return {"hybrid": True, "workers": workers, "readers": readers, "p": len(p_cores), "e": len(e_threads)}


def gpu_info():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
        name, mib = out.splitlines()[0].rsplit(",", 1)
        return name.strip(), int(mib) / 1024
    except (OSError, ValueError, IndexError):
        return None, 0.0


def volumes():
    """The fixed drives with a letter: free space, physical disk, and what kind of disk it is."""
    script = r"""
$out = foreach ($v in Get-Volume | Where-Object { $_.DriveLetter -and $_.DriveType -eq 'Fixed' }) {
  try {
    $d = (Get-Partition -DriveLetter $v.DriveLetter -ErrorAction Stop | Get-Disk -ErrorAction Stop)
    $p = Get-PhysicalDisk -DeviceNumber $d.Number -ErrorAction Stop
    [pscustomobject]@{ letter = [string]$v.DriveLetter; free = [int64]$v.SizeRemaining; fs = $v.FileSystem; disk = [int]$d.Number; media = [string]$p.MediaType; bus = [string]$p.BusType; name = [string]$p.FriendlyName }
  } catch {}
}
@($out) | ConvertTo-Json -Compress
"""
    r = subprocess.run(["powershell", "-NoProfile", "-Command", script], capture_output=True, text=True)
    try:
        v = json.loads(r.stdout or "[]")
    except ValueError:
        return []
    return [v] if isinstance(v, dict) else v


def mirror_candidates(vols, pack_letter, need):
    """Other physical SSDs with room. A second partition of the same disk would gain nothing, a hard disk or a USB drive would slow reads down."""
    here = next((v for v in vols if v["letter"].upper() == pack_letter.upper()), None)
    if here is None:
        return []
    return [v for v in vols if v["disk"] != here["disk"] and v["media"] != "HDD" and v["bus"] not in ("USB", "SD", "MMC", "File Backed Virtual")
            and v["free"] >= need]


def sha256_of(path, out):
    h = hashlib.sha256()
    with open(path, "rb", buffering=0) as f:
        while True:
            b = f.read(16 * 1024 * 1024)
            if not b:
                break
            h.update(b)
    out[str(path)] = h.hexdigest()


def same_file(a, b):
    say("  Comparing the two copies by hash (a few minutes for a large file) ...")
    got = {}
    ts = [threading.Thread(target=sha256_of, args=(p, got)) for p in (a, b)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return got.get(str(a)) is not None and got.get(str(a)) == got.get(str(b))


def file_extents(path):
    """How many pieces the file is stored in (fsutil), or None if Windows won't say."""
    r = subprocess.run(["fsutil", "file", "queryextents", str(path)], capture_output=True, text=True)
    n = r.stdout.count("VCN:")
    return n if r.returncode == 0 and n > 0 else None


FRAGMENTED = 1000   # more pieces than this: some reads of the copy become many small disk requests (68,990 measured 3x slower)


def copy_experts(src, dst):
    """Unbuffered, so the copy does not push everything else out of RAM, and with the whole size reserved first, so the
    copy lands in a few large pieces. A copy written as it goes (robocopy /J) took the free gaps of a well-used drive:
    68,990 pieces, and about 880 experts read 3x slower from it."""
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.CreateFileW.restype = ctypes.c_void_p
    k.CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
    k.ReadFile.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p]
    k.WriteFile.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p]
    k.SetFilePointerEx.argtypes = [ctypes.c_void_p, ctypes.c_longlong, ctypes.c_void_p, ctypes.c_uint32]
    k.SetEndOfFile.argtypes = [ctypes.c_void_p]
    k.CloseHandle.argtypes = [ctypes.c_void_p]
    k.VirtualAlloc.restype = ctypes.c_void_p
    k.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32, ctypes.c_uint32]
    k.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32]
    READ, WRITE, SHARE_RW, OPEN_EXISTING, CREATE_ALWAYS, NO_BUFFERING = 0x80000000, 0x40000000, 3, 3, 2, 0x20000000
    invalid, chunk, size = ctypes.c_void_p(-1).value, 64 << 20, src.stat().st_size
    dst.parent.mkdir(parents=True, exist_ok=True)
    h = k.CreateFileW(str(dst), READ | WRITE, 1, None, CREATE_ALWAYS, NO_BUFFERING, None)
    if h in (None, invalid):
        return False
    k.SetFilePointerEx(h, size, None, 0)   # reserve every cluster now, as contiguous as the free space allows
    reserved = bool(k.SetEndOfFile(h))
    k.CloseHandle(h)
    if not reserved:
        return False
    n = file_extents(dst)
    if n is not None:
        say(f"  Space reserved in {n} piece{'s' if n != 1 else ''}.")
    s = k.CreateFileW(str(src), READ, SHARE_RW, None, OPEN_EXISTING, NO_BUFFERING, None)
    d = k.CreateFileW(str(dst), WRITE, 1, None, OPEN_EXISTING, NO_BUFFERING, None)
    buf = k.VirtualAlloc(None, chunk, 0x3000, 0x04)
    ok = s not in (None, invalid) and d not in (None, invalid) and bool(buf)
    got, put, done = ctypes.c_uint32(0), ctypes.c_uint32(0), 0
    while ok and done < size:
        want = min(chunk, (size - done + 4095) // 4096 * 4096)   # unbuffered I/O moves whole sectors; the tail is cut below
        ok = bool(k.ReadFile(s, buf, want, ctypes.byref(got), None)) and got.value > 0
        if ok:
            n_out = (got.value + 4095) // 4096 * 4096
            ok = bool(k.WriteFile(d, buf, n_out, ctypes.byref(put), None)) and put.value == n_out
            done += got.value
            if done % (10 << 30) < chunk:
                say(f"  {done / GIB:.0f} of {size / GIB:.0f} GB")
    for x in (s, d):
        if x not in (None, invalid):
            k.CloseHandle(x)
    if buf:
        k.VirtualFree(buf, 0, 0x8000)
    if ok and size % 4096:   # the last sector was written whole: cut the file back to the original's size
        with open(dst, "r+b") as f:
            f.truncate(size)
    return ok and done == size and dst.stat().st_size == size


def choose_mirror(a, src, vols):
    """The path to use, or None for one drive. Copies the file if it is not there yet."""
    size = src.stat().st_size
    pack_letter = src.drive[:1]
    if a.no_mirror:
        say("Dual SSD: off (--no-mirror).")
        return None
    if a.mirror_dir:
        dst = Path(a.mirror_dir) / src.name
        here = next((v for v in vols if v["letter"].upper() == pack_letter.upper()), None)
        there = next((v for v in vols if v["letter"].upper() == dst.drive[:1].upper()), None)
        if here is None or there is None or here["disk"] == there["disk"]:
            sys.exit(f"{a.mirror_dir} is not on another physical drive than {src}. It would gain nothing, so nothing was copied.")
    else:
        found = mirror_candidates(vols, pack_letter, size + MIRROR_SPARE)
        if not found:
            say(f"Dual SSD: no other SSD has {(size + MIRROR_SPARE) / GIB:.0f} GiB free, so this stays on one drive. The other settings still apply.")
            return None
        say(f"Dual SSD: a copy of experts.bin ({size / GIB:.1f} GiB) on a second SSD lets each read come from both drives.")
        say("  Measured on two 990 PRO: +12% text, +39% prompt reading. Reading does not wear an SSD; the copy writes the file once.")
        for i, v in enumerate(found, 1):
            say(f"  {i}. {v['letter']}: {v['name']}, {v['free'] / GIB:.0f} GiB free")
        if a.yes:
            pick = max(found, key=lambda v: (v["bus"] == "NVMe", v["free"]))
        else:
            ans = input(f"Which drive? 1-{len(found)}, or Enter for one drive only: ").strip()
            if not ans.isdigit() or not 1 <= int(ans) <= len(found):
                say("Dual SSD: off.")
                return None
            pick = found[int(ans) - 1]
        dst = Path(f"{pick['letter']}:\\StrataMirror") / src.name
    if dst.exists() and dst.stat().st_size == size:
        say(f"Dual SSD: {dst} is already there with the right size.")
        n = file_extents(dst)
        if n is not None and n > FRAGMENTED:
            say(f"  It is stored in {n:,} pieces, so some of its reads are slow (measured: about 880 experts 3x slower at 68,990 pieces).")
            say(f"  To fix it: delete {dst} and run FAST-SETUP.bat again; the new copy reserves its space first.")
        if a.verify and not a.dry_run and not same_file(src, dst):
            sys.exit(f"{dst} differs from {src}. Delete it and run this again.")
        return dst
    if a.dry_run:
        say(f"Dual SSD: would copy {src} to {dst} and compare the hashes.")
        return dst
    say(f"Dual SSD: copying to {dst} ...")
    if not copy_experts(src, dst):
        sys.exit(f"The copy to {dst} failed. Check the drive, delete what is there and run this again.")
    if not same_file(src, dst):
        dst.unlink(missing_ok=True)
        sys.exit("The copy differs from the original, so it was deleted. Check the drive before trying again.")
    say("  The copy matches the original.")
    return dst


def choose_ram(a, ram):
    """GiB of RAM for the experts: --ram-gib, else the person's answer, else the default. Above the safe line it warns, because Windows then pages."""
    measured = 28 <= ram <= 36
    default = 10 if measured else max(4, min(48, round(ram) - 22))
    top = max(1, round(ram) - 8)
    safe = max(default, round(ram) - 18)
    if a.ram_gib is not None:
        pick = a.ram_gib
    elif a.yes:
        pick = default
    else:
        say("How much RAM should hold the model's experts? More is faster, but leaves less for other apps.")
        say("  Measured on 32 GB (2026-11-05, expert skipping on): 14 GiB was faster than 10 GiB, but left under 1 GiB free and wrote up to 36 GB an hour to the page file; 10 GiB wrote none.")
        say("  The engine uses about 8 GiB more on top of this. Other amounts have not been measured on their own.")
        while True:
            ans = input(f"GiB of RAM, 1-{top}, or Enter for {default}: ").strip()
            if not ans:
                pick = default
                break
            try:
                pick = float(ans)
            except ValueError:
                continue
            if 1 <= pick <= top:
                break
    if not 1 <= pick <= top:
        sys.exit(f"--ram-gib must be between 1 and {top} on this PC.")
    if not measured:
        say(f"This fork is tuned for 32 GB. {pick:g} GiB of experts on a {ram:.0f} GB PC has not been measured.")
    if pick > safe:
        say(f"{pick:g} GiB leaves little free RAM: close other large programs, or Windows will page heavily, which writes to your SSD.")
    return pick


def choose_skip(a):
    """The route tail skip: 0 (the fork's default, which the engine applies by itself) or what the person picks."""
    if a.skip is not None:
        pick = a.skip
    elif a.yes:
        pick = 0
    else:
        say("Expert skip: should a missed expert that every token ranks low be skipped (neither read nor computed)?")
        say("  No (recommended): every expert is computed, so the answers are the full model's.")
        say("  Yes: about 22% faster text on the PC this was tuned on (45.9 against 35.9 tok/s), but the answers change")
        say("  (KL 0.081 against no skip, the top choice the same at 90.7% of positions).")
        ans = input("Skip experts? y/N: ").strip().lower()
        pick = 7 if ans in ("y", "yes") else 0
    if not 0 <= pick <= 8:
        sys.exit("--skip must be between 0 and 8 (0 is off, 7 is upstream's default).")
    return pick


def set_arg(args, name, value):
    if name in args:
        args[args.index(name) + 1] = value
    else:
        args += [name, value]


def launcher(cfg_path, env):
    py = HOME / ".venv" / "Scripts" / "python.exe"
    lines = ["@echo off", "rem Written by FAST-SETUP.bat. Chat page and API: http://127.0.0.1:8080", "title SlenderEngine (32 GB tuned)"]
    lines += [f'set "{k}={v}"' for k, v in env.items()]
    lines += ['set "PYTHONUTF8=1"', f'cd /d "{HOME}"', f'"{py}" "{HOME / "serve" / "server.py"}" --engine strata --config "{cfg_path}" --port 8080 --open %*', "if errorlevel 1 pause", ""]
    return "\r\n".join(lines)


def clean_path(text):
    text = (text or "").strip().strip('"').strip("'").strip()
    return Path(text).expanduser() if text else None


def find_model(path):
    """(folder, size name, shard paths) for the model files at `path`: one of the files, the folder with them, or the folder
    above it (the repository's own layout, IQ3_S\\ inside). Else a plain-words reason they cannot be used."""
    if path is None or not path.exists():
        return f"{path} was not found."
    if path.is_file():
        if not MODEL_RE.match(path.name):
            return (f"{path.name} is not one of the model files. They are named like "
                    "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf.")
        firsts = [path.with_name(re.sub(r"-\d{5}-of-", "-00001-of-", path.name, count=1))]
    else:
        firsts = []
        for folder in (path, path / "IQ3_S", path / "IQ3_XXS"):
            try:
                firsts += sorted(f for f in folder.glob("*-00001-of-*.gguf") if MODEL_RE.match(f.name))
            except OSError:
                pass
        if not firsts:
            return (f"No model files in {path}. It needs the two Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-0000N-of-00002.gguf "
                    f"files from {MODEL_PAGE}")
        firsts.sort(key=lambda f: MODEL_RE.match(f.name).group(1).upper() != "IQ3_S")      # IQ3_S first when both are there
    first = firsts[0]
    m = MODEL_RE.match(first.name)
    size, total = m.group(1).upper(), int(m.group(3))
    shards = [first.with_name(re.sub(r"-00001-of-", "-%05d-of-" % i, first.name, count=1)) for i in range(1, total + 1)]
    for sh in shards:
        if not sh.is_file():
            return f"{sh.name} is missing from {first.parent}. Both files are needed."
        want, have = MODEL_SIZES.get(sh.name), sh.stat().st_size
        if want and have != want:
            return (f"{sh.name} is {have / 1e9:.1f} GB, but the published file is {want / 1e9:.1f} GB: the download did not "
                    "finish, or the file was changed. Download it again, then run this again.")
    return first.parent, size, shards


def find_mmproj(folder):
    """The picture encoder beside the model files, or in the folder above them (where the repository keeps it), or None.
    Outside the model's own folder only a file of the full size counts (a half-copied one is not used)."""
    name, size = MMPROJ
    for f in (folder / name, folder.parent / name, folder.parent / "models" / name):
        try:
            if f.is_file() and (f.parent == folder or f.stat().st_size == size):
                return f
        except OSError:
            continue
    return None


def drafting_modules():
    if str(HOME) not in sys.path:
        sys.path.insert(0, str(HOME))
    from strata_control import drafting, drafting_fetch
    return drafting, drafting_fetch


def find_drafter(path):
    """("src", folder) for the drafter's original files, ("pack", file) for its Q4_0 pack, or a plain-words reason. Every
    file is checked against the size and SHA-256 Strata pins, so nothing later fetches a replacement."""
    drafting, _ = drafting_modules()
    if path is None or not path.exists():
        return f"{path} was not found."
    pack = (path if path.is_file() else path / drafting.PACK_NAME)
    if pack.name == drafting.PACK_NAME and pack.is_file():
        if pack.stat().st_size != drafting.PACK_BYTES or drafting.sha256_of(pack) != drafting.PACK_SHA256:
            return f"{pack.name} is not the expected file (its size or checksum differs)."
        return "pack", pack
    folder = path if path.is_dir() else path.parent
    missing = [n for n, _s, _h in drafting.FILES if not (folder / n).is_file()]
    if missing:
        return (f"{folder} has no {' or '.join(missing)}. The drafter is the config.json and model.safetensors files "
                f"from {DRAFTER_PAGE}")
    say("  Checking the drafter files ...")
    for name, size, sha in drafting.FILES:
        f = folder / name
        if f.stat().st_size != size or drafting.sha256_of(f) != sha:
            return f"{name} is not the expected file (its size or checksum differs): download it again from {DRAFTER_PAGE}"
    return "src", folder


def drafter_in_hf_cache():
    """The drafter's pinned files where the Hugging Face tools keep them, already checked; else None."""
    try:
        drafting, drafting_fetch = drafting_modules()
        have = drafting_fetch._present(drafting_fetch._hub_snapshots(), drafting.FILES)
        if len(have) == len(drafting.FILES):
            return have["model.safetensors"].parent
    except Exception:          # noqa: BLE001 - only a shortcut: without it the person is asked
        pass
    return None


def ask_path(question, check, allow_empty):
    while True:
        try:
            ans = clean_path(input(question))
        except EOFError:
            ans = None
        if ans is None:
            if allow_empty:
                return None
            say("  A path is needed. Close this window to stop.")
            continue
        got = check(ans)
        if isinstance(got, str):
            say("  " + got)
            continue
        return got


def link_or_copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def install_drafter(drafter, data_dir):
    """Put the drafter the person pointed to under the data folder and pack it there (no network: every file was checked first)."""
    drafting, drafting_fetch = drafting_modules()
    kind, where = drafter
    folder = data_dir / drafting.FOLDER
    if kind == "pack":
        link_or_copy(where, folder / where.name)
    else:
        for name, _size, _sha in drafting.FILES:
            link_or_copy(where / name, folder / "src" / name)     # packing removes these links or copies, never the originals
    say("Preparing the DFlash2 drafter ...")
    drafting_fetch.ensure(data_dir)
    say("  DFlash2 drafter ready.")


def config_path(a):
    return Path(a.config) if a.config else next((p for p in (HOME / "strata-iq3_s.json", HOME / "strata-iq3_xxs.json") if p.exists()), None)


def model_folder_of(cfg_file):
    """The model folder setup used before (the --native shard in its config), or None."""
    try:
        args = json.loads(Path(cfg_file).read_text(encoding="utf-8")).get("args") or []
        return Path(args[args.index("--native") + 1]).parent
    except (OSError, ValueError, IndexError, TypeError):
        return None


def first_run(a):
    """Ask for the model (and the drafter), then build with setup.py. Returns the drafter to install after the build, or
    "dry" on a dry run."""
    say()
    say(" SlenderEngine setup: Qwen3.8-Flash-Next GSQ-RCO IQ3_S")
    say()
    say(" You need the model's two .gguf files (84 GB), downloaded yourself from")
    say(f"   {MODEL_PAGE}")
    say(f" Optional, beside them or in the folder above: {MMPROJ[0]} (0.9 GB, for pictures).")
    say(f" Optional: the DFlash2 drafter (1.2 GB) from {DRAFTER_PAGE}")
    say()
    if a.path:
        got = find_model(clean_path(a.path))
        if isinstance(got, str):
            sys.exit(got)
    elif a.yes:
        sys.exit('--yes needs the model folder too, as in: FAST-SETUP.bat "D:\\Models\\IQ3_S" --yes')
    else:
        got = ask_path("Where are the model files? Paste or drag their folder into this window, then press Enter:\n  > ",
                       find_model, False)
    folder, size, _shards = got
    say(f"  Model: {size} in {folder}")
    mmproj = find_mmproj(folder)
    say(f"  Pictures: on ({mmproj})" if mmproj else
        f"  Pictures: off. To turn them on later, put {MMPROJ[0]} beside the model and run FAST-SETUP.bat --rebuild")

    drafter = None
    have = installed_drafter(config_path(a))
    if have and not a.drafter_dir:
        say(f"  Drafter: {have}, already set up")
        cmd_drafter = ["--drafter", "dflash2"] if have == "DFlash2" else []
        return run_setup(a, size, folder, mmproj, cmd_drafter, None)
    if a.drafter_dir:
        drafter = find_drafter(clean_path(a.drafter_dir))
        if isinstance(drafter, str):
            sys.exit(drafter)
    elif (cached := drafter_in_hf_cache()) is not None:
        drafter = ("src", cached)
    elif not a.yes:
        say()
        say(" A drafter guesses a few words ahead, which makes answers about twice as fast.")
        drafter = ask_path("Where is the DFlash2 drafter? Paste its folder, or press Enter to download the MTP draft layer "
                           "(about 5 GB) instead:\n  > ", find_drafter, True)
    say(f"  Drafter: DFlash2 ({drafter[1]})" if drafter else "  Drafter: the MTP draft layer, downloaded during setup (about 5 GB)")
    return run_setup(a, size, folder, mmproj, ["--drafter", "dflash2"] if drafter else [], drafter)


def installed_drafter(cfg_file):
    """"DFlash2" or "MTP" when the data folder of an earlier install already holds a drafter, else None."""
    try:
        args = json.loads(Path(cfg_file).read_text(encoding="utf-8")).get("args") or []
        data_dir = Path(args[args.index("--pack") + 1]).parent.parent
    except (OSError, ValueError, IndexError, TypeError):
        return None
    try:
        drafting, _ = drafting_modules()
        if drafting.pack_ready(data_dir):
            return "DFlash2"
    except Exception:          # noqa: BLE001
        pass
    return "MTP" if (data_dir / "mtp" / "rt" / "experts.bin").is_file() else None


def run_setup(a, size, folder, mmproj, drafter_args, drafter):
    cmd = [sys.executable, str(HOME / "setup.py"), "--family", "qwen", "--model", size, "--build", "--context", "262144",
           "--kv", "int8", "--vision", "gpu" if mmproj else "none", "--gguf-dir", str(folder),
           "--experimental-speed-projection", "off", "--yes", "--no-start"] + drafter_args
    say()
    say(" Next, setup checks this PC, installs build tools that are missing (Visual Studio 2022 Build Tools, CUDA 13.0;")
    say(" Windows may ask for permission), builds the engine, and writes a 50 GB experts file into the data folder it")
    say(" names. This takes a while; keep the window open. Setup may say IQ3_S needs 62 GB of RAM: that is upstream")
    say(" Strata's figure. This fork runs it on 32 GB by reading experts from the SSD.")
    if a.dry_run:
        say("\nDry run: nothing was built. Setup would run as\n  " + subprocess.list2cmdline(cmd))
        return "dry"
    say()
    rc = subprocess.call(cmd, cwd=str(HOME))
    if rc != 0 or config_path(a) is None:
        sys.exit("\nSetup did not finish, so nothing was tuned. See the messages above, then run FAST-SETUP.bat again.")
    return drafter


def main():
    if os.name != "nt":
        sys.exit("This setup is for Windows. On Linux, use ./setup.sh and the settings in FORK.md.")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?")
    ap.add_argument("--drafter-dir")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--fast-prompts", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--ram-gib", type=float)
    ap.add_argument("--skip", type=int)
    ap.add_argument("--mirror-dir")
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--config")
    ap.add_argument("--out-dir", default=str(HOME))
    a = ap.parse_args()

    source = config_path(a)
    if a.rebuild and not a.path:
        folder = model_folder_of(source) if source else None
        if folder is None:
            sys.exit("Nothing to rebuild yet: run FAST-SETUP.bat with the model folder first.")
        a.path = str(folder)
    drafter = None
    if a.path or source is None or not source.exists():
        drafter = first_run(a)
        if drafter == "dry":
            return 0
        source = config_path(a)
    cfg = json.loads(source.read_text(encoding="utf-8"))
    args = cfg.get("args") or []
    if "--pack" not in args:
        sys.exit(f"{source} has no --pack: this setup is for the IQ packs of Qwen3.8-Flash-Next.")

    exe = Path(cfg.get("exe") or "")
    if not exe.exists():
        sys.exit(f"The engine {exe} is missing. Build it with: FAST-SETUP.bat --rebuild")
    if PATCHED_MARK not in exe.read_bytes():
        sys.exit(f"{exe} is a stock engine, or this fork's from before it applied its own defaults. Build this one with: FAST-SETUP.bat --rebuild")

    src = Path(args[args.index("--pack") + 1]) / "experts.bin"
    if not src.exists():
        sys.exit(f"{src} is missing. The dual SSD option needs a pack with experts.bin.")

    ram = total_ram_gib()
    cores = cpu_cores()
    cpu = cpu_plan(cores)
    gpu, vram = gpu_info()
    say(f"RAM {ram:.0f} GB. GPU {gpu or 'not found'}{f', {vram:.0f} GB' if gpu else ''}. CPU {cpu['p']} {'P-cores' if cpu['hybrid'] else 'cores'}{f', {cpu["e"]} E-core threads' if cpu['hybrid'] else ''}.")
    if gpu is None:
        sys.exit("No NVIDIA card found. These settings were only measured on NVIDIA, with the CUDA engine.")
    budget = choose_ram(a, ram)
    if vram < 11:
        say(f"Measured on 12 GB of VRAM. With {vram:.0f} GB, fewer experts fit on the card, so expect less speed.")
    if not cpu["hybrid"]:
        say("This CPU has one kind of core, so the engine keeps its own placement. Only a hybrid Intel CPU was measured.")
    if "RTX 40" not in gpu and "RTX 4" not in gpu:
        say("The RTX 40 kernel table will not apply to this card: other cards may gain less.")

    skip = choose_skip(a)
    vols = volumes()
    mirror = choose_mirror(a, src, vols)

    for k, v in (("--spec", "4"), ("--spec-min-p", "0.5"), ("--resident-budget-gib", f"{budget:g}"), ("--adapt-every", "4"), ("--adapt-swaps", "24"),
                 ("--adapt-async", "1"), ("--kv-resident", "32768"), ("--conversation-cache-mib", "0")):
        set_arg(args, k, v)
    # The fork's lossless switches (K/V lending, the stager, read-through, one VRAM arena, 16K prompt chunks
    # with a RAM budget ...) and skip 0 are the engine's own defaults now (FORK.md, "Defaults"), so only this PC's settings go here.
    env = {"STRATA_RESIDENT_HEADROOM_GIB": "1.5"}
    if skip != 0:
        env["STRATA_ROUTE_TAIL_SKIP"] = str(skip)
    if cpu["hybrid"]:
        set_arg(args, "--pool-workers", str(len(cpu["workers"])))
        env["STRATA_POOL_CORES"] = ",".join(map(str, cpu["workers"]))
        if cpu["readers"]:
            env["STRATA_COLD_READER_CPUS"] = ",".join(map(str, cpu["readers"]))
    if mirror is not None:
        env["STRATA_EXPERTS_MIRROR"] = str(mirror)
    if a.fast_prompts:
        # Opt-in prompt switches, rounding-level changes (docs/fork/PERFORMANCE.md): fused prompt experts, the tensor-core GDN scan,
        # and chunks of up to 16,896 tokens (the two byte-identical memory changes that make room for them, then the larger chunks)
        env.update({"STRATA_PF_FUSED": "1", "STRATA_GDN_TC": "1", "STRATA_PF_HC_REGION": "1", "STRATA_EMB_REUSE_ACCOUNT": "1",
                    "STRATA_PREFILL_AUTO_MAX": "32768", "STRATA_PREFILL_RING": "256"})
        set_arg(args, "--prefill", "auto:32768")
    # DFlash2 drafts instead of the MTP layer when its drafter is on disk (Strata-data\dflash2\*.pack): on the PC this was tuned on, follow-ups
    # +5% and 50K prompt reading +11%, thinking-on short answers level (FORK.md). The server then sets --draft dflash2 --spec 6 --draft-conf 0.8.
    data_dir = Path(args[args.index("--pack") + 1]).parent.parent
    if drafter and not a.dry_run:
        install_drafter(drafter, data_dir)
    df2_packs = sorted((data_dir / "dflash2").glob("*.pack"))
    if df2_packs:
        cfg["drafting"] = "dflash2"
    out = Path(a.out_dir)
    cfg["args"] = args
    cfg["log"] = str(out / "strata-fast.log")
    cfg["vram_elastic"] = True
    if isinstance(cfg.get("vision"), dict) and cfg["vision"].get("gpu"):
        cfg["vision"].update({"elastic": True, "elastic_mib": 1536, "elastic_idle_s": 45})

    say()
    say("Settings:")
    for k, v in env.items():
        say(f"  {k}={v}")
    say(f"  Expert skip {skip}" + (" (none: every expert computed)" if skip == 0 else " (answers change)"))
    say("  Plus the fork's defaults, which the engine applies by itself (FORK.md, \"Defaults\"; STRATA_FORK_DEFAULTS=0 turns them off)")
    say(("  DFlash2 drafter" if df2_packs else "  MTP drafter, " + " ".join(args[args.index("--spec"):args.index("--spec") + 2])) +
        f", {budget:g} GiB of experts in RAM, conversations not saved in RAM")
    if a.dry_run:
        say("\nDry run: nothing was written.")
        return 0
    out.mkdir(parents=True, exist_ok=True)
    cfg_path = out / "strata-fast.json"
    cfg_path.write_text(json.dumps(cfg, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "run-fast.bat").write_text(launcher(cfg_path, env), encoding="utf-8", newline="")
    say(f"\nDone. Start it with {out / 'run-fast.bat'}")
    say("Close your browser and other large programs first: this uses most of a 32 GB PC.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
