# Handoff — environment setup for Einstein's Mom

**Audience:** the Claude Opus instance implementing `spec.md`. Read this first.

Everything below was verified on this machine on 2026-09-02, except where marked
otherwise. Facts about the GPU stack are cross-referenced in
`~/Projects/cuda/LLM-local-access.md`, which is the authority on the models and
the second disk.

---

## 0. Do this first: is the GPU actually live?

**As of writing, the NVIDIA driver was installed but the machine had not yet
rebooted into it.** The driver is staged in NixOS generation 10 (kernel 7.1.12).
Until a reboot happens, there is no GPU.

```bash
nvidia-smi                       # expect: NVIDIA GeForce RTX 3090, 24576MiB
~/Projects/cuda/verify-gpu.sh    # fuller check: driver, modules, real generation
```

If `nvidia-smi` is not found, or anything dies with
`libcuda.so.1: cannot open shared object file`, the machine has booted a
generation without the driver. Check `nixos-rebuild list-generations` and reboot
into the newest. **Do not attempt to work around this in software** — nothing in
this project works without the GPU, and the failure is a boot-generation issue,
not a library-path issue.

---

## 1. Machine facts

| | |
|---|---|
| OS | NixOS 26.05, flake config at `~/Projects/nixos` (**not** `/etc/nixos`) |
| GPU | RTX 3090, 24 GB, `sm_86`, driver 595.71.05 |
| Kernel | 7.1.12 — pinned; 595 does not build against 7.2 |
| CPU / RAM | 16 threads, 46 GiB |
| Disk | 814 G free on `/` |
| Inference | llama.cpp b9190 with CUDA, on `PATH` |

**This is a NixOS machine.** Per `~/.claude/CLAUDE.md`: do not use another
distro's package manager, do not edit generated system files, and make
persistent changes in the Nix config. User-scoped changes go in
`~/Projects/nixos/home/user.nix` and need no sudo:

```bash
home-manager switch --flake ~/Projects/nixos#user
```

Flakes only see git-tracked files — `git add` new files before switching.

---

## 2. The gap you have to close

The machine has an **inference** stack, not a **training** stack. llama.cpp does
not train. E2 and E3b in the spec need PyTorch, and PyTorch is not installed.

Two ways to get it, and the recommended one is deliberately *not* the pure-Nix
one:

**Do not** try `nixpkgs.python3Packages.torch` with `cudaSupport`. It is a
multi-hour local build (nothing CUDA-enabled is in the binary cache), and `trl`
/ `peft` / `bitsandbytes` are either unpackaged or version-skewed against it.
You will lose a day and end up with a stack you cannot upgrade.

**Do** use a Nix devShell that provides the *linking environment*, plus `uv` for
the Python packages from upstream wheels. The declarative part stays in Nix; the
fast-moving ML packages come from wheels that already bundle their own CUDA
runtime. This is the idiomatic NixOS approach for ML work and it respects the
"declarative" preference — the shell itself is declared in `flake.nix`.

### Why the devShell is needed at all

PyTorch wheels are built against a normal FHS filesystem. On NixOS there is no
`/usr/lib`, so they cannot find `libstdc++`, `libz`, etc. The devShell fixes
this by setting `LD_LIBRARY_PATH`. Critically it must also expose
`/run/opengl-driver/lib`, which is where NixOS puts the NVIDIA driver's
`libcuda.so.1` — the one library the wheels do *not* bundle, because it must
match the kernel module.

### `flake.nix`

Written to `~/Projects/einsteins-mom/flake.nix`. Verified to contain valid
attribute names against this nixpkgs (`uv` 0.11.21, `python312`), but **the
devShell has not been entered or tested** — the GPU was not live at authoring
time. Expect to iterate on the library list.

```bash
cd ~/Projects/einsteins-mom
nix develop          # first entry downloads the toolchain
```

### Python packages

Inside the devShell:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install transformers peft trl datasets accelerate bitsandbytes scikit-learn
```

Notes:

- **Python 3.12, not 3.13.** Wheel coverage for the ML stack is still better on
  3.12, and `bitsandbytes` in particular lags.
- **Pick the CUDA wheel index deliberately.** Driver 595 is new and supports
  CUDA 12.x and 13.x, so `cu128` is safe; check what PyTorch currently ships and
  prefer the newest CUDA build it offers. Never install a `+cpu` build — it will
  silently work and silently never touch the GPU.
- `bitsandbytes` is only needed for 4-bit QLoRA (the OOM fallback in spec §9).
  Skip it if bf16 LoRA fits.

### Verify the stack before writing any experiment code

```python
import torch
print(torch.__version__, torch.version.cuda)
print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
print(torch.cuda.get_device_capability(0))   # expect (8, 6)
x = torch.randn(4096, 4096, device='cuda'); print((x @ x).sum().item())
```

All five lines must succeed. `torch.cuda.is_available() == False` with no error
almost always means a `+cpu` wheel or a missing `/run/opengl-driver/lib` on
`LD_LIBRARY_PATH`.

---

## 3. Models

### The 4B child model (E2) — you must download this

Training needs **HF safetensors**, not GGUF. The GGUF at
`~/models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf` is inference-only and cannot be
finetuned. Get the real weights (~8 GB):

```bash
hf download Qwen/Qwen3-4B-Instruct-2507 --local-dir ~/models/hf/Qwen3-4B-Instruct-2507
```

`hf` is already installed. There is ample disk.

### The 27B judges and instruments (E1, E3) — already on disk

Six 27B Qwen3.5 GGUFs live on a **second NVMe that is probably not mounted**.
It is not in `/etc/fstab`; udisks2 mounts it on demand, so after a reboot it
will very likely be absent.

```bash
findmnt /dev/nvme1n1p3 || \
  udisksctl mount -b /dev/disk/by-uuid/c730db4b-e495-4387-a1c7-6d40c491ed94
HDD=/run/media/user/c730db4b-e495-4387-a1c7-6d40c491ed94
```

No password or sudo is needed: the files are owned by uid 1000, which is also
this user. Full inventory, per-model metadata, and the VRAM/context budget are
in `~/Projects/cuda/LLM-local-access.md`. Read it before E1 — it records which
models are stock and which are ablated, and flags one file whose name
misreports its architecture.

**Only ever read from that disk.** It is an old Ubuntu install's root
filesystem. Write results into this project directory.

---

## 4. Running the 27Bs for E1

```bash
llama-server -m "$HDD/home/user/models/qwen3.6-27b/Qwen3.6-27B-Q4_K_M.gguf" \
  -ngl 99 -c 8192 --host 127.0.0.1 --port 8080
```

OpenAI-compatible API at `http://127.0.0.1:8080/v1`, so the `openai` Python
client works against it unchanged. Keep it bound to `127.0.0.1`; the firewall is
on with no open ports and there is no reason to change that.

`-ngl 99` puts all layers on the GPU. At ~15.7 GB of weights this is correct for
all six models. **Do not let llama.cpp auto-fit context** — these models
advertise a 262144 training context that does not fit; set `-c` explicitly.

Only one model fits in VRAM at a time. E1 iterates over six models, so shut each
server down before starting the next. Do not run llama-server and PyTorch
training simultaneously.

---

## 5. Sequencing

The GPU is the bottleneck and everything contends for it. Suggested order:

1. Reboot, confirm GPU (§0).
2. Mount the second disk, run **E1**. Inference only, needs no PyTorch — so this
   proceeds while the Python stack is still being sorted out.
3. Set up the devShell and PyTorch (§2). Verify with the five-line check.
4. Download the 4B safetensors (§3).
5. Write `predictions.md`, **commit it**, then run E2.

E1 gates everything. If the probe harness cannot separate the stock 27Bs from
the ablated ones, stop and fix the harness before training anything.

---

## 6. Gotchas, ranked by likelihood of biting you

1. **GPU not live because the machine hasn't rebooted.** §0.
2. **Second disk not mounted after reboot.** §3.
3. **A `+cpu` torch wheel.** Fails silently — you get a working stack that never
   uses the GPU and trains at 1% speed. The five-line check catches it.
4. **Trying to finetune a GGUF.** It cannot be done; download the safetensors.
5. **VRAM contention.** llama-server holds ~17 GB. Kill it before training.
6. **`git add` before `home-manager switch`.** Flakes ignore untracked files, so
   a new file appears to have no effect.
7. **Committing model weights.** Add `*.gguf`, `*.safetensors`, `.venv/`, and
   `results/*/checkpoints` to `.gitignore` early.

---

## 7. Repo state

`~/Projects/einsteins-mom/` is a plain directory — **not yet a git repo, and not
a clone.** The upstream `github.com/Foomax/Einsteins-Mom` currently contains only
a two-line README (one commit, 2026-06-19). Nothing here has been pushed, and
whether to push is the repo owner's call.

To connect it:

```bash
cd ~/Projects/einsteins-mom
git init && git remote add origin git@github.com:Foomax/Einsteins-Mom.git
git fetch origin && git checkout -b main --track origin/main   # keeps the README
```

Write `.gitignore` before the first commit.
