# /// ---
# Zephyros-600M -- GPU pre-training runner (single/dual GPU)
# ---
# Uses official Flax NNX data-parallel pattern:
#   jax.device_put + NamedSharding + @nnx.jit
#
# Kaggle setup (run in a cell BEFORE this script):
#   !pip install --upgrade "jax[cuda12]" flax optax -q
#
# Requires HF_TOKEN env var. Checkpoints to HuggingFace Hub.
# ///
#
# NOTE: This file is intended to be uploaded to the Kaggle notebook and
# run with:  !python gpuTrain.py
# Set JAX memory variables before importing JAX to disable 90% pre-allocation,
# allowing the XLA compiler to use system VRAM dynamically without OOMing.
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

# Set Hugging Face cache directories to Kaggle disk instead of RAM (tmpfs /root)
os.environ["HF_HOME"] = "/kaggle/working/.cache/huggingface"
os.environ["HF_DATASETS_CACHE"] = "/kaggle/working/.cache/huggingface"

# Enable bf16 matmul precision for Tensor Cores on T4 (65 TFLOPS vs 8 TFLOPS fp32).
# This disables the manual float16 casts in layers.py (which risk overflow in q·k
# einsum) and instead rounds inputs to bf16 with fp32 accumulation, safe from overflow.
os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "bfloat16"

# Cache XLA compiled binaries across Kaggle sessions, saving ~60s recompilation
# on resume. Persists on /kaggle/working disk.
os.environ["JAX_COMPILATION_CACHE_DIR"] = "/kaggle/working/.cache/jax"

# XLA GPU flags: fuse softmax with attention matmul via Triton, overlap compute
# with memory via latency-hiding scheduler, enable async collectives for multi-GPU.
os.environ["XLA_FLAGS"] = (
    "--xla_gpu_enable_triton_softmax_fusion=true "
    "--xla_gpu_enable_latency_hiding_scheduler=true "
    "--xla_gpu_enable_async_all_gather=true "
    "--xla_gpu_all_reduce_combine_threshold_bytes=1073741824"
)


import sys
import time
import multiprocessing as mp
import gc
from pathlib import Path

import jax
jax.config.update("jax_default_matmul_precision", "bfloat16")
import jax.numpy as jnp
import numpy as np
import optax
import flax
from flax import nnx
from huggingface_hub import HfApi, hf_hub_download, create_repo

# ── GPU Discovery ─────────────────────────────────────────────────────────────
# Query BEFORE any JAX computation so all devices are initialised.
GPUS = jax.devices("gpu") if any(d.platform == "gpu" for d in jax.devices()) else []
N_GPUS = len(GPUS)
GPU_KIND = GPUS[0].device_kind if N_GPUS > 0 else "cpu"

if N_GPUS == 0:
    sys.exit(f"No GPU found. Devices: {jax.devices()}")

# Use ALL available GPUs. With bf16 + on-device accum the replicated
# model+optimizer (~7.2 GB) fits in 15 GB alongside activations.
N_CHIPS = N_GPUS
ACTIVE_GPUS = GPUS[:N_CHIPS]
print(
    f"JAX: {jax.__version__}  Flax: {flax.__version__}  "
    f"Visible: {N_GPUS} x {GPU_KIND}  Using: {N_CHIPS} x {GPU_KIND}"
)

# Data-parallel mesh: replicate everything, shard the batch dimension
mesh = jax.sharding.Mesh(jax.devices(), axis_names=("batch",))
DATA_SHARDED = jax.sharding.NamedSharding(
    mesh, jax.sharding.PartitionSpec("batch")
)

# ── Config ──────────────────────────────────────────────────────────────────

MODEL_CONFIG = {
    "vocabSize": 49152, "dModel": 1280, "dFF": 4800, "nLayers": 24,
    "nQueryHeads": 20, "nKVHeads": 4, "headDim": 64, "maxSeqLen": 2048,
    "ropeTheta": 500000.0, "rmsNormEps": 1e-6, "tieEmbeddings": True,
}
TRAINING_CONFIG = {
    "totalTokens": 12_000_000_000, "seqLen": 2048,
    "optimizer": {"name": "adamw", "beta1": 0.9, "beta2": 0.95,
                  "epsilon": 1e-8, "weightDecay": 0.01, "gradClipNorm": 1.0},
    "schedule": {"peakLR": 3e-4, "minLR": 3e-5,
                 "warmupFraction": 0.01, "decay": "cosine"},
    "checkpoint": {"intervalMinutes": 15,
                   "weightsRepo": "h0ll0wpurple/zephyros-600m"},
}

SEQ_LEN = TRAINING_CONFIG["seqLen"]

# With bf16 + on-device accum + remat, 4 seq/GPU fits in 15 GB.
# 4 seq × N_GPUS × 2048 × 32 accum = 524 288 tok/step for 2× T4.
MICRO_BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 32

print(f"Config: microBatch={MICRO_BATCH_SIZE}, nChips={N_CHIPS}, gradAccum={GRAD_ACCUM_STEPS}")

class ModelConfig:
    def __init__(self, data):
        for k, v in data.items():
            setattr(self, k, v)
modelConfig = ModelConfig(MODEL_CONFIG)

REPO_ID = TRAINING_CONFIG["checkpoint"]["weightsRepo"]
CKPT_FILE = "checkpoint.msgpack"

DATASET_MIX: list[tuple[str, str | None, float]] = [
    ("HuggingFaceFW/fineweb", "train", 0.65),
    ("emozilla/pg19", "train", 0.20),
    ("AI-MO/NuminaMath-CoT", None, 0.07),
    ("open-web-math/open-web-math", "train", 0.05),
]



# ── Quick Checks ────────────────────────────────────────────────────────────

print("Running checks...")
hfToken = os.environ.get("HF_TOKEN", "")
if not hfToken:
    sys.exit("HF_TOKEN not set")
print("  [OK] HF_TOKEN found")

try:
    create_repo(REPO_ID, exist_ok=True, token=hfToken, private=True)
    print(f"  [OK] Hub repo {REPO_ID}")
except Exception as e:
    sys.exit(f"Hub: {e}")

try:
    path = hf_hub_download(REPO_ID, "tokenizer.json", token=hfToken)
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(path)
    # Determine EOS token ID for document separation in packing
    EOS_ID = 0
    for eos_candidate in ["<|endoftext|>", "[EOS]", "</s>", "<eos>", "[eos]"]:
        tid = tokenizer.token_to_id(eos_candidate)
        if tid is not None:
            EOS_ID = tid
            break
    print(f"  [OK] Tokenizer (vocab {tokenizer.get_vocab_size()}, EOS={EOS_ID})")
except Exception as e:
    sys.exit(f"Tokenizer: {e}")

print(f"  [OK] jax={jax.__version__} flax={flax.__version__} optax={optax.__version__}")

HF_API = HfApi(token=hfToken)

# ── Training Setup ──────────────────────────────────────────────────────────

nTokensPerStep = MICRO_BATCH_SIZE * N_GPUS * SEQ_LEN * GRAD_ACCUM_STEPS
nTotalSteps = TRAINING_CONFIG["totalTokens"] // nTokensPerStep
nWarmupSteps = max(1, int(nTotalSteps * TRAINING_CONFIG["schedule"]["warmupFraction"]))
nDecaySteps = nTotalSteps - nWarmupSteps
print(f"Steps: {nTotalSteps} ({nTokensPerStep:,} tok/step, warmup {nWarmupSteps})")

lrSchedule = optax.warmup_cosine_decay_schedule(
    TRAINING_CONFIG["schedule"]["minLR"],
    TRAINING_CONFIG["schedule"]["peakLR"],
    nWarmupSteps, nDecaySteps,
    TRAINING_CONFIG["schedule"]["minLR"],
)

# ── Data Pipeline ────────────────────────────────────────────────────────────
# Tokenisation runs in a background thread so the GPU is never starved.
# The training loop pops pre-built numpy arrays from a thread-safe queue.

from datasets import load_dataset, interleave_datasets

print("\n[data] Loading datasets...")
t0 = time.time()
allDs = []
for name, split, weight in DATASET_MIX:
    split = split or "train"
    print(f"  {name} ({split})...", end=" ", flush=True)
    try:
        ds  = load_dataset(name, split=split, streaming=True)
        key = "text" if "text" in ds.features else list(ds.features.keys())[0]
        allDs.append((ds, key, weight))
        print(f"OK  ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"SKIP  {e}")

if not allDs:
    raise RuntimeError("No datasets loaded")

probs    = [w / sum(w for _, _, w in allDs) for _, _, w in allDs]
combined = interleave_datasets([d for d, _, _ in allDs], probabilities=probs)
dIter    = iter(combined)

# ── Background tokeniser processes ────────────────────────────────────────────
# Multiple processes keep the GPU fed. Each has its own iterator so they
# consume the dataset mix independently. On Linux (Kaggle), fork handles
# the global state (tokenizer, allDs, combined) without pickling overhead.

_N_TOK_WORKERS = 4
_DATA_QUEUE: mp.Queue = mp.Queue(maxsize=1000)

def _tokeniser_worker_proc(worker_id: int):
    """Process worker: pull from dataset mix, tokenize, push to shared queue."""
    buf: list[int] = []
    local_iter = iter(combined)
    while True:
        try:
            while len(buf) < SEQ_LEN + 1:
                try:
                    ex = next(local_iter)
                except StopIteration:
                    local_iter = iter(combined)
                    ex = next(local_iter)

                text = None
                for _, key, _ in allDs:
                    if key in ex:
                        text = ex[key]
                        break
                if text is None:
                    text = ex.get("text") or ex.get("content") or list(ex.values())[0]
                if isinstance(text, str) and text:
                    if buf:
                        buf.append(EOS_ID)
                    buf.extend(tokenizer.encode(text).ids)

            seq  = buf[:SEQ_LEN + 1]
            buf  = buf[SEQ_LEN + 1:]
            inp  = np.array(seq[:SEQ_LEN],       dtype=np.int32)
            tgt  = np.array(seq[1:SEQ_LEN + 1], dtype=np.int32)
            _DATA_QUEUE.put((inp, tgt), timeout=60)
        except Exception as e:
            print(f"[tokenizer:{worker_id}] WARNING: {e}; restarting iterator", flush=True)
            local_iter = iter(combined)
            time.sleep(0.5)

_tok_processes = []
for wid in range(_N_TOK_WORKERS):
    p = mp.Process(target=_tokeniser_worker_proc, args=(wid,), daemon=True)
    p.start()
    _tok_processes.append(p)

# Pre-warm: wait until we have enough sequences to fill the first optimizer step
print(f"[data] Pre-buffering...", end=" ", flush=True)
needed = MICRO_BATCH_SIZE * GRAD_ACCUM_STEPS
while _DATA_QUEUE.qsize() < min(needed, 32):
    time.sleep(0.2)
print(f"{_DATA_QUEUE.qsize()} seqs buffered ({time.time()-t0:.1f}s)")
print(f"[data] Ready ({time.time()-t0:.1f}s)\n")

def get_batch() -> dict:
    """Pull MICRO_BATCH_SIZE sequences from the shared queue."""
    inpSeqs = []
    tgtSeqs = []
    for _ in range(MICRO_BATCH_SIZE):
        inp, tgt = _DATA_QUEUE.get(timeout=300)
        inpSeqs.append(inp)
        tgtSeqs.append(tgt)
    positions = np.broadcast_to(
        np.arange(SEQ_LEN, dtype=np.int32),
        (MICRO_BATCH_SIZE, SEQ_LEN),
    )
    return {
        "inputIds":  np.stack(inpSeqs),
        "positions": positions.copy(),
        "targetIds": np.stack(tgtSeqs),
    }

def _make_device_batch(batch: dict) -> dict:
    """Place batch onto GPU devices with data-parallel sharding."""
    return jax.device_put(batch, DATA_SHARDED)

# ── Model + Optimizer ───────────────────────────────────────────────────────

from src.model import DecoderOnlyLM
from src.checkpoint import (
    serializeCheckpoint, deserializeCheckpoint,
    has_orbax, save_checkpoint_orbax, load_checkpoint_orbax,
)

print("[model] Creating 600M model...")
t0 = time.time()
model = DecoderOnlyLM(modelConfig, rngs=nnx.Rngs(0))

# Compute weight-decay mask: exclude embeddings (2-D but not a weight),
# biases (1-D), and norm gains (1-D). Only 2-D Linear kernels get WD.
params = nnx.state(model, nnx.Param)
_wd_mask = jax.tree_util.tree_map_with_path(
    lambda path, v: (
        isinstance(v, jax.Array) and v.ndim >= 2
        and "embed" not in "/".join(str(k) for k in path).lower()
    ),
    params,
)

optimizer = nnx.Optimizer(
    model,
    optax.chain(
        optax.clip_by_global_norm(TRAINING_CONFIG["optimizer"]["gradClipNorm"]),
        optax.scale_by_adam(b1=0.9, b2=0.95, eps=1e-8, mu_dtype=jnp.bfloat16),
        optax.add_decayed_weights(
            TRAINING_CONFIG["optimizer"]["weightDecay"], mask=_wd_mask,
        ),
        optax.scale_by_learning_rate(lrSchedule),
    ),
    wrt=nnx.Param,
)
print(f"[model] Created ({time.time()-t0:.1f}s)")
gc.collect()  # free any temporary arrays from model/optimizer init

# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _dereplicate():
    """No-op for single GPU (state is already on host-accessible device)."""
    pass

def _replicate():
    """No-op for single GPU."""
    pass

# ── Checkpoint Resume ───────────────────────────────────────────────────────

def _find_latest_ckpt() -> Path | None:
    """Find the highest-step orbax checkpoint directory."""
    ckpt_dir = Path("/kaggle/working/checkpoints")
    if not ckpt_dir.exists():
        return None
    dirs = [d for d in ckpt_dir.iterdir()
            if d.is_dir() and d.name.startswith("checkpoint-")]
    if not dirs:
        return None
    return max(dirs, key=lambda d: int(d.name.split("-")[1]))

def tryResume():
    gs, ts = 0, 0
    try:
        if has_orbax():
            latest = _find_latest_ckpt()
            if latest is not None:
                gs, ts = load_checkpoint_orbax(model, optimizer, str(latest))
                print(f"  Resumed (orbax) step {gs} ({ts:,} tokens)")
                return gs, ts
        p = hf_hub_download(REPO_ID, CKPT_FILE, token=hfToken)
        _dereplicate()
        gs, ts = deserializeCheckpoint(model, optimizer, Path(p).read_bytes())
        _replicate()
        print(f"  Resumed (msgpack) step {gs} ({ts:,} tokens)")
    except Exception as e:
        print(f"  Fresh start ({e})")
    return gs, ts

def saveCheckpoint(gs, ts):
    d = Path("/kaggle/working/checkpoints")
    d.mkdir(parents=True, exist_ok=True)
    try:
        _dereplicate()
        if has_orbax():
            ckpt_dir = d / f"checkpoint-{gs}"
            save_checkpoint_orbax(model, optimizer, gs, ts, str(ckpt_dir))
            print(f"  Saved (orbax) {ckpt_dir}")
            # Convert to msgpack bytes for Hub upload (single file)
            b = serializeCheckpoint(model, optimizer, gs, ts)
        else:
            p = d / f"checkpoint-{gs}.msgpack"
            b = serializeCheckpoint(model, optimizer, gs, ts)
            p.write_bytes(b)
            nb = len(b)
            print(f"  Saved local ({p}, {nb//1024**2} MB)")
        _replicate()
        HF_API.upload_file(path_or_fileobj=b, path_in_repo=CKPT_FILE, repo_id=REPO_ID)
        del b
        print(f"  Uploaded to Hub step {gs}")
    except Exception as e:
        print(f"  [WARN] Save failed: {e}")

globalStep, tokensSeen = tryResume()

# ── JIT-Compiled Training Step ──────────────────────────────────────────────
# @nnx.remat on _loss_fn recomputes all 24 block activations during backward,
# keeping peak memory low enough for microBatch=4 per GPU.
# On-device gradient accumulation via micro_step avoids host transfers
# (the #3 bottleneck). The ONLY host sync is the final float(loss_accum).

@nnx.remat
def _loss_fn(model, batch):
    """Forward pass; returns scalar loss. @nnx.remat recomputes activations during backward."""
    logits = model(batch["inputIds"], batch["positions"], enableDropout=False)
    return optax.softmax_cross_entropy_with_integer_labels(
        logits, batch["targetIds"]
    ).mean()


@nnx.jit(donate_argnames=("model", "grad_accum"))
def micro_step_accum(model, batch, grad_accum, loss_accum):
    """One forward+backward. Grads accumulate ON DEVICE (no host transfer)."""
    loss, grads = nnx.value_and_grad(_loss_fn)(model, batch)
    grad_accum = jax.tree.map(jnp.add, grad_accum, grads)
    return grad_accum, loss_accum + loss


@nnx.jit(donate_argnames=("model", "optimizer"))
def apply_gradients(model, optimizer, avg_grads):
    """Apply pre-averaged gradients. Separated so optimizer.update runs on-device."""
    optimizer.update(model, avg_grads)


# ── Training Loop ───────────────────────────────────────────────────────────

SESH_DURATION = 28800  # 8 hours (leaves margin in 12h Kaggle limit)
CKPT_INTERVAL = 900    # 15 minutes

print(f"\n{'='*50}")
print(f"Session: {SESH_DURATION}s  CKPT: {CKPT_INTERVAL}s")
print(f"Resume: step {globalStep}  tokens {tokensSeen:,}")
print(f"Micro-batch: {MICRO_BATCH_SIZE} seq ({MICRO_BATCH_SIZE * SEQ_LEN:,} tok/micro, "
      f"{N_GPUS} GPU(s)), accum {GRAD_ACCUM_STEPS} -> {nTokensPerStep:,} tok/step")
print(f"{'='*50}\n")

step          = globalStep
totalTokens   = tokensSeen
startTime     = lastCkptTime = time.time()
sessionFailed = False
nSamples      = 0

# Build a zero-gradient pytree for the accumulator (same structure as params)
_zero_grads = jax.tree.map(lambda p: jnp.zeros_like(p), nnx.state(model, nnx.Param))

try:
    while time.time() - startTime < SESH_DURATION:
        # ── On-device gradient accumulation ────────────────────────────────
        # grads accumulate via jnp.add inside @nnx.jit (no host transfer).
        # Only float(loss_accum) at the end syncs to host.

        if step == globalStep:
            print(f"  [JIT] Compiling micro_step_accum + apply_gradients "
                  f"(first step ~60-180s)...", flush=True)

        grad_accum = _zero_grads
        loss_accum = jnp.float32(0.0)

        for _ in range(GRAD_ACCUM_STEPS):
            batch = _make_device_batch(get_batch())
            grad_accum, loss_accum = micro_step_accum(model, batch, grad_accum, loss_accum)

        avg_loss = float(loss_accum) / GRAD_ACCUM_STEPS  # single host sync

        if np.isnan(avg_loss) or np.isinf(avg_loss):
            raise RuntimeError(f"Diverged at step {step + 1}: {avg_loss}")

        avg_grads = jax.tree.map(lambda g: g / GRAD_ACCUM_STEPS, grad_accum)
        apply_gradients(model, optimizer, avg_grads)

        totalTokens += nTokensPerStep
        step        += 1
        nSamples    += 1
        lr           = float(lrSchedule(step))

        e   = time.time() - startTime
        tps = (totalTokens - tokensSeen) / e if e > 0 else 0

        print(f"  step {step:6d} | loss {avg_loss:.4f} | lr {lr:.2e} | tok/s {tps:,.0f} | {int(e)}s")

        if time.time() - lastCkptTime >= CKPT_INTERVAL:
            saveCheckpoint(step, totalTokens)
            lastCkptTime = time.time()

except Exception as e:
    print(f"\n[ERROR] {e}")
    sessionFailed = True
    raise
finally:
    try:
        saveCheckpoint(step, totalTokens)
    except Exception as e2:
        print(f"  Final save failed: {e2}")
    e = time.time() - startTime
    print(f"\n{'='*50}")
    print(f"{'INTERRUPTED' if sessionFailed else 'COMPLETE'}")
    print(f"  Steps: {nSamples}  Tokens: {totalTokens-tokensSeen:,}  Elapsed: {int(e)}s")
    if e > 0:
        print(f"  Avg tok/s: {(totalTokens-tokensSeen)//int(e):,}")
    print(f"{'='*50}")
