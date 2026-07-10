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


import sys
import time
import queue
import threading
import gc
from pathlib import Path

import jax
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

# Force single-GPU training. Data-parallel replication doubles the 7.2 GB
# model+optimizer onto EACH 15 GB T4, leaving <7.8 GB for activations +
# gradients + XLA compilation workspace — not enough.
# With 1 GPU we get the full 15 GB and avoid replication overhead entirely.
N_CHIPS = 1
ACTIVE_GPUS = GPUS[:N_CHIPS]
print(
    f"JAX: {jax.__version__}  Flax: {flax.__version__}  "
    f"Visible: {N_GPUS} x {GPU_KIND}  Using: {N_CHIPS} x {GPU_KIND}"
)

# ── Config ──────────────────────────────────────────────────────────────────

MODEL_CONFIG = {
    "vocabSize": 49152, "dModel": 1280, "dFF": 4800, "nLayers": 24,
    "nQueryHeads": 20, "nKVHeads": 4, "headDim": 64, "maxSeqLen": 2048,
    "ropeTheta": 500000.0, "rmsNormEps": 1e-6, "tieEmbeddings": True,
    "remat": True,
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

# Micro-batch: 1 sequence per step to minimize peak VRAM.
# With seqLen=2048 and 512 accumulation steps → 1,048,576 tokens per optimizer step.
MICRO_BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 512

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

# Single-GPU: no mesh or sharding needed.
print("Single GPU: no sharding")

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
    print(f"  [OK] Tokenizer (vocab {tokenizer.get_vocab_size()})")
except Exception as e:
    sys.exit(f"Tokenizer: {e}")

print(f"  [OK] jax={jax.__version__} flax={flax.__version__} optax={optax.__version__}")

HF_API = HfApi(token=hfToken)

# ── Training Setup ──────────────────────────────────────────────────────────

nTokensPerStep = MICRO_BATCH_SIZE * SEQ_LEN * GRAD_ACCUM_STEPS
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

# ── Background tokeniser thread ───────────────────────────────────────────────
# Keeps several steps of data pre-tokenised so the GPU never blocks on the CPU.

_DATA_QUEUE_SIZE = min(GRAD_ACCUM_STEPS, 256)  # cap prefetch to limit CPU RAM
_data_queue: queue.Queue = queue.Queue(maxsize=_DATA_QUEUE_SIZE)
_tok_buf: list[int] = []

def _tokeniser_worker():
    """Runs forever: tokenises documents and pushes (inp, tgt) pairs to queue."""
    global _tok_buf
    local_iter = dIter
    while True:
        # Refill buffer until we have at least SEQ_LEN+1 tokens (inp + tgt)
        while len(_tok_buf) < SEQ_LEN + 1:
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
            if isinstance(text, str):
                _tok_buf.extend(tokenizer.encode(text).ids)

        seq      = _tok_buf[:SEQ_LEN + 1]
        _tok_buf = _tok_buf[SEQ_LEN + 1:]
        # Correct next-token targets: tgt[i] = inp[i+1] (no padding hack)
        inp = np.array(seq[:SEQ_LEN],       dtype=np.int32)
        tgt = np.array(seq[1:SEQ_LEN + 1], dtype=np.int32)
        _data_queue.put((inp, tgt))  # blocks when queue is full (backpressure)

_tok_thread = threading.Thread(target=_tokeniser_worker, daemon=True)
_tok_thread.start()

# Pre-warm: wait until we have enough sequences to fill the first optimizer step
print(f"[data] Pre-buffering...", end=" ", flush=True)
needed = MICRO_BATCH_SIZE * GRAD_ACCUM_STEPS
while _data_queue.qsize() < min(needed, _DATA_QUEUE_SIZE // 2):
    time.sleep(0.2)
print(f"{_data_queue.qsize()} seqs buffered ({time.time()-t0:.1f}s)")
print(f"[data] Ready ({time.time()-t0:.1f}s)\n")

def get_batch() -> dict:
    """Pull MICRO_BATCH_SIZE sequences from the prefetch queue."""
    inpSeqs = []
    tgtSeqs = []
    for _ in range(MICRO_BATCH_SIZE):
        inp, tgt = _data_queue.get()
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

# ── Model + Optimizer ───────────────────────────────────────────────────────

from src.model import DecoderOnlyLM
from src.checkpoint import serializeCheckpoint, deserializeCheckpoint

print("[model] Creating 600M model...")
t0 = time.time()
model = DecoderOnlyLM(modelConfig, rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(
    model,
    optax.chain(
        optax.clip_by_global_norm(TRAINING_CONFIG["optimizer"]["gradClipNorm"]),
        optax.adamw(lrSchedule, b1=0.9, b2=0.95, eps=1e-8, weight_decay=0.01),
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

def tryResume():
    gs, ts = 0, 0
    try:
        p      = hf_hub_download(REPO_ID, CKPT_FILE, token=hfToken)
        _dereplicate()
        gs, ts = deserializeCheckpoint(model, optimizer, Path(p).read_bytes())
        _replicate()
        print(f"  Resumed step {gs} ({ts:,} tokens)")
    except Exception as e:
        print(f"  Fresh start ({e})")
    return gs, ts

def saveCheckpoint(gs, ts):
    d = Path("/kaggle/working/checkpoints")
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"checkpoint-{gs}.msgpack"
    try:
        _dereplicate()
        b = serializeCheckpoint(model, optimizer, gs, ts)
        p.write_bytes(b)
        nb = len(b)
        del b
        _replicate()
        print(f"  Saved local ({p}, {nb//1024**2} MB)")
        HF_API.upload_file(path_or_fileobj=str(p), path_in_repo=CKPT_FILE, repo_id=REPO_ID)
        print(f"  Uploaded to Hub step {gs}")
    except Exception as e:
        print(f"  [WARN] Save failed: {e}")

globalStep, tokensSeen = tryResume()

# ── JIT-Compiled Training Step ──────────────────────────────────────────────
# Design: two small @nnx.jit functions instead of one monolithic one.
#
# WHY NOT fori_loop / lax.scan:
#   NNX models carry stateful variables (RngCount, etc.) that live at the
#   top-level JAX trace. lax control-flow primitives open a *new* trace level
#   and cannot close over those variables → "Cannot extract graph node from
#   different trace level" crash. They also require XLA to hold activations
#   for ALL iterations simultaneously → OOM on 16 GB T4.
#
# WHAT WE DO INSTEAD:
#   • micro_step: one @nnx.jit forward+backward per micro-batch.
#     All 128 dispatches are ASYNC — JAX/XLA queues them without blocking
#     Python. jnp.add on grads is also async.
#   • apply_gradients: one @nnx.jit optimizer update.
#   • float(loss_accum) at the very end is the ONLY host sync per step.
#
# Peak VRAM: 1 micro-batch activations (freed by remat) + full grad pytree.

def _loss_fn(model, batch):
    """Forward pass; returns scalar loss. Block-level remat handles activation memory."""
    logits = model(batch["inputIds"], batch["positions"], enableDropout=False)
    return optax.softmax_cross_entropy_with_integer_labels(
        logits, batch["targetIds"]
    ).mean()

@nnx.jit
def micro_step(model, batch):
    """Grad for one micro-batch [MICRO_BATCH_SIZE, SEQ_LEN]. No optimizer update."""
    return nnx.value_and_grad(_loss_fn)(model, batch)

@nnx.jit
def apply_gradients(model, optimizer, avg_grads):
    """Apply pre-averaged gradients. Separated so optimizer.update runs on-device."""
    optimizer.update(model, avg_grads)

# ── Training Loop ───────────────────────────────────────────────────────────

SESH_DURATION = 28800  # 8 hours (leaves margin in 12h Kaggle limit)
CKPT_INTERVAL = 900    # 15 minutes

print(f"\n{'='*50}")
print(f"Session: {SESH_DURATION}s  CKPT: {CKPT_INTERVAL}s")
print(f"Resume: step {globalStep}  tokens {tokensSeen:,}")
print(f"Micro-batch: {MICRO_BATCH_SIZE} seq ({MICRO_BATCH_SIZE * SEQ_LEN:,} tok/micro), "
      f"accum {GRAD_ACCUM_STEPS} -> {nTokensPerStep:,} tok/step")
print(f"{'='*50}\n")

step          = globalStep
totalTokens   = tokensSeen
startTime     = lastCkptTime = time.time()
sessionFailed = False
nSamples      = 0

def _make_device_batch(mb: dict) -> dict:
    """Convert a numpy micro-batch dict to JAX arrays on the default GPU."""
    return {k: jnp.array(v) for k, v in mb.items()}

try:
    while time.time() - startTime < SESH_DURATION:
        # ── Gradient accumulation on Host RAM ─────────────────────────────
        # To avoid OOM, we do not accumulate gradients in GPU memory.
        # Instead, after each micro_step, we transfer the gradients to host
        # memory (CPU RAM, which has plenty of headroom: 30 GB vs 15 GB on GPU)
        # using jax.device_get, accumulate them on CPU, and delete the GPU copy.
        # This saves 2.4 GB of VRAM per GPU, preventing OOM.
        grads_accum = None
        loss_accum  = 0.0   # plain Python float

        if step == globalStep:
            print(f"  [JIT] Compiling micro_step + apply_gradients "
                  f"(first step ~60-180s)...", flush=True)

        for _ in range(GRAD_ACCUM_STEPS):
            batch = _make_device_batch(get_batch())
            loss, grads = micro_step(model, batch)
            
            # Sync scalar to CPU first (tiny, fast; avoids sharding issues).
            loss_accum += float(loss)
            
            # Transfer grads to CPU and accumulate there to save GPU memory
            grads_cpu = jax.device_get(grads)
            if grads_accum is None:
                grads_accum = grads_cpu
            else:
                # Accumulate on CPU in-place to avoid allocating new 2.4 GB arrays
                jax.tree.map(lambda x, y: np.add(x, y, out=x), grads_accum, grads_cpu)
            
            # Clean up device and host references immediately
            del grads, grads_cpu
            
        # Average on CPU in-place
        avg_grads_cpu = jax.tree.map(lambda g: np.divide(g, GRAD_ACCUM_STEPS, out=g), grads_accum)
        
        # Move avg_grads back to GPU with appropriate sharding
        avg_grads = jax.device_put(avg_grads_cpu)
            
        apply_gradients(model, optimizer, avg_grads)
        jax.block_until_ready(nnx.state(model, nnx.Param))  # ensure update lands
        
        # Clean up CPU references
        del grads_accum, avg_grads_cpu, avg_grads
        gc.collect()

        avg_loss = loss_accum / GRAD_ACCUM_STEPS

        if np.isnan(avg_loss) or np.isinf(avg_loss):
            raise RuntimeError(f"Diverged at step {step + 1}: {avg_loss}")

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
