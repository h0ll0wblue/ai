# /// ---
# Zephyros-600M -- GPU pre-training runner
# ---
# Uses the official Flax NNX data-parallel pattern:
#   NamedSharding + jax.device_put + @nnx.jit
# Requires HF_TOKEN env var (set as Kaggle Secret).
# Checkpoints save to HuggingFace Hub (private repo).
# ///

import os, sys, time, gc
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")

import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec
import optax
from flax import nnx
from huggingface_hub import HfApi, hf_hub_download, create_repo

GPUS = jax.devices()
N_GPUS = len(GPUS)
GPU_KIND = GPUS[0].device_kind if N_GPUS > 0 else "cpu"

if not any(d.platform == "gpu" for d in GPUS):
    sys.exit(f"No GPU found ({N_GPUS} x {GPU_KIND})")

N_CHIPS = N_GPUS
MICRO_BATCH_PER_CHIP = 1
GRAD_ACCUM_STEPS = 128

device_mesh = Mesh(mesh_utils.create_device_mesh((N_GPUS,)), ("data",))
model_sharding = NamedSharding(device_mesh, PartitionSpec())
data_sharding = NamedSharding(device_mesh, PartitionSpec("data"))

print(f"Devices: {N_GPUS} x {GPU_KIND}")
print(f"Config: microBatch=1, nChips={N_CHIPS}, gradAccum={GRAD_ACCUM_STEPS}")

# ── Config ──────────────────────────────────────────────────────────────────

MODEL_CONFIG = {
    "vocabSize": 49152, "dModel": 1280, "dFF": 4800, "nLayers": 24,
    "nQueryHeads": 20, "nKVHeads": 4, "headDim": 64, "maxSeqLen": 4096,
    "ropeTheta": 500000.0, "rmsNormEps": 1e-6, "tieEmbeddings": True,
}
TRAINING_CONFIG = {
    "totalTokens": 12_000_000_000, "seqLen": 4096,
    "optimizer": {"name": "adamw", "beta1": 0.9, "beta2": 0.95,
                  "epsilon": 1e-8, "weightDecay": 0.01, "gradClipNorm": 1.0},
    "schedule": {"peakLR": 3e-4, "minLR": 3e-5,
                 "warmupFraction": 0.01, "decay": "cosine"},
    "checkpoint": {"intervalMinutes": 15,
                   "weightsRepo": "h0ll0wpurple/zephyros-600m"},
}
SEQ_LEN = TRAINING_CONFIG["seqLen"]
MICRO_BATCH_SIZE = MICRO_BATCH_PER_CHIP * N_CHIPS

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
    print(f"  [OK] Tokenizer (vocab {tokenizer.get_vocab_size()})")
except Exception as e:
    sys.exit(f"Tokenizer: {e}")

import flax
print(f"  [OK] jax={jax.__version__} flax={flax.__version__} optax={optax.__version__}")

HF_API = HfApi(token=hfToken)

# ── Training Setup ──────────────────────────────────────────────────────────

nTokensPerStep = MICRO_BATCH_PER_CHIP * SEQ_LEN * N_CHIPS * GRAD_ACCUM_STEPS
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

# ── Data Pipeline (no model on GPU yet) ─────────────────────────────────────

from datasets import load_dataset, interleave_datasets

print("\n[data] Loading datasets...")
t0 = time.time()
allDs = []
for name, split, weight in DATASET_MIX:
    split = split or "train"
    print(f"  {name} ({split})...", end=" ", flush=True)
    try:
        ds = load_dataset(name, split=split, streaming=True)
        key = "text" if "text" in ds.features else list(ds.features.keys())[0]
        allDs.append((ds, key, weight))
        print(f"OK  ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"SKIP  {e}")

if not allDs:
    raise RuntimeError("No datasets loaded")

probs = [w / sum(w for _, _, w in allDs) for _, _, w in allDs]
combined = interleave_datasets([d for d, _, _ in allDs], probabilities=probs)
combined = combined.shuffle(buffer_size=1000, seed=42)
dIter = iter(combined)

print(f"[data] Pre-buffering...", end=" ", flush=True)
buf = []
while len(buf) < SEQ_LEN * 2:
    ex = next(dIter)
    for _, key, _ in allDs:
        if key in ex:
            t = ex[key]
            break
    else:
        t = ex.get("text") or ex.get("content") or list(ex.values())[0]
    if isinstance(t, str):
        buf.extend(tokenizer.encode(t).ids)
print(f"{len(buf)} tokens ({time.time()-t0:.1f}s)")

def dataGen():
    global buf, dIter
    while True:
        if len(buf) < SEQ_LEN:
            ex = next(dIter)
            for _, key, _ in allDs:
                if key in ex:
                    t = ex[key]
                    break
            else:
                t = ex.get("text") or ex.get("content") or list(ex.values())[0]
            if isinstance(t, str):
                buf.extend(tokenizer.encode(t).ids)
        if len(buf) >= SEQ_LEN:
            seq = buf[:SEQ_LEN]
            buf = buf[SEQ_LEN:]
            yield {
                "inputIds": jnp.array(seq, dtype=jnp.int32),
                "targetIds": jnp.array(seq[1:] + [0], dtype=jnp.int32),
            }

dataIter = dataGen()
print(f"[data] Ready ({time.time()-t0:.1f}s)\n")

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
state = nnx.state((model, optimizer))
state = jax.device_put(state, model_sharding)
nnx.update((model, optimizer), state)
print(f"[model] Created + replicated ({time.time()-t0:.1f}s)")

# ── Checkpoint Resume ───────────────────────────────────────────────────────

def tryResume():
    gs, ts = 0, 0
    try:
        p = hf_hub_download(REPO_ID, CKPT_FILE, token=hfToken)
        gs, ts = deserializeCheckpoint(model, optimizer, Path(p).read_bytes())
        state = nnx.state((model, optimizer))
        state = jax.device_put(state, model_sharding)
        nnx.update((model, optimizer), state)
        print(f"  Resumed step {gs} ({ts:,} tokens)")
    except Exception as e:
        print(f"  Fresh start ({e})")
    return gs, ts

def saveCheckpoint(gs, ts):
    d = Path("/kaggle/working/checkpoints")
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"checkpoint-{gs}.msgpack"
    try:
        state = nnx.state((model, optimizer))
        state = jax.device_get(state)
        nnx.update((model, optimizer), state)
        b = serializeCheckpoint(model, optimizer, gs, ts)
        p.write_bytes(b)
        nb = len(b)
        del b
        print(f"  Saved local ({p}, {nb//1024**2} MB)")
        HF_API.upload_file(path_or_fileobj=str(p), path_in_repo=CKPT_FILE, repo_id=REPO_ID)
        print(f"  Uploaded to Hub step {gs}")
    except Exception as e:
        print(f"  [WARN] Save failed: {e}")

globalStep, tokensSeen = tryResume()

# ── JIT-Compiled Micro-Step ─────────────────────────────────────────────────

@nnx.remat
def lossFn(m, inputIds, positions, targetIds):
    lg = m(inputIds, positions, enableDropout=False)
    return optax.softmax_cross_entropy_with_integer_labels(lg, targetIds).mean()

@nnx.jit
def microStep(model, inputIds, positions, targetIds):
    return nnx.value_and_grad(lossFn)(model, inputIds, positions, targetIds)

# ── Training Loop ───────────────────────────────────────────────────────────

SESH_DURATION = 28800
CKPT_INTERVAL = 900

print(f"\n{'='*50}")
print(f"Session: {SESH_DURATION}s  CKPT: {CKPT_INTERVAL}s")
print(f"Resume: step {globalStep}  tokens {tokensSeen:,}")
print(f"Micro-batch: {MICRO_BATCH_SIZE} seq ({MICRO_BATCH_SIZE*SEQ_LEN:,} tok)")
print(f"{'='*50}\n")

step = globalStep
totalTokens = tokensSeen
startTime = lastCkptTime = time.time()
sessionFailed = False
nSamples = 0

try:
    while time.time() - startTime < SESH_DURATION:
        gradAccum = None
        lossAccum = 0.0

        for _ in range(GRAD_ACCUM_STEPS):
            inpSeq = []
            tgtSeq = []
            for _ in range(MICRO_BATCH_SIZE):
                d = next(dataIter)
                inpSeq.append(d["inputIds"])
                tgtSeq.append(d["targetIds"])

            inputIds = jnp.stack(inpSeq)
            positions = jnp.broadcast_to(
                jnp.arange(SEQ_LEN, dtype=jnp.int32),
                (MICRO_BATCH_SIZE, SEQ_LEN),
            )
            targetIds = jnp.stack(tgtSeq)

            inputIds, positions, targetIds = jax.device_put(
                (inputIds, positions, targetIds), data_sharding
            )

            loss, grads = microStep(model, inputIds, positions, targetIds)

            gradAccum = grads if gradAccum is None else jax.tree.map(jnp.add, gradAccum, grads)
            lossAccum += float(loss)

        avgGrads = jax.tree.map(lambda g: g / GRAD_ACCUM_STEPS, gradAccum)
        optimizer.update(model, avgGrads)

        avgLoss = lossAccum / GRAD_ACCUM_STEPS
        totalTokens += nTokensPerStep
        step += 1
        nSamples += 1
        lr = float(lrSchedule(step))

        if jnp.isnan(avgLoss) or jnp.isinf(avgLoss):
            raise RuntimeError(f"Diverged at step {step}: {avgLoss}")

        e = time.time() - startTime
        tps = (totalTokens - tokensSeen) / e if e > 0 else 0

        print(f"  step {step:6d} | loss {avgLoss:.4f} | lr {lr:.2e} | tok/s {tps:,.0f} | {int(e)}s")

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
