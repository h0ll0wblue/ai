# /// ---
# Zephyros-600M -- TPU pre-training runner
# ---
# Run this as a notebook with a TPU accelerator (8 chips expected).
# Each session trains for 1 hour with checkpoints every 15 minutes.
# Aborts if TPU is not available.
#
# Requires HF_TOKEN env var (set as platform Secret or local .env).
# Checkpoints save to HuggingFace Hub (private repo, 100GB free).
# ///

import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from huggingface_hub import HfApi, hf_hub_download, create_repo

# ── Device Detection ───────────────────────────────────────────────────────────

jaxDevices = jax.devices()
nDevices = len(jaxDevices)
deviceKind = jaxDevices[0].device_kind if nDevices > 0 else "cpu"
isTpu = any(d.platform == "tpu" for d in jaxDevices)

print(f"Devices: {nDevices} x {deviceKind}")

if not isTpu or nDevices < 8:
    sys.exit(
        f"TPU (8 chips) not available (found {nDevices} x {deviceKind})."
    )

N_CHIPS = nDevices

# ── Distributed Setup ─────────────────────────────────────────────────────────

from jax.sharding import Mesh, PartitionSpec

mesh = Mesh(jaxDevices, ("devices",))
distCtx = nnx.Distributed(mesh, PartitionSpec())
distCtx.__enter__()

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_CONFIG = {
    "vocabSize": 49152,
    "dModel": 1280,
    "dFF": 4800,
    "nLayers": 24,
    "nQueryHeads": 20,
    "nKVHeads": 4,
    "headDim": 64,
    "maxSeqLen": 4096,
    "ropeTheta": 500000.0,
    "rmsNormEps": 1e-6,
    "tieEmbeddings": True,
}

TRAINING_CONFIG = {
    "totalTokens": 12_000_000_000,
    "seqLen": 4096,
    "microBatchPerChip": 4,
    "nChips": 8,
    "gradAccumSteps": 8,
    "optimizer": {
        "name": "adamw",
        "beta1": 0.9,
        "beta2": 0.95,
        "epsilon": 1e-8,
        "weightDecay": 0.01,
        "gradClipNorm": 1.0,
    },
    "schedule": {
        "peakLR": 3e-4,
        "minLR": 3e-5,
        "warmupFraction": 0.01,
        "decay": "cosine",
    },
    "checkpoint": {
        "intervalMinutes": 15,
        "weightsRepo": "zephyros-600m",
    },
}

MICRO_BATCH_PER_CHIP = TRAINING_CONFIG["microBatchPerChip"]
GRAD_ACCUM_STEPS = TRAINING_CONFIG["gradAccumSteps"]
SEQ_LEN = TRAINING_CONFIG["seqLen"]
MICRO_BATCH_SIZE = MICRO_BATCH_PER_CHIP * N_CHIPS

class ModelConfig:
    def __init__(self, data):
        for key, value in data.items():
            setattr(self, key, value)

modelConfig = ModelConfig(MODEL_CONFIG)

# ── Tokenizer ──────────────────────────────────────────────────────────────────

from tokenizers import Tokenizer

TOKENIZER_REPO = "zephyros-600m"

def loadTokenizer():
    try:
        path = hf_hub_download(TOKENIZER_REPO, "tokenizer.json")
        return Tokenizer.from_file(path)
    except Exception:
        localPath = "/kaggle/input/zephyros-tokenizer/tokenizer.json"
        if os.path.exists(localPath):
            return Tokenizer.from_file(localPath)
        raise RuntimeError("Tokenizer not found. Run buildTokenizer.py first.")

tokenizer = loadTokenizer()

# ── Model ──────────────────────────────────────────────────────────────────────

from src.model import DecoderOnlyLM
from src.checkpoint import serializeCheckpoint, deserializeCheckpoint

model = DecoderOnlyLM(modelConfig, rngs=nnx.Rngs(0))

# ── Checkpoint Resume ─────────────────────────────────────────────────────────

HF_TOKEN = os.environ.get("HF_TOKEN", "")
REPO_ID = TRAINING_CONFIG["checkpoint"]["weightsRepo"]
CKPT_FILE = "checkpoint.msgpack"

if HF_TOKEN:
    create_repo(REPO_ID, exist_ok=True, token=HF_TOKEN, private=True)
    HF_API = HfApi(token=HF_TOKEN)
else:
    HF_API = HfApi()

def tryResume(model, optimizer):
    globalStep = 0
    tokensSeen = 0
    try:
        ckptPath = hf_hub_download(REPO_ID, CKPT_FILE, token=HF_TOKEN or None)
        ckptBytes = Path(ckptPath).read_bytes()
        globalStep, tokensSeen = deserializeCheckpoint(model, optimizer, ckptBytes)
        print(f"  Loaded checkpoint step {globalStep} ({tokensSeen:,} tokens)")
    except Exception as e:
        print(f"  No checkpoint found, starting from scratch ({e})")
    return model, globalStep, tokensSeen


def saveCheckpoint(model, optimizer, globalStep, tokensSeen):
    ckptBytes = serializeCheckpoint(model, optimizer, globalStep, tokensSeen)
    try:
        HF_API.upload_file(
            path_or_fileobj=ckptBytes,
            path_in_repo=CKPT_FILE,
            repo_id=REPO_ID,
        )
        print(f"  Checkpoint saved to Hub (step {globalStep})")
    except Exception as e:
        print(f"  [WARN] Hub upload failed: {e}")
    localDirObj = Path("/kaggle/working/checkpoints")
    localDirObj.mkdir(parents=True, exist_ok=True)
    localPath = localDirObj / f"checkpoint-{globalStep}.msgpack"
    try:
        localPath.write_bytes(ckptBytes)
        print(f"  Local backup saved ({localPath})")
    except Exception as e:
        print(f"  [WARN] Local save failed: {e}")


# ── Training Setup ────────────────────────────────────────────────────────────

nTokensPerStep = (
    MICRO_BATCH_PER_CHIP * SEQ_LEN * N_CHIPS * GRAD_ACCUM_STEPS
)
nSteps = TRAINING_CONFIG["totalTokens"] // nTokensPerStep
nWarmupSteps = max(1, int(nSteps * TRAINING_CONFIG["schedule"]["warmupFraction"]))
nDecaySteps = nSteps - nWarmupSteps

print(f"Tokens per step: {nTokensPerStep:,}")
print(f"Total steps: {nSteps:,}")
print(f"Warmup steps: {nWarmupSteps}")

lrSchedule = optax.warmup_cosine_decay_schedule(
    init_value=TRAINING_CONFIG["schedule"]["minLR"],
    peak_value=TRAINING_CONFIG["schedule"]["peakLR"],
    warmup_steps=nWarmupSteps,
    decay_steps=nDecaySteps,
    end_value=TRAINING_CONFIG["schedule"]["minLR"],
)

optimizer = nnx.Optimizer(
    model,
    optax.chain(
        optax.clip_by_global_norm(TRAINING_CONFIG["optimizer"]["gradClipNorm"]),
        optax.adamw(
            learning_rate=lrSchedule,
            b1=TRAINING_CONFIG["optimizer"]["beta1"],
            b2=TRAINING_CONFIG["optimizer"]["beta2"],
            eps=TRAINING_CONFIG["optimizer"]["epsilon"],
            weight_decay=TRAINING_CONFIG["optimizer"]["weightDecay"],
        ),
    ),
    wrt=nnx.Param,
)

model, globalStep, tokensSeen = tryResume(model, optimizer)

# ── Gradient-Checkpointed Loss ────────────────────────────────────────────────

@nnx.remat
def computeLoss(model, batch):
    logits = model(batch["inputIds"], batch["positions"], enableDropout=True)
    return optax.softmax_cross_entropy_with_integer_labels(
        logits, batch["targetIds"]
    ).mean()

# ── JIT-Wrapped Micro-Batch Step ──────────────────────────────────────────────

@nnx.jit
def microBatchStep(model, batch):
    return nnx.value_and_grad(computeLoss)(model, batch)

# ── Data Pipeline ──────────────────────────────────────────────────────────────

from datasets import load_dataset, interleave_datasets

DATASET_MIX: list[tuple[str, str | None, float]] = [
    ("HuggingFaceTB/fineweb-edu", "train", 0.60),
    ("emozilla/pg19", "train", 0.15),
    ("bookcorpus2", None, 0.10),
    ("AI-MO/NuminaMath-CoT", None, 0.05),
    ("open-web-math/open-web-math", "train", 0.05),
    ("wikipedia", "20231101.en", 0.03),
    ("CShorten/arxiv-abs", "train", 0.02),
]

def tokenizeFn(examples, textKey="text"):
    textCol = examples.get(textKey) or examples.get("content") or []
    if not textCol:
        n = len(next(iter(examples.values()), []))
        return {"inputIds": [[] for _ in range(n)]}
    tokens = [tokenizer.encode(t).ids for t in textCol]
    return {"inputIds": tokens}

def makeDataPipeline(sessionSeed: int = 42):
    loaded: list = []
    loadedWeights: list[float] = []
    for name, split, weight in DATASET_MIX:
        try:
            ds = load_dataset(name, split=split or "train", streaming=True)
        except Exception as e:
            print(f"  [SKIP] {name}: {e}")
            continue
        dsFeatures = list(ds.features.keys())
        textKey = "text" if "text" in dsFeatures else ("content" if "content" in dsFeatures else dsFeatures[0])
        ds = ds.map(
            lambda x, tk=textKey: tokenizeFn(x, tk),
            remove_columns=dsFeatures,
            batched=True,
            batch_size=100,
        )
        loaded.append(ds)
        loadedWeights.append(weight)

    if not loaded:
        raise RuntimeError("No datasets could be loaded.")

    totalW = sum(loadedWeights)
    probs = [w / totalW for w in loadedWeights]
    combined = interleave_datasets(loaded, probabilities=probs)
    combined = combined.shuffle(buffer_size=10_000, seed=sessionSeed)

    def packAndBatch(iterator):
        buffer = []
        for example in iterator:
            buffer.extend(example["inputIds"])
            while len(buffer) >= SEQ_LEN:
                seq = buffer[:SEQ_LEN]
                buffer = buffer[SEQ_LEN:]
                inputIds = jnp.array(seq, dtype=jnp.int32)
                targetIds = jnp.concatenate([
                    inputIds[1:],
                    jnp.zeros((1,), dtype=jnp.int32),
                ])
                yield {"inputIds": inputIds, "targetIds": targetIds}

    return packAndBatch(combined)

# ── Main Training Loop ────────────────────────────────────────────────────────

# Kaggle sessions have a 9-hour limit; run for 8h to leave time for final checkpoint.
SESH_DURATION = 28800
CKPT_INTERVAL = 900

print(f"\n{'='*60}")
print("Starting training session")
print(f"  Session duration: {SESH_DURATION}s")
print(f"  Checkpoint interval: {CKPT_INTERVAL}s")
print(f"  Resume from step: {globalStep} ({tokensSeen:,} tokens)")
print(f"  Micro-batch size: {MICRO_BATCH_SIZE} sequences ({MICRO_BATCH_SIZE * SEQ_LEN:,} tokens)")
print(f"{'='*60}\n")

sampleStep = 0
step = globalStep
totalTokens = tokensSeen
startTime = lastCkptTime = time.time()
sessionFailed = False
dataIter = None

try:
    dataIter = makeDataPipeline(sessionSeed=globalStep + 42)

    while time.time() - startTime < SESH_DURATION:
        gradsAccum = None
        accumLoss = 0.0

        for _ in range(GRAD_ACCUM_STEPS):
            batchInputs = []
            batchTargets = []
            for _ in range(MICRO_BATCH_SIZE):
                data = next(dataIter)
                batchInputs.append(data["inputIds"])
                batchTargets.append(data["targetIds"])

            batch = {
                "inputIds": jnp.stack(batchInputs),
                "positions": jnp.broadcast_to(
                    jnp.arange(SEQ_LEN, dtype=jnp.int32),
                    (MICRO_BATCH_SIZE, SEQ_LEN),
                ),
                "targetIds": jnp.stack(batchTargets),
            }

            loss, grads = microBatchStep(model, batch)

            if gradsAccum is None:
                gradsAccum = grads
            else:
                gradsAccum = jax.tree.map(jnp.add, gradsAccum, grads)

            accumLoss += float(loss)

        avgGrads = jax.tree.map(lambda g: g / GRAD_ACCUM_STEPS, gradsAccum)
        optimizer.update(model, avgGrads)

        avgLoss = accumLoss / GRAD_ACCUM_STEPS
        totalTokens += nTokensPerStep
        step += 1
        sampleStep += 1
        currentLR = float(lrSchedule(step))

        if jnp.isnan(avgLoss) or jnp.isinf(avgLoss):
            raise RuntimeError(f"Loss diverged at step {step}: {avgLoss}")

        elapsed = time.time() - startTime
        tokensThisSession = totalTokens - tokensSeen
        tokensPerSec = tokensThisSession / elapsed if elapsed > 0 else 0

        print(
            f"  step {step:6d} | "
            f"loss {avgLoss:.4f} | "
            f"lr {currentLR:.2e} | "
            f"tok/s {tokensPerSec:,.0f} | "
            f"elapsed {int(elapsed)}s"
        )

        if time.time() - lastCkptTime >= CKPT_INTERVAL:
            saveCheckpoint(model, optimizer, step, totalTokens)
            lastCkptTime = time.time()

except Exception as e:
    print(f"\n[WARN] Session interrupted: {e}")
    sessionFailed = True
    raise

finally:
    try:
        saveCheckpoint(model, optimizer, step, totalTokens)
    except Exception as e:
        print(f"  [WARN] Final checkpoint save failed: {e}")
    finalElapsed = time.time() - startTime
    finalTokens = totalTokens - tokensSeen
    print(f"\n{'='*60}")
    print(f"Session {'interrupted' if sessionFailed else 'complete'}")
    print(f"  Steps this session: {sampleStep}")
    print(f"  Tokens this session: {finalTokens:,}")
    print(f"  Total tokens: {totalTokens:,}")
    print(f"  Elapsed: {finalElapsed:.0f}s")
    if finalElapsed > 0:
        print(f"  Average tok/s: {finalTokens / finalElapsed:,.0f}")
    print(f"{'='*60}")

    distCtx.__exit__(None, None, None)
