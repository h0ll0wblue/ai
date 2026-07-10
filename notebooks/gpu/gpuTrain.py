# /// ---
# Zephyros-600M -- GPU pre-training runner
# ---
# Run this as a notebook with a GPU accelerator (T4x2 or P100).
# Runs for the full Kaggle session (9h limit) with checkpoints every 15 min.
# Aborts if no GPU is detected.
#
# Tokens per optimizer step (auto-detected)
#   single GPU 16GB (P100)   microBatch=1, gradAccumSteps=128 -> 524K tok/step
#   dual   GPU 16GB (T4x2)   microBatch=1, gradAccumSteps=128 -> 1.0M tok/step
#
# Requires HF_TOKEN env var (set as Kaggle Secret).
# Checkpoints save to HuggingFace Hub (private repo).
# ///

import os
import sys
import time
import gc
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.80")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ["XLA_FLAGS"] = (
    "--xla_dump_to=/dev/null "
    "--xla_gpu_enable_triton_gemm=false "
    "--xla_gpu_enable_cudnn_fmha=false"
)

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import optax
from flax import nnx
from huggingface_hub import HfApi, hf_hub_download, create_repo

# ── Device Detection & Mesh Setup ─────────────────────────────────────────────

devices = jax.devices()
nDevices = len(devices)
deviceKind = devices[0].device_kind if nDevices > 0 else "cpu"
isGpu = any(d.platform == "gpu" for d in devices)

if not isGpu:
    sys.exit(
        f"GPU not available (found {nDevices} x {deviceKind})."
    )

# Auto-detect: use all available GPUs for data parallelism
N_CHIPS = nDevices
MICRO_BATCH_PER_CHIP = 1
GRAD_ACCUM_STEPS = 128  # Keeps stable: 2 GPUs -> 2x tokens/step -> 1.0M tok/step

# Set up device mesh for data parallelism
mesh = Mesh(jax.devices(), ("data",))
nnx.spmd.set_mesh(mesh)

print(f"Devices: {nDevices} x {deviceKind}")
print(f"Config: microBatch=1, nChips={N_CHIPS}, gradAccum={GRAD_ACCUM_STEPS}")

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
        "weightsRepo": "h0ll0wpurple/zephyros-600m",
    },
}

SEQ_LEN = TRAINING_CONFIG["seqLen"]
MICRO_BATCH_SIZE = MICRO_BATCH_PER_CHIP * N_CHIPS

class ModelConfig:
    def __init__(self, data):
        for key, value in data.items():
            setattr(self, key, value)

modelConfig = ModelConfig(MODEL_CONFIG)

REPO_ID = TRAINING_CONFIG["checkpoint"]["weightsRepo"]
CKPT_FILE = "checkpoint.msgpack"

DATASET_MIX: list[tuple[str, str | None, float]] = [
    ("HuggingFaceFW/fineweb", "train", 0.65),
    ("emozilla/pg19", "train", 0.20),
    ("AI-MO/NuminaMath-CoT", None, 0.07),
    ("open-web-math/open-web-math", "train", 0.05),
    ("HuggingFaceTB/smollm-corpus", "corpus", 0.03),
]

# ── Preflight Checks ──────────────────────────────────────────────────────────

def preflightCheck():
    errors: list[str] = []
    warnings: list[str] = []
    print("Running preflight checks...")

    # 1. HF_TOKEN
    hfToken = os.environ.get("HF_TOKEN", "")
    if not hfToken:
        errors.append("HF_TOKEN not set — add it as a Kaggle Secret")
    else:
        print("  [OK] HF_TOKEN found")

    # 2. HuggingFace Hub access
    try:
        create_repo(REPO_ID, exist_ok=True, token=hfToken, private=True)
        print(f"  [OK] HuggingFace Hub accessible (repo: {REPO_ID})")
    except Exception as e:
        errors.append(f"HuggingFace Hub: {e}")

    # 3. GPU
    localDevices = jax.devices()
    if not any(d.platform == "gpu" for d in localDevices):
        errors.append("No GPU detected")
    else:
        print(f"  [OK] GPU detected: {len(localDevices)} x {localDevices[0].device_kind}")

    # 4. Tokenizer (optional — may be on Hub, Kaggle Dataset, or not yet built)
    from tokenizers import Tokenizer
    tokenizerOk = False
    for attempt in ["hub", "kaggleDataset"]:
        try:
            if attempt == "hub" and hfToken:
                path = hf_hub_download(REPO_ID, "tokenizer.json", token=hfToken)
                label = "Hub"
            elif attempt == "kaggleDataset":
                path = "/kaggle/input/zephyros-tokenizer/tokenizer.json"
                if not os.path.exists(path):
                    continue
                label = "Kaggle Dataset"
            else:
                continue
            t = Tokenizer.from_file(path)
            testToken = t.encode("Hello world").ids
            assert len(testToken) > 0
            print(f"  [OK] Tokenizer loaded from {label} (vocab: {t.get_vocab_size()})")
            tokenizerOk = True
            break
        except Exception:
            continue
    if not tokenizerOk:
        warnings.append("Tokenizer not found on Hub or Kaggle Dataset — will retry during data pipeline init")

    # 5. Model forward pass
    from src.model import DecoderOnlyLM
    try:
        probeModel = DecoderOnlyLM(modelConfig, rngs=nnx.Rngs(0))
        probeIds = jnp.zeros((1, 4), dtype=jnp.int32)
        probePos = jnp.arange(4)
        out = probeModel(probeIds, probePos, enableDropout=False)
        assert out.shape == (1, 4, MODEL_CONFIG["vocabSize"])
        nParams = sum(v.size for _, v in nnx.to_flat_state(nnx.state(probeModel, nnx.Param)))
        print(f"  [OK] Model forward pass OK ({nParams:,} params)")
        del probeModel
    except Exception as e:
        errors.append(f"Model forward pass: {e}")

    # 6. Checkpoint I/O
    from src.checkpoint import serializeCheckpoint
    try:
        probeModel2 = DecoderOnlyLM(modelConfig, rngs=nnx.Rngs(1))
        probeOpt = nnx.Optimizer(probeModel2, optax.adamw(3e-4), wrt=nnx.Param)
        _bytes = serializeCheckpoint(probeModel2, probeOpt, 0, 0)
        assert len(_bytes) > 0
        print(f"  [OK] Checkpoint serialization works ({len(_bytes)} bytes)")
        del probeModel2, probeOpt
    except Exception as e:
        errors.append(f"Checkpoint I/O: {e}")

    # 7. Dataset access
    try:
        from datasets import load_dataset
        checkDs = "emozilla/pg19"
        ds = load_dataset(checkDs, split="train", streaming=True)
        sample = next(iter(ds))
        assert "text" in sample
        print(f"  [OK] HuggingFace datasets accessible (sampled {checkDs})")
        del ds
    except Exception as e:
        warnings.append(f"Dataset streaming check failed: {e}")

    # 8. Versions
    import flax
    print(f"  [OK] jax={jax.__version__}, flax={flax.__version__}, optax={optax.__version__}")

    if errors:
        print(f"\n{'='*60}")
        print(f"Preflight FAILED — {len(errors)} fatal issue(s):")
        for i, err in enumerate(errors, 1):
            print(f"  {i}. {err}")
        print(f"{'='*60}")
        sys.exit(1)

    if warnings:
        print(f"\n{'='*60}")
        print(f"Preflight passed with {len(warnings)} warning(s):")
        for i, w in enumerate(warnings, 1):
            print(f"  {i}. {w}")
        print(f"{'='*60}")

    print(f"\n{'='*60}")
    print("All preflight checks passed — starting training")
    print(f"{'='*60}\n")


preflightCheck()

# ── Tokenizer ──────────────────────────────────────────────────────────────────

from tokenizers import Tokenizer

HF_TOKEN = os.environ.get("HF_TOKEN", "")

def loadTokenizer():
    try:
        path = hf_hub_download(REPO_ID, "tokenizer.json", token=HF_TOKEN)
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

if HF_TOKEN:
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
    localDirObj = Path("/kaggle/working/checkpoints")
    localDirObj.mkdir(parents=True, exist_ok=True)
    localPath = localDirObj / f"checkpoint-{globalStep}.msgpack"
    try:
        ckptBytes = serializeCheckpoint(model, optimizer, globalStep, tokensSeen)
        localPath.write_bytes(ckptBytes)
        nBytes = len(ckptBytes)
        del ckptBytes
        print(f"  Local checkpoint saved ({localPath}, {nBytes // 1024**2} MB)")
    except Exception as e:
        print(f"  [WARN] Local save failed: {e}")
        return
    try:
        HF_API.upload_file(
            path_or_fileobj=str(localPath),
            path_in_repo=CKPT_FILE,
            repo_id=REPO_ID,
        )
        print(f"  Uploaded to Hub (step {globalStep})")
    except Exception as e:
        print(f"  [WARN] Hub upload failed: {e}")


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

# ── JIT-Compiled Training Step ───────────────────────────────────────────────

@nnx.jit
def trainStep(model, batch):
    def lossFn(m):
        logits = m(batch["inputIds"], batch["positions"], enableDropout=True)
        return optax.softmax_cross_entropy_with_integer_labels(
            logits, batch["targetIds"]
        ).mean()
    return nnx.value_and_grad(lossFn)(model)

print("Compiling training step (first step will be slow)...")

# ── Data Pipeline ──────────────────────────────────────────────────────────────

from datasets import load_dataset, interleave_datasets

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
    combined = combined.shuffle(buffer_size=1000, seed=sessionSeed)

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

            loss, grads = trainStep(model, batch)

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

    pass
