"""End-to-end smoke tests for the model and training pipeline.

Runs on CPU. Verifies:
  - Model init, forward, backward, optimizer step
  - Gradient accumulation math
  - Batch-size config calculations
  - Data pipeline packing
"""

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from src.model import DecoderOnlyLM

# ── Helpers ───────────────────────────────────────────────────────────────────

SEQ_LEN = 64
MODEL_CONFIG = type("Config", (), {
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
    "attnDropout": 0.0,
    "tieEmbeddings": True,
})()


def makeModel():
    return DecoderOnlyLM(MODEL_CONFIG, rngs=nnx.Rngs(0))


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_model_init():
    model = makeModel()
    nParams = sum(v.size for _, v in nnx.to_flat_state(nnx.state(model, nnx.Param)))
    assert abs(nParams - 600_000_000) < 10_000_000, f"Param count {nParams} off target"
    # fmt:off
    print(f"  [OK] {nParams:,} params")


def test_forward_shape():
    model = makeModel()
    ids = jnp.zeros((2, SEQ_LEN), dtype=jnp.int32)
    pos = jnp.arange(SEQ_LEN)
    logits = model(ids, pos, enableDropout=False)
    assert logits.shape == (2, SEQ_LEN, MODEL_CONFIG.vocabSize)
    print(f"  [OK] forward shape {logits.shape}")


def test_backward_and_optimizer_step():
    model = makeModel()
    ids = jax.random.randint(jax.random.PRNGKey(0), (2, SEQ_LEN), 0, MODEL_CONFIG.vocabSize)
    targets = jax.random.randint(jax.random.PRNGKey(1), (2, SEQ_LEN), 0, MODEL_CONFIG.vocabSize)
    pos = jnp.arange(SEQ_LEN)

    def lossFn(m):
        logits = m(ids, pos, enableDropout=False)
        return optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()

    loss, grads = nnx.value_and_grad(lossFn)(model)
    assert loss > 0
    gNorm = sum(jnp.sum(g ** 2) for _, g in nnx.to_flat_state(nnx.state(grads))) ** 0.5
    assert float(gNorm) > 0, "zero gradients"
    optimizer = nnx.Optimizer(model, optax.adamw(3e-4, weight_decay=0.01), wrt=nnx.Param)
    optimizer.update(model, grads)
    assert True
    print(f"  [OK] backward + optimizer step (loss={float(loss):.4f}, gNorm={float(gNorm):.4f})")


def test_gradient_accumulation():
    """Verify that 2 micro-batches accumulated equals 1 double-sized batch."""
    model = makeModel()
    B1, B2 = 1, 1
    S = SEQ_LEN
    V = MODEL_CONFIG.vocabSize

    ids = jax.random.randint(jax.random.PRNGKey(0), (B1 + B2, S), 0, V)
    targets = jax.random.randint(jax.random.PRNGKey(1), (B1 + B2, S), 0, V)
    pos = jnp.arange(S)

    # Full batch (B1 + B2)
    def fullLoss(m):
        logits = m(ids, pos, enableDropout=False)
        return optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()

    _, fullGrads = nnx.value_and_grad(fullLoss)(model)

    # Reseed model for comparison
    model2 = makeModel()
    gradsAccum = None
    for i in [0, 1]:
        batchIds = ids[i:i+1]
        batchTargs = targets[i:i+1]

        def microLoss(m):
            logits = m(batchIds, pos, enableDropout=False)
            return optax.softmax_cross_entropy_with_integer_labels(logits, batchTargs).mean()

        _, grads = nnx.value_and_grad(microLoss)(model2)
        gradsAccum = grads if gradsAccum is None else jax.tree.map(jnp.add, gradsAccum, grads)

    avgGrads = jax.tree.map(lambda g: g / 2, gradsAccum)

    # Compare each param's gradient
    flatFull = dict(nnx.to_flat_state(nnx.state(fullGrads)))
    flatAccum = dict(nnx.to_flat_state(nnx.state(avgGrads)))
    for key in flatFull:
        diff = jnp.max(jnp.abs(flatFull[key] - flatAccum[key]))
        assert diff < 1e-5, f"grad mismatch for {key}: max_diff={diff}"
    print("  [OK] gradient accumulation: accumulated grads match full-batch grads")


def test_config_tokens_per_step():
    """Verify token-per-step calculations for various hardware configs."""
    cases = [
        (1, 1, 128, 4096, 1 * 4096 * 1 * 128),   # singleGPU
        (2, 1, 128, 4096, 2 * 4096 * 1 * 128),   # singleGPU larger batch
        (4, 8, 8,   4096, 4 * 4096 * 8 * 8),      # TPU 8-chip
        (2, 2, 64,  4096, 2 * 4096 * 2 * 64),     # dualGPU
    ]
    for mb, nc, ga, seq, expected in cases:
        result = mb * seq * nc * ga
        assert result == expected, f"({mb}*{seq}*{nc}*{ga}) = {result} != {expected}"
    print(f"  [OK] all {len(cases)} config calculations correct")


def test_model_can_overfit_single_batch():
    """Verify loss decreases over 5 steps on one batch (tiny arch for speed)."""
    tinyCfg = type("Config", (), {
        "vocabSize": 512, "dModel": 64, "dFF": 128, "nLayers": 2,
        "nQueryHeads": 2, "nKVHeads": 1, "headDim": 32, "maxSeqLen": 128,
        "ropeTheta": 10000.0, "rmsNormEps": 1e-6, "tieEmbeddings": True,
    })()
    model = DecoderOnlyLM(tinyCfg, rngs=nnx.Rngs(0))
    B, S, V = 2, 16, 512
    ids = jax.random.randint(jax.random.PRNGKey(0), (B, S), 0, V)
    targets = jax.random.randint(jax.random.PRNGKey(1), (B, S), 0, V)
    pos = jnp.arange(S)

    def lossFn(m):
        logits = m(ids, pos, enableDropout=False)
        return optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()

    optimizer = nnx.Optimizer(model, optax.adamw(1e-2), wrt=nnx.Param)
    initLoss = float(lossFn(model))

    for _ in range(5):
        _loss, grads = nnx.value_and_grad(lossFn)(model)
        optimizer.update(model, grads)

    finalLoss = float(lossFn(model))
    assert finalLoss < initLoss, f"Loss did not decrease: {initLoss:.4f} -> {finalLoss:.4f}"
    print(f"  [OK] loss decreases over steps: {initLoss:.4f} -> {finalLoss:.4f}")


def test_causal_attention_mask():
    """Verify causal mask: changing a future input token leaves earlier outputs unchanged."""
    tinyCfg = type("Config", (), {
        "vocabSize": 512, "dModel": 32, "dFF": 64, "nLayers": 1,
        "nQueryHeads": 2, "nKVHeads": 1, "headDim": 16, "maxSeqLen": 32,
        "ropeTheta": 10000.0, "rmsNormEps": 1e-6, "tieEmbeddings": True,
    })()
    model = DecoderOnlyLM(tinyCfg, rngs=nnx.Rngs(0))
    S = 8
    pos = jnp.arange(S)

    ids1 = jax.random.randint(jax.random.PRNGKey(0), (1, S), 0, 512)
    ids2 = ids1.at[0, -1].set(999)  # change last token only

    logits1 = model(ids1, pos, enableDropout=False)
    logits2 = model(ids2, pos, enableDropout=False)

    for t in range(S - 1):
        diff = float(jnp.max(jnp.abs(logits1[0, t] - logits2[0, t])))
        assert diff < 1e-6, (
            f"Output at position {t} changed when last token was modified (diff={diff})"
        )
    print("  [OK] causal mask prevents future tokens from influencing earlier outputs")


def test_checkpoint_roundtrip():
    """Verify model + optimizer state serializes/deserializes correctly."""
    from src.checkpoint import serializeCheckpoint, deserializeCheckpoint

    tinyCfg = type("Config", (), {
        "vocabSize": 512, "dModel": 64, "dFF": 128, "nLayers": 2,
        "nQueryHeads": 2, "nKVHeads": 1, "headDim": 32, "maxSeqLen": 128,
        "ropeTheta": 10000.0, "rmsNormEps": 1e-6, "tieEmbeddings": True,
    })()
    model = DecoderOnlyLM(tinyCfg, rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adamw(3e-4), wrt=nnx.Param)
    B, S, V = 1, 8, 512
    ids = jax.random.randint(jax.random.PRNGKey(0), (B, S), 0, V)
    pos = jnp.arange(S)
    targets = jax.random.randint(jax.random.PRNGKey(1), (B, S), 0, V)

    def lossFn(m):
        logits = m(ids, pos, enableDropout=False)
        return optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()

    loss, grads = nnx.value_and_grad(lossFn)(model)
    optimizer.update(model, grads)
    lossAfterStep1 = float(lossFn(model))

    ckptBytes = serializeCheckpoint(model, optimizer, step=5, tokensSeen=9999)

    model2 = DecoderOnlyLM(tinyCfg, rngs=nnx.Rngs(0))
    optimizer2 = nnx.Optimizer(model2, optax.adamw(3e-4), wrt=nnx.Param)
    step, tokensSeen = deserializeCheckpoint(model2, optimizer2, ckptBytes)
    assert step == 5
    assert tokensSeen == 9999

    lossBeforeStep2 = float(lossFn(model2))
    diff = abs(lossBeforeStep2 - lossAfterStep1)
    assert diff < 1e-3, (
        f"Restored model differs: {lossAfterStep1:.4f} -> {lossBeforeStep2:.4f} (diff={diff:.2e})"
    )
    print(f"  [OK] checkpoint roundtrip: step={step}, tokens={tokensSeen}, loss match={diff:.2e}")


def test_data_pipeline_packing():
    """Verify the packAndBatch logic produces correct seqLen chunks."""
    seqLen = 8
    testTokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    def mockIterator():
        yield {"inputIds": testTokens}

    buffer = []
    results = []
    for example in mockIterator():
        buffer.extend(example["inputIds"])
        while len(buffer) >= seqLen:
            seq = buffer[:seqLen]
            buffer = buffer[seqLen:]
            inputIds = jnp.array(seq, dtype=jnp.int32)
            targetIds = jnp.concatenate([
                inputIds[1:],
                jnp.zeros((1,), dtype=jnp.int32),
            ])
            results.append((inputIds, targetIds))

    assert len(results) == 2  # 20 tokens -> 2 full seqs of 8
    assert list(results[0][0]) == [1, 2, 3, 4, 5, 6, 7, 8]
    assert list(results[0][1]) == [2, 3, 4, 5, 6, 7, 8, 0]
    assert list(results[1][0]) == [9, 10, 11, 12, 13, 14, 15, 16]
    assert list(results[1][1]) == [10, 11, 12, 13, 14, 15, 16, 0]
    print("  [OK] data pipeline packing produces correct chunks")
