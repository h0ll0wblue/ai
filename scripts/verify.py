import yaml
import jax
import jax.numpy as jnp
from flax import nnx
import optax

from src.model import DecoderOnlyLM


class Config:
    vocabSize: int
    dModel: int
    dFF: int
    nLayers: int
    nQueryHeads: int
    nKVHeads: int
    headDim: int
    maxSeqLen: int
    ropeTheta: float
    rmsNormEps: float
    tieEmbeddings: bool

    def __init__(self, path: str) -> None:
        with open(path) as f:
            data = yaml.safe_load(f)
        self.vocabSize = data["vocabSize"]
        self.dModel = data["dModel"]
        self.dFF = data["dFF"]
        self.nLayers = data["nLayers"]
        self.nQueryHeads = data["nQueryHeads"]
        self.nKVHeads = data["nKVHeads"]
        self.headDim = data["headDim"]
        self.maxSeqLen = data["maxSeqLen"]
        self.ropeTheta = data["ropeTheta"]
        self.rmsNormEps = data["rmsNormEps"]
        self.tieEmbeddings = data["tieEmbeddings"]


def paramCount(model: DecoderOnlyLM) -> int:
    total = 0
    for _, var in nnx.to_flat_state(nnx.state(model, nnx.Param)):
        total += var.size
    return total


def verify():
    config = Config("configs/model.yaml")
    rngs = nnx.Rngs(0)

    B, S = 2, 64
    model = DecoderOnlyLM(config, rngs=rngs)

    nParams = paramCount(model)
    print(f"Parameters: {nParams:,}")
    assert abs(nParams - 600_000_000) < 10_000_000
    print("  [OK] Within 590M-610M target")

    inputIds = jnp.zeros((B, S), dtype=jnp.int32)
    positions = jnp.arange(S)

    logits = model(inputIds, positions, enableDropout=False)
    print(f"Forward output shape: {logits.shape}")
    assert logits.shape == (B, S, config.vocabSize)
    print("  [OK] Correct output shape")

    randomIds = jax.random.randint(jax.random.PRNGKey(7), (B, S), 0, config.vocabSize)
    randomLogits = model(randomIds, positions, enableDropout=False)
    assert randomLogits.shape == (B, S, config.vocabSize)
    row0 = randomLogits[0, 0, :10]
    row1 = randomLogits[0, 1, :10]
    assert not jnp.allclose(row0, row1), "Adjacent positions produce identical logits"
    print("  [OK] Non-uniform outputs from non-zero inputs")

    def lossFn(model):
        logits = model(randomIds, positions, enableDropout=False)
        targets = jax.random.randint(jax.random.PRNGKey(99), (B, S), 0, config.vocabSize)
        return optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()

    loss, grads = nnx.value_and_grad(lossFn)(model)
    lossBefore = float(loss)
    print(f"Loss (untrained): {lossBefore:.4f}")
    assert lossBefore > 0
    print("  [OK] Backward pass computed successfully")

    gNorm = sum(jnp.sum(g ** 2) for _, g in nnx.to_flat_state(nnx.state(grads))) ** 0.5
    assert float(gNorm) > 0, "Gradients are all zero"
    print(f"  [OK] Gradient norm: {float(gNorm):.4f} (non-zero)")

    optimizer = nnx.Optimizer(model, optax.adamw(3e-4, weight_decay=0.01), wrt=nnx.Param)
    optimizer.update(model, grads)
    print("  [OK] Optimizer step applied successfully")

    lossAfter = float(lossFn(model))
    assert lossAfter < lossBefore, f"Loss did not decrease: {lossBefore:.4f} -> {lossAfter:.4f}"
    print(f"  [OK] Loss decreased: {lossBefore:.4f} -> {lossAfter:.4f}")

    print("  [OK] All checks passed")


if __name__ == "__main__":
    verify()
