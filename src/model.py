from flax import nnx
import jax
import jax.numpy as jnp

from src.layers import RmsNorm, TransformerBlock


class TokenEmbedding(nnx.Module):
    def __init__(self, vocabSize: int, dModel: int, rngs: nnx.Rngs):
        self.weight = nnx.Param(
            jax.random.normal(rngs.params(), (vocabSize, dModel)) * 0.02
        )

    def __call__(self, inputIds: jax.Array) -> jax.Array:
        return self.weight[inputIds]


class DecoderOnlyLM(nnx.Module):
    def __init__(self, config, rngs: nnx.Rngs):
        baseKey = rngs()
        embedKey, layersKey = jax.random.split(baseKey)

        self.attnDropout = getattr(config, "attnDropout", 0.0)
        self.tieEmbeddings = getattr(config, "tieEmbeddings", True)

        self.tokenEmbed = TokenEmbedding(
            config.vocabSize, config.dModel,
            rngs=nnx.Rngs(embedKey),
        )

        self.nLayers = config.nLayers
        layerKeys = jax.random.split(layersKey, config.nLayers)

        self.blocks = nnx.List([
            TransformerBlock(
                config.dModel, config.dFF,
                config.nQueryHeads, config.nKVHeads,
                config.headDim, config.maxSeqLen,
                config.ropeTheta, config.rmsNormEps,
                rngs=nnx.Rngs(layerKeys[i]),
                attnDropout=self.attnDropout,
            )
            for i in range(config.nLayers)
        ])

        self._scan_fn = nnx.scan(
            self._block_fn,
            in_axes=(nnx.Carry, None, None),
            out_axes=nnx.Carry,
            length=config.nLayers,
            unroll=2,
        )

        self.finalNorm = RmsNorm(config.dModel, config.rmsNormEps)

        if not self.tieEmbeddings:
            self.outputProj = nnx.Linear(
                config.dModel, config.vocabSize, use_bias=False,
                rngs=nnx.Rngs(rngs()),
            )

    def _block_fn(self, x, positions, enableDropout):
        return self.blocks(x, positions, enableDropout=enableDropout)

    def __call__(self, inputIds: jax.Array, positions: jax.Array, enableDropout: bool = True) -> jax.Array:
        x = self.tokenEmbed(inputIds)
        x = self._scan_fn(x, positions, enableDropout)
        x = self.finalNorm(x)
        if self.tieEmbeddings:
            logits = x @ self.tokenEmbed.weight.T
        else:
            logits = self.outputProj(x)
        return logits
