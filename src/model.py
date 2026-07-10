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

        self.tokenEmbed = TokenEmbedding(
            config.vocabSize, config.dModel,
            rngs=nnx.Rngs(embedKey),
        )

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

        self.finalNorm = RmsNorm(config.dModel, config.rmsNormEps)

    def __call__(self, inputIds: jax.Array, positions: jax.Array, enableDropout: bool = True) -> jax.Array:
        x = self.tokenEmbed(inputIds)
        for block in self.blocks:
            x = block(x, positions, enableDropout=enableDropout)
        x = self.finalNorm(x)
        logits = x @ self.tokenEmbed.weight.T
        return logits
