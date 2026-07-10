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
        self.remat = getattr(config, "remat", False)

        self.tokenEmbed = TokenEmbedding(
            config.vocabSize, config.dModel,
            rngs=nnx.Rngs(embedKey),
        )

        self.nLayers = config.nLayers
        layerKeys = jax.random.split(layersKey, config.nLayers)
        
        for i in range(config.nLayers):
            setattr(self, f"block_{i}",
                TransformerBlock(
                    config.dModel, config.dFF,
                    config.nQueryHeads, config.nKVHeads,
                    config.headDim, config.maxSeqLen,
                    config.ropeTheta, config.rmsNormEps,
                    rngs=nnx.Rngs(layerKeys[i]),
                    attnDropout=self.attnDropout,
                )
            )

        self.finalNorm = RmsNorm(config.dModel, config.rmsNormEps)

    def __call__(self, inputIds: jax.Array, positions: jax.Array, enableDropout: bool = True) -> jax.Array:
        x = self.tokenEmbed(inputIds)
        for i in range(self.nLayers):
            block = getattr(self, f"block_{i}")
            if self.remat:
                x = nnx.remat(TransformerBlock.__call__, static_argnums=3)(block, x, positions, enableDropout)
            else:
                x = block(x, positions, enableDropout=enableDropout)
        x = self.finalNorm(x)
        logits = x @ self.tokenEmbed.weight.T
        return logits
