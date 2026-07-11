from flax import nnx
import jax
import jax.numpy as jnp


class RmsNorm(nnx.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        self.weight = nnx.Param(jnp.ones((dim,)))
        self.eps = eps

    def __call__(self, x: jax.Array) -> jax.Array:
        dtype = x.dtype
        x = x.astype(jnp.float32)
        variance = jnp.mean(x * x, axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(variance + self.eps)
        return (x * self.weight).astype(dtype)


class RotaryPositionEncoding(nnx.Module):
    def __init__(self, dim: int, maxSeqLen: int, theta: float = 500000.0):
        freqs = theta ** (-jnp.arange(0, dim, 2, dtype=jnp.float32) / dim)
        positions = jnp.arange(maxSeqLen, dtype=jnp.float32)
        angles = jnp.outer(positions, freqs)
        self.cos = nnx.Variable(jnp.cos(angles))
        self.sin = nnx.Variable(jnp.sin(angles))

    def __call__(
        self, q: jax.Array, k: jax.Array, positions: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        if positions.ndim == 1:
            cos = self.cos[positions][None, None, :, :]
            sin = self.sin[positions][None, None, :, :]
        else:
            cos = self.cos[positions][:, None, :, :]
            sin = self.sin[positions][:, None, :, :]

        q = q.reshape(*q.shape[:-1], -1, 2)
        k = k.reshape(*k.shape[:-1], -1, 2)

        q = jnp.stack([
            q[..., 0] * cos - q[..., 1] * sin,
            q[..., 1] * cos + q[..., 0] * sin,
        ], axis=-1)

        k = jnp.stack([
            k[..., 0] * cos - k[..., 1] * sin,
            k[..., 1] * cos + k[..., 0] * sin,
        ], axis=-1)

        return q.reshape(*q.shape[:-2], -1), k.reshape(*k.shape[:-2], -1)


class GroupedQueryAttention(nnx.Module):
    def __init__(
        self,
        dModel: int,
        nHeads: int,
        nKVHeads: int,
        headDim: int,
        maxSeqLen: int,
        ropeTheta: float,
        rngs: nnx.Rngs,
        attnDropout: float = 0.0,
    ):
        self.nHeads = nHeads
        self.nKVHeads = nKVHeads
        self.headDim = headDim
        self.attnDropout = attnDropout

        _, qKey, oKey = jax.random.split(rngs(), 3)

        qkvDim = nHeads * headDim + 2 * nKVHeads * headDim
        self.qkvProj = nnx.Linear(dModel, qkvDim, use_bias=False, rngs=nnx.Rngs(qKey))
        self.outProj = nnx.Linear(nHeads * headDim, dModel, use_bias=False, rngs=nnx.Rngs(oKey))

        self.rope = RotaryPositionEncoding(headDim, maxSeqLen, ropeTheta)
        self.attnDropoutLayer = nnx.Dropout(rate=attnDropout, rngs=rngs)

    def __call__(self, x: jax.Array, positions: jax.Array, enableDropout: bool = True) -> jax.Array:
        B, S, _ = x.shape

        qkv = self.qkvProj(x)
        q, k, v = jnp.split(
            qkv,
            [self.nHeads * self.headDim, (self.nHeads + self.nKVHeads) * self.headDim],
            axis=-1,
        )

        q = q.reshape(B, S, self.nHeads, self.headDim).transpose(0, 2, 1, 3)
        k = k.reshape(B, S, self.nKVHeads, self.headDim).transpose(0, 2, 1, 3)
        v = v.reshape(B, S, self.nKVHeads, self.headDim).transpose(0, 2, 1, 3)

        q, k = self.rope(q, k, positions)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        output = jax.nn.dot_product_attention(
            q, k, v, is_causal=True, implementation="xla"
        )

        if enableDropout and self.attnDropout > 0.0:
            output = self.attnDropoutLayer(output, deterministic=False)

        output = output.reshape(B, S, self.nHeads * self.headDim)
        return self.outProj(output)


class SwiGLU(nnx.Module):
    def __init__(self, dModel: int, dFF: int, rngs: nnx.Rngs):
        gateKey, upKey, downKey = jax.random.split(rngs(), 3)

        self.gate = nnx.Linear(dModel, dFF, use_bias=False, rngs=nnx.Rngs(gateKey))
        self.up = nnx.Linear(dModel, dFF, use_bias=False, rngs=nnx.Rngs(upKey))
        self.down = nnx.Linear(dFF, dModel, use_bias=False, rngs=nnx.Rngs(downKey))

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.down(jax.nn.silu(self.gate(x)) * self.up(x))


class TransformerBlock(nnx.Module):
    def __init__(
        self,
        dModel: int,
        dFF: int,
        nHeads: int,
        nKVHeads: int,
        headDim: int,
        maxSeqLen: int,
        ropeTheta: float,
        rmsNormEps: float,
        rngs: nnx.Rngs,
        attnDropout: float = 0.0,
    ):
        attnKey, ffnKey = jax.random.split(rngs())

        self.norm1 = RmsNorm(dModel, rmsNormEps)
        self.attn = GroupedQueryAttention(
            dModel, nHeads, nKVHeads, headDim, maxSeqLen, ropeTheta,
            rngs=nnx.Rngs(attnKey),
            attnDropout=attnDropout,
        )
        self.norm2 = RmsNorm(dModel, rmsNormEps)
        self.ffn = SwiGLU(dModel, dFF, rngs=nnx.Rngs(ffnKey))

    def __call__(self, x: jax.Array, positions: jax.Array, enableDropout: bool = True) -> jax.Array:
        x = x + self.attn(self.norm1(x), positions, enableDropout=enableDropout)
        x = x + self.ffn(self.norm2(x))
        return x
