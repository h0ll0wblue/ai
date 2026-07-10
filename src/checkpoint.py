"""Checkpoint serialization helpers for Flax NNX.

Flax 0.12.x does not expose nnx.to_bytes/from_bytes, so we use
msgpack with manual JAX array conversion.
"""

import msgpack
import numpy as np
import jax.numpy as jnp
from flax import nnx


def _arrays_to_lists(d):
    if isinstance(d, dict):
        return {k: _arrays_to_lists(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [_arrays_to_lists(v) for v in d]
    elif isinstance(d, (jnp.ndarray, np.ndarray)):
        return {
            "__ndarray__": True,
            "shape": list(d.shape),
            "dtype": str(d.dtype),
            "data": msgpack.dumps(np.asarray(d).tobytes()),
        }
    return d


def _lists_to_arrays(d):
    if isinstance(d, dict):
        if "__ndarray__" in d:
            arr = np.frombuffer(
                msgpack.loads(d["data"]), dtype=d["dtype"]
            ).reshape(d["shape"])
            return jnp.array(arr)
        return {k: _lists_to_arrays(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [_lists_to_arrays(v) for v in d]
    return d


def serializeCheckpoint(model, optimizer, step, tokensSeen):
    """Serialize model + optimizer state + metadata to bytes."""
    params = nnx.state(model, nnx.Param)
    optState = nnx.state(optimizer)
    ckpt = {
        "model": nnx.to_pure_dict(params),
        "optimizer": nnx.to_pure_dict(optState),
        "step": step,
        "tokensSeen": tokensSeen,
    }
    return msgpack.dumps(_arrays_to_lists(ckpt))


def deserializeCheckpoint(model, optimizer, data):
    """Restore model + optimizer state from bytes. Returns (step, tokensSeen)."""
    loaded = _lists_to_arrays(msgpack.loads(data, strict_map_key=False))
    modelState = nnx.state(model, nnx.Param)
    nnx.replace_by_pure_dict(modelState, loaded["model"])
    nnx.update(model, modelState)
    if "optimizer" in loaded:
        optState = nnx.state(optimizer)
        nnx.replace_by_pure_dict(optState, loaded["optimizer"])
        nnx.update(optimizer, optState)
    return loaded.get("step", 0), loaded.get("tokensSeen", 0)
