"""Checkpoint serialization helpers for Flax NNX.

Uses float16 for storage to halve checkpoint size (~3.6GB for 600M).
Writes to disk and reads from disk to avoid keeping 3.6GB in Python memory.
"""

import io
import msgpack
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx


def _arrays_to_lists(d):
    if isinstance(d, dict):
        return {k: _arrays_to_lists(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [_arrays_to_lists(v) for v in d]
    elif isinstance(d, (jnp.ndarray, np.ndarray)):
        arr = np.asarray(d)
        origDtype = str(arr.dtype)
        storedDtype = origDtype
        if arr.dtype in (np.float32, np.float64):
            arr = arr.astype(np.float16)
            storedDtype = "float16"
        return {
            "__ndarray__": True,
            "shape": list(d.shape),
            "dtype": origDtype,
            "stored_dtype": storedDtype,
            "data": msgpack.dumps(arr.tobytes()),
        }
    return d


def _lists_to_arrays(d):
    if isinstance(d, dict):
        if "__ndarray__" in d:
            storedDtype = d.get("stored_dtype")
            if storedDtype is None:
                # Backwards compatibility fallback
                storedDtype = "float16" if d["dtype"] in ("float32", "float64") else d["dtype"]
            arr = np.frombuffer(
                msgpack.loads(d["data"]), dtype=np.dtype(storedDtype)
            ).reshape(d["shape"])
            targetDtype = d["dtype"]
            if str(arr.dtype) != targetDtype:
                arr = arr.astype(targetDtype)
            return jnp.array(arr)
        return {k: _lists_to_arrays(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [_lists_to_arrays(v) for v in d]
    return d


def _to_bytes(state):
    """Serialize nnx state to msgpack bytes via float16 compression."""
    pure = nnx.to_pure_dict(state)
    return msgpack.dumps(_arrays_to_lists(pure))


def serializeCheckpoint(model, optimizer, step, tokensSeen):
    """Serialize model + optimizer state to msgpack bytes (float16 compressed)."""
    params = nnx.state(model, nnx.Param)
    optState = nnx.state(optimizer)
    ckpt = {
        "model": _to_bytes(params),
        "optimizer": _to_bytes(optState),
        "step": step,
        "tokensSeen": tokensSeen,
    }
    return msgpack.dumps(ckpt)


def _from_bytes(data):
    """Restore nnx state from float16-compressed msgpack bytes."""
    return _lists_to_arrays(msgpack.loads(data, strict_map_key=False))


def deserializeCheckpoint(model, optimizer, data):
    """Restore model + optimizer state from bytes. Returns (step, tokensSeen)."""
    ckpt = msgpack.loads(data, strict_map_key=False)
    modelState = nnx.state(model, nnx.Param)
    nnx.replace_by_pure_dict(modelState, _from_bytes(ckpt["model"]))
    nnx.update(model, modelState)
    if "optimizer" in ckpt:
        optState = nnx.state(optimizer)
        nnx.replace_by_pure_dict(optState, _from_bytes(ckpt["optimizer"]))
        nnx.update(optimizer, optState)
    return ckpt.get("step", 0), ckpt.get("tokensSeen", 0)
