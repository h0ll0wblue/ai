"""Checkpoint serialization helpers for Flax NNX.

Uses float16 for storage of model weights (halves checkpoint size for 600M).
Optimizer state (Adam m, v) is kept in fp32 — float16 corrupts v (underflows to
zero) causing NaN on resume.
Supports two backends:
  - Orbax (v0.5+): memory-efficient directory-based checkpointing (preferred).
  - Msgpack: single-file serialization (fallback, also used for Hub upload).
"""

import io
import msgpack
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx


try:
    import orbax.checkpoint as ocp
    _HAS_ORBAX = True
except ImportError:
    _HAS_ORBAX = False


def _arrays_to_lists(d, allow_f16_compress=False):
    if isinstance(d, dict):
        return {k: _arrays_to_lists(v, allow_f16_compress) for k, v in d.items()}
    elif isinstance(d, list):
        return [_arrays_to_lists(v, allow_f16_compress) for v in d]
    elif isinstance(d, (jnp.ndarray, np.ndarray)):
        arr = np.asarray(d)
        origDtype = str(arr.dtype)
        storedDtype = origDtype
        # Only compress MODEL WEIGHTS (large, tolerant of f16).
        # NEVER compress optimizer state — Adam v underflows and causes NaN.
        if allow_f16_compress and arr.dtype in (np.float32, np.float64):
            arr = arr.astype(np.float16)
            storedDtype = "float16"
        return {
            "__ndarray__": True,
            "shape": list(arr.shape),
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


def _to_bytes(state, allow_f16_compress=False):
    """Serialize nnx state to msgpack bytes, optionally compressing f32 to f16."""
    pure = nnx.to_pure_dict(state)
    return msgpack.dumps(_arrays_to_lists(pure, allow_f16_compress))


def serializeCheckpoint(model, optimizer, step, tokensSeen):
    """Serialize model + optimizer state to msgpack bytes."""
    params = nnx.state(model, nnx.Param)
    optState = nnx.state(optimizer)
    ckpt = {
        "model":     _to_bytes(params,  allow_f16_compress=True),   # weights: f16 OK
        "optimizer": _to_bytes(optState, allow_f16_compress=False), # f32 preserved
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
    nnx.update(model, _from_bytes(ckpt["model"]))
    if "optimizer" in ckpt:
        nnx.update(optimizer, _from_bytes(ckpt["optimizer"]))
    return ckpt.get("step", 0), ckpt.get("tokensSeen", 0)


# ── Orbax backend (optional, v0.5+) ───────────────────────────────────────────

if _HAS_ORBAX:

    def save_checkpoint_orbax(model, optimizer, step, tokensSeen, path):
        """Save checkpoint using Orbax (async, memory-efficient, directory-based)."""
        checkpointer = ocp.PyTreeCheckpointer()
        checkpointer.save(
            path,
            {
                "model": nnx.state(model, nnx.Param),
                "optimizer": nnx.state(optimizer),
                "step": step,
                "tokensSeen": tokensSeen,
            },
            force_ckpt=True,
        )

    def load_checkpoint_orbax(model, optimizer, path):
        """Restore checkpoint from Orbax directory. Returns (step, tokensSeen)."""
        checkpointer = ocp.PyTreeCheckpointer()
        restored = checkpointer.restore(path)
        modelState = nnx.state(model, nnx.Param)
        nnx.replace_by_pure_dict(modelState, restored["model"])
        nnx.update(model, modelState)
        optState = nnx.state(optimizer)
        nnx.replace_by_pure_dict(optState, restored["optimizer"])
        nnx.update(optimizer, optState)
        return restored.get("step", 0), restored.get("tokensSeen", 0)

else:
    # Stub: fall back silently when orbax is not installed.
    def save_checkpoint_orbax(model, optimizer, step, tokensSeen, path):
        raise RuntimeError("orbax-checkpoint not installed; use msgpack backend")

    def load_checkpoint_orbax(model, optimizer, path):
        raise RuntimeError("orbax-checkpoint not installed; use msgpack backend")


def has_orbax() -> bool:
    """Whether Orbax is available for checkpointing."""
    return _HAS_ORBAX
