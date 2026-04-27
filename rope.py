import jax
import jax.numpy as jnp
from flax.typing import Dtype


class RoPE:
    def __init__(self, features: int, max_len: int, *, dtype: Dtype = jnp.float32):
        self.features = features
        self.max_len = max_len
        self.dtype = dtype

        inv_freq = (1.0 / (10000 ** (jnp.arange(0, features, 2) / features)))[None, ...]
        t = jnp.arange(max_len)[..., None]

        freqs = inv_freq * t  # (max_len, features // 2)

        emb = jnp.concatenate((freqs, freqs), axis=-1, dtype=dtype)  # (max_len, features)
        self.cos_cache = jnp.cos(emb)
        self.sin_cache = jnp.sin(emb)

    @staticmethod
    def _rotate_half(x: jax.Array):
        x1, x2 = jnp.split(x, 2, axis=-1)
        return jnp.concatenate((-x2, x1), axis=-1)

    def __call__(self, x: jax.Array, offset: int = 0):
        seq_len = x.shape[1]
        cos = jax.lax.dynamic_slice(self.cos_cache, (offset, 0), (seq_len, self.features))
        sin = jax.lax.dynamic_slice(self.sin_cache, (offset, 0), (seq_len, self.features))

        x1 = x
        x2 = self._rotate_half(x)
        return x1 * cos + x2 * sin
