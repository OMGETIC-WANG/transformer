from rope import RoPE
import jax
import jax.numpy as jnp
from flax import nnx
import typing as tp
from flax.typing import Dtype


class RoPEAttention(nnx.Module):
    def __init__(
        self,
        num_heads: int,
        in_features: int,
        rope: RoPE,
        *,
        rngs: nnx.Rngs,
        kernel_init=nnx.initializers.lecun_normal(),
        param_dtype: Dtype = jnp.float32,
        dtype: tp.Optional[Dtype] = None,
    ):
        self.num_heads = num_heads
        self.in_features = in_features
        self.rope = rope
        self.head_dim = in_features // num_heads
        self.param_dtype = param_dtype
        self.dtype = dtype

        init_qkv = lambda: nnx.LinearGeneral(
            in_features=in_features,
            out_features=(num_heads, self.head_dim),
            kernel_init=kernel_init,
            rngs=rngs,
            param_dtype=param_dtype,
            dtype=dtype,
        )
        self.query = init_qkv()
        self.key = init_qkv()
        self.value = init_qkv()

        self.key_cache: tp.Optional[nnx.Cache[jax.Array]] = nnx.data(None)
        self.value_cache: tp.Optional[nnx.Cache[jax.Array]] = nnx.data(None)
        self.cache_index: nnx.Cache[int] = nnx.Cache(0)

    @staticmethod
    def _MergeHeads(x: jax.Array):
        batch_size, seq_len, num_heads, head_dim = x.shape
        return x.reshape(batch_size, seq_len, num_heads * head_dim)

    def _SplitHeads(self, x: jax.Array):
        batch_size, seq_len, features = x.shape
        return x.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

    def _Q(self, in_q: jax.Array, offset: int = 0):
        return self._SplitHeads(self.rope(self._MergeHeads(self.query(in_q)), offset=offset))

    def _K(self, in_k: jax.Array, offset: int = 0):
        return self._SplitHeads(self.rope(self._MergeHeads(self.key(in_k)), offset=offset))

    def _V(self, in_v: jax.Array):
        return self.value(in_v)

    def __call__(
        self,
        in_q: jax.Array,
        in_k: tp.Optional[jax.Array] = None,
        in_v: tp.Optional[jax.Array] = None,
        mask: tp.Optional[jax.Array] = None,
        offset: int = 0,
    ):
        if in_k is None:
            in_k = in_q
        if in_v is None:
            in_v = in_k

        q = self._Q(in_q, offset=offset)
        k = self._K(in_k, offset=offset)
        v = self._V(in_v)

        return self._MergeHeads(nnx.dot_product_attention(q, k, v, mask=mask, dtype=self.dtype))

    def InitKVCache(self, batch_size: int, max_seq_len: int):
        cache_shape = (batch_size, max_seq_len, self.num_heads, self.head_dim)
        self.key_cache = nnx.Cache(jnp.zeros(cache_shape, dtype=self.param_dtype))
        self.value_cache = nnx.Cache(jnp.zeros(cache_shape, dtype=self.param_dtype))
        self.cache_index = nnx.Cache(0)
        self.max_seq_len = max_seq_len

    def ResetKVCache(self):
        if self.key_cache is not None and self.value_cache is not None:
            self.key_cache[...] = jnp.zeros_like(self.key_cache[...])
            self.value_cache[...] = jnp.zeros_like(self.value_cache[...])
        self.cache_index[...] = 0

    def ReleaseKVCache(self):
        self.key_cache = None
        self.value_cache = None
        self.cache_index = nnx.Cache(0)

    def PrefillKVCache(self, in_q: jax.Array, fill_count: int):
        assert self.key_cache is not None and self.value_cache is not None
        q = self._Q(in_q)
        k = self._K(in_q)
        v = self._V(in_q)

        self.key_cache[...] = k
        self.value_cache[...] = v
        self.cache_index[...] = fill_count

        mask = jnp.tril(
            jnp.ones((self.key_cache[...].shape[1], self.key_cache[...].shape[1]), dtype=jnp.bool)
        )[None, ...] & (
            jnp.arange(0, self.key_cache[...].shape[1])[None, None, None, :] < fill_count
        )

        return self._MergeHeads(
            nnx.dot_product_attention(
                q, self.key_cache[...], self.value_cache[...], mask=mask, dtype=self.dtype
            )
        )

    def CachedDecode(self, in_q: jax.Array, mask: tp.Optional[jax.Array] = None):
        assert self.key_cache is not None and self.value_cache is not None

        q = self._Q(in_q, offset=self.cache_index.get_value())
        k = self._K(in_q, offset=self.cache_index.get_value())
        v = self._V(in_q)

        self.key_cache[...] = (
            self.key_cache[...].at[:, self.cache_index.get_value(), :].set(k[:, 0, :, :])
        )
        self.value_cache[...] = (
            self.value_cache[...].at[:, self.cache_index.get_value(), :].set(v[:, 0, :, :])
        )
        self.cache_index[...] += 1

        index_mask = (
            jnp.arange(0, self.key_cache[...].shape[1])[None, None, None, :]
            < self.cache_index.get_value()
        )
        if mask is not None:
            combined_mask = mask & index_mask
        else:
            combined_mask = index_mask

        return self._MergeHeads(
            nnx.dot_product_attention(
                q, self.key_cache[...], self.value_cache[...], mask=combined_mask, dtype=self.dtype
            )
        )
