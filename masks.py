import jax
import jax.numpy as jnp


def PaddingMask(seq: jax.Array, padding_token: int):
    return ~(seq == padding_token)


def CausalMask(seq: jax.Array):
    batch_size, seq_len = seq.shape
    mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool))[None, ...]
    return mask


def PredMask(seq: jax.Array, end_token: int):
    return jnp.cumsum(seq == end_token, axis=-1) == 1


def PredAnswerMask(seq: jax.Array, pred_mask: jax.Array, think_end_token: int):
    return pred_mask & (jnp.cumsum(seq == think_end_token, axis=-1) == 1)


def ResponseMask(in_seq: jax.Array, out_seq: jax.Array, padding_token: int):
    return (in_seq == padding_token) & (out_seq != padding_token)


def AnswerMask(out_seq: jax.Array, response_mask: jax.Array, start_token: int):
    return response_mask & (jnp.cumsum(out_seq == start_token, axis=-1) == 2)
