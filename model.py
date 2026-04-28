from flax import nnx
import jax.numpy as jnp
import jax
import typing as tp
from flax.typing import Dtype
from rope import RoPE
from rope_attention import RoPEAttention


class TokenConfig(tp.NamedTuple):
    num_embeddings: int
    start_token: int
    end_token: int
    padding_token: int
    think_start_token: int
    think_end_token: int
    seperate_token: int
    mul_token: int
    equal_token: int
    zero_token: int


class MLP(nnx.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: tp.Sequence[int],
        activation: tp.Callable[[jax.Array], jax.Array],
        *,
        rngs: nnx.Rngs,
        param_dtype: Dtype = jnp.float32,
        dtype: tp.Optional[Dtype] = None,
    ):
        self.layers = nnx.Sequential()
        prev_features = [in_features, *hidden_features]
        post_features = [*hidden_features, out_features]
        for i in range(len(prev_features)):
            self.layers.layers.append(
                nnx.Linear(
                    prev_features[i],
                    post_features[i],
                    rngs=rngs,
                    param_dtype=param_dtype,
                    dtype=dtype,
                )
            )
            if i < len(prev_features) - 1:
                self.layers.layers.append(activation)

    def __call__(self, x: jax.Array):
        return self.layers(x)


class RoPETransformerBlock(nnx.Module):
    def __init__(
        self,
        in_features: int,
        num_heads: int,
        rope_cache: RoPE,
        *,
        rngs: nnx.Rngs,
        fnn_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0,
        param_dtype: Dtype = jnp.float32,
        dtype: tp.Optional[Dtype] = None,
    ):
        self.attention = RoPEAttention(
            num_heads,
            in_features,
            rope_cache,
            rngs=rngs,
            kernel_init=nnx.initializers.xavier_normal(),
        )
        self.fnn = MLP(
            in_features,
            in_features,
            [in_features * 4],
            nnx.gelu,
            rngs=rngs,
            param_dtype=param_dtype,
            dtype=dtype,
        )

        self.pre_attention_norm = nnx.LayerNorm(in_features, rngs=rngs)
        self.pre_fnn_norm = nnx.LayerNorm(in_features, rngs=rngs)

        self.fnn_dropout = nnx.Dropout(fnn_dropout_rate, rngs=rngs)
        self.attention_dropout = nnx.Dropout(attention_dropout_rate, rngs=rngs)

    def _Call(self, x: jax.Array, atten: jax.Array):
        x = x + self.attention_dropout(atten)
        y = self.fnn(self.pre_fnn_norm(x))
        return x + self.fnn_dropout(y)

    def __call__(self, x: jax.Array, mask: tp.Optional[jax.Array] = None, offset: int = 0):
        return self._Call(x, self.attention(self.pre_attention_norm(x), mask=mask))

    def CachedDecode(self, x: jax.Array, mask: tp.Optional[jax.Array] = None):
        return self._Call(x, self.attention.CachedDecode(self.pre_attention_norm(x), mask=mask))

    def PrefillKVCache(self, x: jax.Array, fill_count):
        return self._Call(x, self.attention.PrefillKVCache(self.pre_attention_norm(x), fill_count))


T = tp.TypeVar("T")


def _LinkParams(inst: T, target: T, ignore: tp.Sequence = []):
    if type(inst) != type(target):
        raise ValueError("Expect same type for inst and target")
    if not isinstance(ignore, tuple):
        ignore = (*ignore,)
    for name, value in vars(target).items():
        if isinstance(value, ignore):
            continue
        elif isinstance(value, nnx.Module):
            _LinkParams(getattr(inst, name), value)
        elif isinstance(value, nnx.Param):
            setattr(inst, name, value)


# Get the count of tokens before the first end_token
def _GetPrefillKVCount(x: jax.Array, end_token: int):
    end_token_poses = jnp.argmax(x == end_token, axis=-1)
    return jnp.min(end_token_poses)


class Transformer(nnx.Module):
    def __init__(
        self,
        num_embeddings: int,
        model_features: int,
        num_heads: int,
        num_decoders: int,
        max_seq_len: int,
        *,
        rngs: nnx.Rngs,
        token_config: TokenConfig,
        use_loop: bool = False,
        decoder_droprate: float = 0.1,
        param_dtype: Dtype = jnp.float32,
        dtype: tp.Optional[Dtype] = None,
    ):
        self.model_features = model_features
        self.num_heads = num_heads
        self.token_config = token_config
        self.max_seq_len = max_seq_len
        self.param_dtype = param_dtype
        self.dtype = dtype

        self.embeddings = nnx.Embed(
            num_embeddings, model_features, rngs=rngs, param_dtype=param_dtype, dtype=dtype
        )
        self.rope_cache = RoPE(model_features, max_seq_len, dtype=param_dtype)
        init_decoder = lambda: RoPETransformerBlock(
            model_features,
            num_heads,
            self.rope_cache,
            rngs=rngs,
            fnn_dropout_rate=decoder_droprate,
            param_dtype=param_dtype,
            dtype=dtype,
        )
        if not use_loop:
            self.decoders = nnx.List([init_decoder() for _ in range(num_decoders)])
        else:
            self.decoders = nnx.List[RoPETransformerBlock]()
            ignore = (nnx.LayerNorm, nnx.Dropout)
            for _ in range(num_decoders):
                self.decoders.append(init_decoder())
                self.decoders.append(init_decoder())
                _LinkParams(self.decoders[-1], self.decoders[-2], ignore=ignore)

        self.output_proj = nnx.Linear(
            model_features, num_embeddings, rngs=rngs, param_dtype=param_dtype, dtype=dtype
        )

    def __call__(self, x: jax.Array, mask: jax.Array):
        mask = mask[:, None, :, :]
        x = self.embeddings(x)
        for decoder in self.decoders:
            x = decoder(x, mask=mask)
        return self.output_proj(x)

    def InitKVCache(self, batch_size: int):
        for decoder in self.decoders:
            decoder.attention.InitKVCache(batch_size, self.max_seq_len)

    def ResetKVCache(self):
        for decoder in self.decoders:
            decoder.attention.ResetKVCache()

    def ReleaseKVCache(self):
        for decoder in self.decoders:
            decoder.attention.ReleaseKVCache()

    def Generate(self, x: jax.Array):
        self.eval()
        self.ResetKVCache()

        finish = jnp.full((x.shape[0],), False)
        max_seq_len = x.shape[1]
        output = x

        prefill_count = _GetPrefillKVCount(x, self.token_config.end_token)

        x = self.embeddings(x)

        def get_next_logits(this: tp.Self, x, i):
            emb_tokens = jax.lax.dynamic_slice(x, (0, i, 0), (x.shape[0], 1, x.shape[2]))
            for decoder in this.decoders:
                emb_tokens = decoder.CachedDecode(emb_tokens)
            return jnp.argmax(self.output_proj(emb_tokens), axis=-1)

        def while_body(val):
            i, finish, output, x, this = val
            next_tokens = get_next_logits(this, x, i)

            need_append = (~finish) & (output[:, i + 1] == self.token_config.padding_token)
            output = output.at[:, i + 1].set(
                jnp.where(
                    need_append,
                    next_tokens[:, 0],
                    output[:, i + 1],
                )
            )
            x = x.at[:, i + 1].set(
                jnp.where(
                    need_append[:, None],
                    this.embeddings(next_tokens[:, 0]),
                    x[:, i + 1],
                )
            )

            finish |= (next_tokens[:, 0] == self.token_config.end_token) & need_append

            return i + 1, finish, output, x, this

        def while_cond(val):
            i, finish, output, x, this = val
            return (i < max_seq_len) & (~finish.all())

        curr_emb = x
        for decoder in self.decoders:
            curr_emb = decoder.PrefillKVCache(curr_emb, prefill_count)

        _, _, output, _, _ = jax.lax.while_loop(
            while_cond, while_body, (prefill_count, finish, output, x, self)
        )

        return output
