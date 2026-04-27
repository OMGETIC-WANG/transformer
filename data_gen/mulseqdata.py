from model import TokenConfig
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import os
import typing as tp
import pickle
import numpy as np


class MulSeqDataMetaData(tp.NamedTuple):
    token_config: TokenConfig
    seq_len: int
    data_count: int


def LoadMulSeqData(path: str, config: TokenConfig, seq_len: int, data_count: int, rngs: nnx.Rngs):
    metadata_path = path + ".metadata"
    x_path = path + ".x.npy"
    y_path = path + ".y.npy"

    if os.path.exists(metadata_path) and os.path.exists(x_path) and os.path.exists(y_path):
        with open(metadata_path, "rb") as f:
            meta = pickle.load(f)
        if (
            meta.token_config == config
            and meta.seq_len == seq_len
            and meta.data_count >= data_count
        ):
            x = jnp.load(x_path)
            y = jnp.load(y_path)
            return x[:data_count], y[:data_count]

    if os.path.exists(x_path):
        os.remove(x_path)
    if os.path.exists(y_path):
        os.remove(y_path)
    if os.path.exists(metadata_path):
        os.remove(metadata_path)
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    x, y = BuildMulSeqData(config, seq_len, data_count, rngs)
    jnp.save(x_path, x)
    jnp.save(y_path, y)
    with open(metadata_path, "wb") as f:
        pickle.dump(MulSeqDataMetaData(config, seq_len, data_count), f)
    return x, y


# return: x,y
# x format: <start_token> a_1 a_2 ... a_N <mul_token> b_1 b_2 ... b_M <end_token> <padding_token> ...
# y format: <start_token> a_1 a_2 ... a_N <mul_token> b_1 b_2 ... b_M <end_token> <start_token> c_1 c_2 ... c_K <end_token> <padding_token> ...
# x.shape=y.shape=(data_count,seq_len)
def BuildMulSeqData(
    config: TokenConfig, seq_len: int, data_count: int, rngs: nnx.Rngs
) -> tuple[jax.Array, jax.Array]:
    if seq_len < 9:
        raise ValueError("seq_len must be at least 9")

    base = 10

    if data_count == 0:
        empty = jnp.empty((0, seq_len), dtype=jnp.int32)
        return empty, empty

    # 利用更新后的 TokenConfig 直接获取数字与乘号的 Token
    digit_tokens = np.arange(config.zero_token, config.zero_token + base)
    mul_token = config.mul_token

    max_ab_len = (seq_len - 5) // 2

    # 利用 JAX RNG 生成随机种子，传递给高效的 numpy.random.Generator 以及 Python 原生的 Random
    seed = int(jax.random.randint(rngs.params(), (), minval=0, maxval=2**31 - 1).item())
    rng = np.random.default_rng(seed)

    import random

    py_rng = random.Random(seed)

    # 预分配全为 padding 的 x 和 y
    x = np.full((data_count, seq_len), config.padding_token, dtype=np.int32)
    y = np.full((data_count, seq_len), config.padding_token, dtype=np.int32)

    L_ab_total = rng.integers(2, max_ab_len + 1, size=data_count)
    L_a = rng.integers(1, L_ab_total, size=data_count)
    L_b = L_ab_total - L_a

    for i in range(data_count):
        la = int(L_a[i])
        lb = int(L_b[i])

        # 随机生成 A 和 B 的值 (确保长度恰好等于 la 和 lb)
        a_val = py_rng.randint(base ** (la - 1) if la > 1 else 0, base**la - 1)
        b_val = py_rng.randint(base ** (lb - 1) if lb > 1 else 0, base**lb - 1)
        c_val = a_val * b_val

        # 提取 A 的十进制表示
        if a_val == 0:
            a_dig = [0]
        else:
            a_dig = []
            temp = a_val
            while temp > 0:
                a_dig.append(temp % base)
                temp //= base
            a_dig.reverse()

        # 提取 B 的十进制表示
        if b_val == 0:
            b_dig = [0]
        else:
            b_dig = []
            temp = b_val
            while temp > 0:
                b_dig.append(temp % base)
                temp //= base
            b_dig.reverse()

        # 提取 C = A * B 的十进制表示
        if c_val == 0:
            c_dig = [0]
        else:
            c_dig = []
            temp = c_val
            while temp > 0:
                c_dig.append(temp % base)
                temp //= base
            # c_dig.reverse()

        # 映射回相应的 Token ID
        a_toks = digit_tokens[a_dig]
        b_toks = digit_tokens[b_dig]
        c_toks = digit_tokens[c_dig]

        idx = 0

        # ----- 构建 x: <start> A * B <end> -----
        x[i, idx] = config.start_token
        idx += 1
        x[i, idx : idx + len(a_toks)] = a_toks
        idx += len(a_toks)
        x[i, idx] = mul_token
        idx += 1
        x[i, idx : idx + len(b_toks)] = b_toks
        idx += len(b_toks)
        x[i, idx] = config.end_token
        idx += 1

        # ----- y 的前一部分等于 x 的有效部分 -----
        y[i, :idx] = x[i, :idx]

        # ----- 构建 y 的剩余部分: <start> C <end> -----
        y[i, idx] = config.start_token
        idx += 1
        y[i, idx : idx + len(c_toks)] = c_toks
        idx += len(c_toks)
        y[i, idx] = config.end_token

    # 执行最终转换，把内存连续的 Numpy 数组一口气送入 Jax 环境中
    return jnp.asarray(x), jnp.asarray(y)
