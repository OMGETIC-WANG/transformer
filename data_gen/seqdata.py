from model import TokenConfig
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import os
import typing as tp
import pickle
import numpy as np


class SeqDataMetaData(tp.NamedTuple):
    token_config: TokenConfig
    seq_len: int
    data_count: int


def LoadSeqData(path: str, config: TokenConfig, seq_len: int, data_count: int, rngs: nnx.Rngs):
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

    x, y = BuildRevSeqData(config, seq_len, data_count, rngs)
    jnp.save(x_path, x)
    jnp.save(y_path, y)
    with open(metadata_path, "wb") as f:
        pickle.dump(SeqDataMetaData(config, seq_len, data_count), f)
    return x, y


# return: x,y
# x format: <start_token> x_1 x_2 ... x_N <end_token> <padding_token> ...
# y format: <start_token> x_1 x_2 ... x_N <end_token> <start_token> x_N x_(N-1) ... x_1 <end_token> <padding_token> ...
# x.shape=y.shape=(data_count,seq_len)
def BuildRevSeqData(
    config: TokenConfig, seq_len: int, data_count: int, rngs: nnx.Rngs
) -> tuple[jax.Array, jax.Array]:
    if seq_len < 4:
        raise ValueError("seq_len must be at least 4")

    # 使用 NumPy 收集 Valid Tokens
    valid_tokens = np.array(
        [
            token_id
            for token_id in range(config.num_embeddings)
            if token_id not in {config.start_token, config.end_token, config.padding_token}
        ],
        dtype=np.int32,
    )
    if valid_tokens.size == 0:
        raise ValueError("TokenConfig must provide at least one non-special token")

    if data_count == 0:
        empty = jnp.empty((0, seq_len), dtype=jnp.int32)
        return empty, empty

    max_content_len = (seq_len - 4) // 2
    min_content_len = 0 if max_content_len == 0 else 1

    # 利用 JAX RNG 生成随机种子，传递给高效的 numpy.random.Generator
    seed = int(jax.random.randint(rngs.params(), (), minval=0, maxval=2**31 - 1).item())
    rng = np.random.default_rng(seed)

    # 预分配全为 padding 的 x 和 y
    x = np.full((data_count, seq_len), config.padding_token, dtype=np.int32)
    y = np.full((data_count, seq_len), config.padding_token, dtype=np.int32)

    # 一次性生成所有 sequence 对应的 content_len
    content_lengths = rng.integers(min_content_len, max_content_len + 1, size=data_count)

    # 设置所有的特殊 token 标志 (利用 NumPy 高级索引一次性赋值)
    row_indices = np.arange(data_count)

    x[:, 0] = config.start_token
    x[row_indices, 1 + content_lengths] = config.end_token

    y[:, 0] = config.start_token
    y[row_indices, 1 + content_lengths] = config.end_token
    y[row_indices, 2 + content_lengths] = config.start_token
    y[row_indices, 3 + 2 * content_lengths] = config.end_token

    if max_content_len > 0:
        # 批量生成所有可能存在的 random tokens (超出的部分会被后续的 mask 忽略掉)
        random_tokens = rng.choice(valid_tokens, size=(data_count, max_content_len))

        # 创建遮罩并抽取需要赋值的坐标系
        cols = np.arange(max_content_len)
        mask = cols < content_lengths[:, None]  # 形状为 (data_count, max_content_len)

        # r_idx 是数据的行号， c_idx 对应 content 长度里的相对偏移
        r_idx, c_idx = np.where(mask)

        # ----- 处理正向 Tokens -----
        forward_c_idx = 1 + c_idx
        x[r_idx, forward_c_idx] = random_tokens[r_idx, c_idx]
        y[r_idx, forward_c_idx] = random_tokens[r_idx, c_idx]

        # ----- 处理反向 Tokens -----
        # 对于给定的第 i 行，相对坐标 c_idx 会被映射成从后往前拿： L_i - 1 - c_idx
        rev_c_idx = content_lengths[r_idx] - 1 - c_idx

        # y 中放置反向 token 的绝对起始坐标是 3 + content_length
        y_c_idx = 3 + content_lengths[r_idx] + c_idx

        # 精准赋值反向 tokens
        y[r_idx, y_c_idx] = random_tokens[r_idx, rev_c_idx]

    # 执行最终转换，把内存连续的 Numpy 数组一口气送入 Jax 环境中
    return jnp.asarray(x), jnp.asarray(y)
