from model import TokenConfig
import numpy as np
import typing as tp
import os
import pickle
import jax
import jax.numpy as jnp


class CotMulSeqDataMetaData(tp.NamedTuple):
    token_config: TokenConfig
    seq_len: int
    data_count: int
    res_seq_reverse: bool = False


def SplitNum(x: int) -> list[int]:
    """Example: 123 -> [100, 20, 3]"""
    if x == 0:
        return [0]
    res: list[int] = []
    while x > 0:
        res.append(x % 10)
        x //= 10
    res.reverse()
    scale = 1
    for i in range(len(res)):
        res[len(res) - 1 - i] *= scale
        scale *= 10
    return res


def ToList(x: int, config: TokenConfig, reverse: bool = False) -> list[int]:
    if x == 0:
        return [config.zero_token]
    res = []
    while x > 0:
        res.append(config.zero_token + x % 10)
        x //= 10
    if not reverse:
        res.reverse()
    return res


def PadSeq(seq: list[int], max_len: int, padding_token: int) -> list[int]:
    return seq + [padding_token] * (max_len - len(seq))


def MakeSeq(
    max_len: int, a: int, b: int, config: TokenConfig, res_seq_reverse: bool
) -> tuple[list[int], list[int]]:
    c = a * b

    res_x = [
        config.start_token,
        *ToList(a, config),
        config.mul_token,
        *ToList(b, config),
        config.end_token,
    ]
    res_y = res_x + [config.think_start_token]

    b_splited = SplitNum(b)
    parts = []
    for i in range(len(b_splited)):
        if b_splited[i] == 0:
            continue
        parts.append([
            *ToList(a, config),
            config.mul_token,
            *ToList(b_splited[i], config),
            config.equal_token,
            *ToList(a * b_splited[i], config, reverse=res_seq_reverse),
        ])
    for i in range(len(parts)):
        res_y += parts[i]
        if i != len(parts) - 1:
            res_y += [config.seperate_token]

    res_y += [
        config.think_end_token,
        config.start_token,
        *ToList(c, config, reverse=res_seq_reverse),
        config.end_token,
    ]
    return PadSeq(res_x, max_len, config.padding_token), PadSeq(
        res_y, max_len, config.padding_token
    )


def BuildCotMulSeqData(config: TokenConfig, seq_len: int, nums: np.ndarray, res_seq_reverse: bool):
    x = np.full((nums.shape[0], seq_len), config.padding_token, dtype=np.int32)
    y = np.full((nums.shape[0], seq_len), config.padding_token, dtype=np.int32)

    for i in range(nums.shape[0]):
        x[i], y[i] = MakeSeq(seq_len, nums[i, 0], nums[i, 1], config, res_seq_reverse)

    return jnp.array(x), jnp.array(y)


def TryLoad(
    x_path: str,
    y_path: str,
    metadata_path: str,
    data_count: int,
    config: TokenConfig,
    seq_len: int,
    res_seq_reverse: bool,
):
    if os.path.exists(metadata_path) and os.path.exists(x_path) and os.path.exists(y_path):
        with open(metadata_path, "rb") as f:
            meta: CotMulSeqDataMetaData = pickle.load(f)
        if (
            meta.token_config == config
            and meta.seq_len == seq_len
            and meta.data_count >= data_count
            and meta.res_seq_reverse == res_seq_reverse
        ):
            x = jnp.load(x_path)
            y = jnp.load(y_path)
            return x[:data_count], y[:data_count]
    return None


def LoadCotMulSeqData(
    dataset_folder: str,
    config: TokenConfig,
    num_digits: int,
    seq_len: int,
    data_count: int,
    res_seq_reverse: bool,
) -> tuple[jax.Array, jax.Array]:
    x_path = os.path.join(dataset_folder, "data.x.npy")
    y_path = os.path.join(dataset_folder, "data.y.npy")
    metadata_path = os.path.join(dataset_folder, "data.metadata")

    try:
        data = TryLoad(x_path, y_path, metadata_path, data_count, config, seq_len, res_seq_reverse)
    except Exception as e:
        data = None
    if data is not None:
        x, y = data
        return x[:data_count], y[:data_count]

    def remove_if_exists(file_path: str):
        if os.path.exists(file_path):
            os.remove(file_path)

    map(
        remove_if_exists,
        [x_path, y_path, metadata_path],
    )

    os.makedirs(os.path.dirname(dataset_folder), exist_ok=True)

    if data_count > (10**num_digits) ** 2:
        raise ValueError("train_count + test_count must be less than or equal to (10^num_digits)^2")

    indices = np.random.choice((10**num_digits) ** 2, size=data_count, replace=False)
    rows = indices // (10**num_digits)
    cols = indices % (10**num_digits)
    nums = np.stack([rows, cols], axis=1)

    x, y = BuildCotMulSeqData(config, seq_len, nums, res_seq_reverse)

    os.makedirs(os.path.dirname(x_path), exist_ok=True)
    os.makedirs(os.path.dirname(y_path), exist_ok=True)
    jnp.save(x_path, x)
    jnp.save(y_path, y)
    with open(metadata_path, "wb") as f:
        pickle.dump(
            CotMulSeqDataMetaData(
                token_config=config,
                seq_len=seq_len,
                data_count=data_count,
                res_seq_reverse=res_seq_reverse,
            ),
            f,
        )

    return x, y
