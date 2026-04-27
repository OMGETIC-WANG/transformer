from model import TokenConfig
import numpy as np
import typing as tp
import os
import pickle
import jax.numpy as jnp


class CotMulSeqDataMetaData(tp.NamedTuple):
    token_config: TokenConfig
    seq_len: int
    data_count: int


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


def ToList(x: int, config: TokenConfig) -> list[int]:
    if x == 0:
        return [config.zero_token]
    res = []
    while x > 0:
        res.append(config.zero_token + x % 10)
        x //= 10
    res.reverse()
    return res


def PadSeq(seq: list[int], max_len: int, padding_token: int) -> list[int]:
    return seq + [padding_token] * (max_len - len(seq))


def MakeSeq(max_len: int, num_digits: int, config: TokenConfig) -> tuple[list[int], list[int]]:
    while True:
        a = np.random.randint(0, 10**num_digits)
        b = np.random.randint(0, 10**num_digits)
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
                *ToList(a * b_splited[i], config),
            ])
        for i in range(len(parts)):
            res_y += parts[i]
            if i != len(parts) - 1:
                res_y += [config.seperate_token]

        res_y += [
            config.think_end_token,
            config.start_token,
            *ToList(c, config),
            config.end_token,
        ]
        if len(res_y) <= max_len:
            return PadSeq(res_x, max_len, config.padding_token), PadSeq(
                res_y, max_len, config.padding_token
            )


def BuildCotMulSeqData(config: TokenConfig, num_digits: int, seq_len: int, data_count: int):
    x = np.full((data_count, seq_len), config.padding_token, dtype=np.int32)
    y = np.full((data_count, seq_len), config.padding_token, dtype=np.int32)

    for i in range(data_count):
        x[i], y[i] = MakeSeq(seq_len, num_digits, config)

    return jnp.array(x), jnp.array(y)


def LoadCotMulSeqData(
    path: str, config: TokenConfig, num_digits: int, seq_len: int, data_count: int
):
    metadata_path = path + ".metadata"
    x_path = path + ".x.npy"
    y_path = path + ".y.npy"

    if os.path.exists(metadata_path) and os.path.exists(x_path) and os.path.exists(y_path):
        with open(metadata_path, "rb") as f:
            meta: CotMulSeqDataMetaData = pickle.load(f)
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
    os.makedirs(os.path.dirname(path), exist_ok=True)

    x, y = BuildCotMulSeqData(config, num_digits, seq_len, data_count)
    jnp.save(x_path, x)
    jnp.save(y_path, y)
    with open(metadata_path, "wb") as f:
        pickle.dump(
            CotMulSeqDataMetaData(token_config=config, seq_len=seq_len, data_count=data_count), f
        )
    return x, y
