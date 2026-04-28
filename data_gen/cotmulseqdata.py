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


def MakeSeq(max_len: int, a: int, b: int, config: TokenConfig) -> tuple[list[int], list[int]]:
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
    return PadSeq(res_x, max_len, config.padding_token), PadSeq(
        res_y, max_len, config.padding_token
    )


def BuildCotMulSeqData(config: TokenConfig, num_digits: int, seq_len: int, nums: np.ndarray):
    x = np.full((nums.shape[0], seq_len), config.padding_token, dtype=np.int32)
    y = np.full((nums.shape[0], seq_len), config.padding_token, dtype=np.int32)

    for i in range(nums.shape[0]):
        x[i], y[i] = MakeSeq(seq_len, nums[i, 0], nums[i, 1], config)

    return jnp.array(x), jnp.array(y)


def TryLoad(
    x_path: str,
    y_path: str,
    metadata_path: str,
    metadata: CotMulSeqDataMetaData,
    data_count: int,
    config: TokenConfig,
    seq_len: int,
):
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
    return None


def LoadCotMulSeqDataset(
    dataset_folder: str,
    config: TokenConfig,
    num_digits: int,
    seq_len: int,
    train_count: int,
    test_count: int,
):
    train_x_path = os.path.join(dataset_folder, "train.x.npy")
    train_y_path = os.path.join(dataset_folder, "train.y.npy")
    train_metadata_path = os.path.join(dataset_folder, "train.metadata")
    test_x_path = os.path.join(dataset_folder, "test.x.npy")
    test_y_path = os.path.join(dataset_folder, "test.y.npy")
    test_metadata_path = os.path.join(dataset_folder, "test.metadata")

    train_data = TryLoad(
        train_x_path,
        train_y_path,
        train_metadata_path,
        CotMulSeqDataMetaData(config, seq_len, train_count),
        train_count,
        config,
        seq_len,
    )
    test_data = TryLoad(
        test_x_path,
        test_y_path,
        test_metadata_path,
        CotMulSeqDataMetaData(config, seq_len, test_count),
        test_count,
        config,
        seq_len,
    )
    if train_data is not None and test_data is not None:
        return train_data, test_data

    def remove_if_exists(file_path: str):
        if os.path.exists(file_path):
            os.remove(file_path)

    map(
        remove_if_exists,
        [
            train_x_path,
            train_y_path,
            train_metadata_path,
            test_x_path,
            test_y_path,
            test_metadata_path,
        ],
    )

    os.makedirs(os.path.dirname(dataset_folder), exist_ok=True)

    if train_count + test_count > (10**num_digits) ** 2:
        raise ValueError("train_count + test_count must be less than or equal to (10^num_digits)^2")

    indices = np.random.choice((10**num_digits) ** 2, size=train_count + test_count, replace=False)
    rows = indices // (10**num_digits)
    cols = indices % (10**num_digits)
    nums = np.stack([rows, cols], axis=1)

    x, y = BuildCotMulSeqData(config, num_digits, seq_len, nums)
    jnp.save(train_x_path, x[:train_count])
    jnp.save(train_y_path, y[:train_count])
    with open(train_metadata_path, "wb") as f:
        pickle.dump(
            CotMulSeqDataMetaData(token_config=config, seq_len=seq_len, data_count=train_count), f
        )

    jnp.save(test_x_path, x[train_count:])
    jnp.save(test_y_path, y[train_count:])
    with open(test_metadata_path, "wb") as f:
        pickle.dump(
            CotMulSeqDataMetaData(token_config=config, seq_len=seq_len, data_count=test_count), f
        )

    return (x[:train_count], y[:train_count]), (x[train_count:], y[train_count:])
