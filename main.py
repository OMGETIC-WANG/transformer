from model import Transformer, TokenConfig

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
import typing as T
import masks
import data_gen.cotmulseqdata as cotmulseqdata

import os
import time
import model_serialization
from dashboard import Dashboard
import sys
import matplotlib.pyplot as plt

from ml_collections import config_flags
from absl import app


def BatchDatas(xs: T.Sequence[jax.Array], batch_size: int):
    dataset_size = xs[0].shape[0]
    batch_count = dataset_size // batch_size
    if dataset_size % batch_size != 0:
        sys.stdout.write(
            f"Warning: dataset size {dataset_size} % batch size {batch_size} != 0, {dataset_size % batch_size} data will not be used\n"
        )
    return [
        x[: batch_count * batch_size].reshape(batch_count, batch_size, *x.shape[1:]) for x in xs
    ]


@nnx.jit
def TrainBatch(
    model_optimizer: tuple[Transformer, nnx.Optimizer[Transformer]],
    x: jax.Array,
    y: jax.Array,
):
    model, optimzier = model_optimizer

    def loss_fn(model: Transformer):
        causal_mask = masks.CausalMask(y)
        padding_mask = masks.PaddingMask(y, model.token_config.padding_token)
        mask = causal_mask & padding_mask[..., None]

        pred_mask = masks.PredMask(y, model.token_config.end_token)

        logits = model(y, mask=mask)
        target_y = jnp.roll(y, -1, axis=-1)
        target = nnx.one_hot(target_y, logits.shape[-1])
        loss = (optax.softmax_cross_entropy(logits, target) * pred_mask).sum() / pred_mask.sum()

        pred = jnp.argmax(logits, axis=-1)

        match = (pred == target_y) & pred_mask
        accuracy = match.sum() / pred_mask.sum()

        full_match = ~jnp.any((pred != target_y) & pred_mask, axis=-1)
        full_accuracy = full_match.mean()

        answer_pred_mask = masks.PredAnswerMask(y, pred_mask, model.token_config.think_end_token)
        answer_match = ~jnp.any((pred != target_y) & answer_pred_mask, axis=-1)
        answer_accuracy = answer_match.mean()

        return loss, (full_accuracy, accuracy, answer_accuracy)

    (loss, (full_accuracy, accuracy, answer_accuracy)), grads = nnx.value_and_grad(
        loss_fn, has_aux=True
    )(model)
    optimzier.update(model, grads)
    return (model, optimzier), loss, full_accuracy, accuracy, answer_accuracy


@nnx.jit(static_argnames=["batch_size"])
def TrainModel(
    model: Transformer,
    optimizer: nnx.Optimizer[Transformer],
    x: jax.Array,
    y: jax.Array,
    batch_size: int,
    *,
    rngs: nnx.Rngs,
    metrics: nnx.Metric,
):

    indices = jnp.arange(x.shape[0])
    indices = jax.random.permutation(rngs.params(), indices)
    x, y = x[indices], y[indices]  # Shuffle

    x, y = BatchDatas((x, y), batch_size)

    _, losses, full_accuracies, token_accuracies, answer_accuracies = nnx.scan(
        TrainBatch, in_axes=(nnx.Carry, 0, 0), out_axes=(nnx.Carry, 0, 0, 0, 0)
    )((model, optimizer), x, y)

    metrics.update(
        values=losses,
        accuracy=full_accuracies,
        token_accuracy=token_accuracies,
        answer_accuracy=answer_accuracies,
    )


@nnx.scan(in_axes=(None, 0, 0), out_axes=0)
@nnx.jit
def TestBatch(model: Transformer, x: jax.Array, y: jax.Array):
    pred = model.Generate(x)

    response_mask = masks.ResponseMask(x, y, model.token_config.padding_token)
    matches = (pred == y) & response_mask

    answer_mask = masks.AnswerMask(y, response_mask, model.token_config.seperate_token)
    answer_matches = matches & answer_mask

    full_match = ~jnp.any(
        (pred != y) & response_mask, axis=-1
    )  # full_match = No any mismatch in response
    token_match = matches.sum(axis=-1) / response_mask.sum(axis=-1)
    answer_match = ~jnp.any((pred != y) & answer_mask, axis=-1)
    return full_match, token_match, answer_match


@nnx.jit(static_argnames=["batch_size"])
def TestModel(model: Transformer, x: jax.Array, y: jax.Array, batch_size: int):
    model.eval()

    x, y = BatchDatas((x, y), batch_size)

    full_match, token_match, answer_match = TestBatch(model, x, y)
    return full_match.mean(), token_match.mean(), answer_match.mean()


def Train(
    model: Transformer,
    optimizer: nnx.Optimizer[Transformer],
    x: jax.Array,
    y: jax.Array,
    batch_size: int,
    epoch_count: int,
    *,
    rngs: nnx.Rngs,
    x_test: jax.Array,
    y_test: jax.Array,
    test_batch_size: int = -1,
    state_save_path: str = "",
    state_save_per_epoch: int = -1,
    use_graphic: bool = True,
    eval_per_epoch: int = 1,
):
    train_metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average(),
        accuracy=nnx.metrics.Average("accuracy"),
        token_accuracy=nnx.metrics.Average("token_accuracy"),
        answer_accuracy=nnx.metrics.Average("answer_accuracy"),
    )

    enable_eval = eval_per_epoch > 0 and test_batch_size > 0
    if enable_eval:
        if test_batch_size is None:
            test_batch_size = batch_size
        model.InitKVCache(test_batch_size)

    enable_state_saving = state_save_path != "" and state_save_per_epoch > 0

    if use_graphic:
        dashboard = Dashboard(
            "Dashboard",
            {
                "Loss": ["loss"],
                "Accuracy": [
                    "accuracy",
                    "token_accuracy",
                    "answer_accuracy",
                    "test_accuracy",
                    "test_token_accuracy",
                    "test_answer_accuracy",
                ],
            },
        )
    else:
        dashboard = None

    for epoch in range(epoch_count):
        train_start_time = time.time()
        TrainModel(model, optimizer, x, y, batch_size, rngs=rngs, metrics=train_metrics)
        jax.block_until_ready(model)
        train_end_time = time.time()
        train_time = train_end_time - train_start_time

        epoch_metrics = train_metrics.compute()
        train_metrics.reset()

        if enable_state_saving:
            if (epoch + 1) % state_save_per_epoch == 0:
                model_serialization.SaveTrainingState(
                    state_save_path,
                    model,
                    optimizer,
                )

        loss_plot_dict = {
            "loss": epoch_metrics["loss"],
            "accuracy": epoch_metrics["accuracy"],
            "token_accuracy": epoch_metrics["token_accuracy"],
            "answer_accuracy": epoch_metrics["answer_accuracy"],
        }

        epoch_msg = (
            f"loss: {epoch_metrics['loss']:.4f},"
            + f" acc: {epoch_metrics['accuracy']:.4f}, token_acc: {epoch_metrics['token_accuracy']:.4f}, answer_acc: {epoch_metrics['answer_accuracy']:.4f}"
        )

        eval_time = -1.0
        if enable_eval and epoch % eval_per_epoch == 0:
            eval_start_time = time.time()
            test_results = TestModel(model, x_test, y_test, test_batch_size)
            jax.block_until_ready(test_results)
            test_full_acc, test_token_acc, test_answer_acc = test_results
            eval_end_time = time.time()
            eval_time = eval_end_time - eval_start_time

            model.train()
            loss_plot_dict["test_accuracy"] = test_full_acc
            loss_plot_dict["test_token_accuracy"] = test_token_acc
            loss_plot_dict["test_answer_accuracy"] = test_answer_acc
            epoch_msg += f", test_acc: {test_full_acc:.4f}, test_token_acc: {test_token_acc:.4f}, test_answer_acc: {test_answer_acc:.4f}"

        sys.stdout.write(
            f"Epoch {epoch + 1}/{epoch_count} ({train_time:.2f}s|{eval_time:.2f}s) - {epoch_msg}\n"
        )

        if dashboard is not None:
            dashboard.Update(loss_plot_dict)


def CountModuleParams(module: nnx.Module):
    params = nnx.state(module, nnx.Param)
    leaves = jax.tree_util.tree_leaves(params)
    return sum(leaf.size for leaf in leaves)
    # unique_leaves = {id(leaf): leaf for leaf in leaves}.values()
    # return sum(leaf.size for leaf in unique_leaves)


_CONFIG = config_flags.DEFINE_config_file(
    "config", "config.py", "Configuration file for training the model."
)


def EnableJaxOptimization(precision: str = "float16"):
    xla_flags = [
        "--xla_gpu_triton_gemm_any=true",
        "--xla_gpu_enable_latency_hiding_scheduler=true",
        "--xla_gpu_enable_highest_priority_async_stream=true",
        "--xla_gpu_all_gather_combine_threshold_bytes=1073741824",
    ]

    existing_flags = os.environ.get("XLA_FLAGS", "")
    os.environ["XLA_FLAGS"] = existing_flags + " " + " ".join(xla_flags)

    jax.config.update("jax_enable_x64", False)
    jax.config.update("jax_enable_pgle", True)

    jax.config.update("jax_default_matmul_precision", "tensorfloat32")


def LoadCotMulSeqDataset(
    dataset_dir: str, token_config: TokenConfig, config
) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
    x_data, y_data = cotmulseqdata.LoadCotMulSeqData(
        dataset_dir,
        token_config,
        config.num_digits,
        config.seq_len,
        config.trainset_size + config.testset_size,
        config.res_seq_reverse,
    )
    return (x_data[: config.trainset_size], y_data[: config.trainset_size]), (
        x_data[config.trainset_size :],
        y_data[config.trainset_size :],
    )


def main(_):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    config = _CONFIG.value

    token_config = TokenConfig(
        num_embeddings=config.num_embeddings,
        start_token=config.start_token,
        end_token=config.end_token,
        padding_token=config.padding_token,
        think_start_token=config.think_start_token,
        think_end_token=config.think_end_token,
        seperate_token=config.seperate_token,
        mul_token=config.mul_token,
        equal_token=config.equal_token,
        zero_token=config.zero_token,
    )

    if config.enable_optimization:
        EnableJaxOptimization()

    sys.stdout.write(f"-----Config-----\n{config}----------------\n")

    sys.stdout.write("Initing model\n")
    rngs = nnx.Rngs(config.seed)

    sys.stdout.write("Loading data\n")
    (x_train, y_train), (x_test, y_test) = LoadCotMulSeqDataset(
        config.dataset_dir, token_config, config
    )

    init_model_with_rngs = lambda rngs: Transformer(
        num_embeddings=config.num_embeddings,
        model_features=config.model_features,
        num_heads=config.num_heads,
        num_decoders=config.num_decoders,
        max_seq_len=config.seq_len,
        rngs=rngs,
        token_config=token_config,
        use_loop=config.use_loop,
        decoder_droprate=config.decoder_droprate,
        param_dtype=config.param_dtype,
        dtype=config.dtype,
    )
    init_model = lambda: init_model_with_rngs(rngs)
    init_model_abstract = lambda: init_model_with_rngs(nnx.Rngs(0))

    if not config.test_only:
        if config.use_training_model:
            model, optimizer = model_serialization.LoadTrainingState(
                config.train_state_path,
                init_model_abstract,
                lambda model: nnx.Optimizer(model, optax.adamw(0.001), wrt=nnx.Param),
            )
        else:
            model = init_model()

            total_steps = config.epoch_count * (config.trainset_size // config.train_batch_size)
            if config.warmup_steps != -1:
                warmup_steps = config.warmup_steps
            elif config.warmup_steps_percent != -1.0:
                warmup_steps = int(total_steps * config.warmup_steps_percent)
            else:
                warmup_steps = total_steps // 10

            optimizer_schedule = optax.warmup_cosine_decay_schedule(
                init_value=config.init_learning_rate,
                peak_value=config.peek_learning_rate,
                end_value=config.end_learning_rate,
                warmup_steps=warmup_steps,
                decay_steps=total_steps,
            )
            optimizer = nnx.Optimizer(
                model,
                optax.adamw(optimizer_schedule, weight_decay=config.adamw_weight_decay),
                wrt=nnx.Param,
            )

        param_count = CountModuleParams(model)
        sys.stdout.write(f"Model param count: {param_count} ({param_count / 1e6:.2f}M)\n")
        sys.stdout.write("Starting training\n")
        Train(
            model,
            optimizer,
            x_train,
            y_train,
            config.train_batch_size,
            config.epoch_count,
            rngs=rngs,
            x_test=x_test,
            y_test=y_test,
            test_batch_size=config.test_batch_size,
            state_save_path=config.train_state_path,
            state_save_per_epoch=config.state_save_per_epoch,
            eval_per_epoch=config.eval_per_epoch,
            use_graphic=config.use_graphic,
        )
        if config.model_save_dir != "":
            model.ReleaseKVCache()
            model_serialization.SaveModel(
                os.path.join(config.model_save_dir, f"{time.time()}.{config.model_suffix}"),
                model,
            )
    else:
        model = model_serialization.LoadNewestModel(
            config.model_save_dir,
            config.model_suffix,
            init_model_abstract,
        )
        model.InitKVCache(config.test_batch_size)
        sys.stdout.write("Start testing\n")
        model.eval()
        full_acc, token_acc, answer_acc = TestModel(model, x_test, y_test, config.test_batch_size)
        sys.stdout.write(f"Test accuracy: {full_acc * 100:.4f}%\n")
        sys.stdout.write(f"Test token accuracy: {token_acc * 100:.4f}%\n")
        sys.stdout.write(f"Test answer accuracy: {answer_acc * 100:.4f}%\n")

    if config.use_graphic:
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    app.run(main)
