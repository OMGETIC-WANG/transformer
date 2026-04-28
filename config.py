from ml_collections import ConfigDict


def get_config():
    config = ConfigDict()

    config.seed = 666

    config.param_dtype = "bfloat16"
    config.dtype = "bfloat16"

    config.test_only = False
    config.use_training_model = False
    config.train_state_path = "./cache/latest.trainstate"

    config.train_batch_size = 256
    config.epoch_count = 200

    config.test_batch_size = 256

    config.init_learning_rate = 0.0
    config.peek_learning_rate = 0.002
    config.end_learning_rate = 0.0001
    config.warmup_steps = -1  # prior than warmup_steps_percent, default = total_steps // 10
    config.warmup_steps_percent = -1.0  # default use warmup_steps
    config.adamw_weight_decay = 1e-2

    config.state_save_per_epoch = 100
    config.model_save_dir = "./cache"
    config.model_suffix = "model"

    config.eval_per_epoch = 1
    config.use_graphic = True

    # Token settings
    config.start_token = 31
    config.end_token = 30
    config.padding_token = 29
    config.think_start_token = 28
    config.think_end_token = 27
    config.seperate_token = 26
    config.mul_token = 25
    config.equal_token = 24
    config.zero_token = 0

    # Data settings
    config.seq_len = 64
    config.num_digits = 3
    config.trainset_size = 65536
    config.testset_size = 8192
    config.res_seq_reverse = True

    # Model settings
    config.num_embeddings = 32
    config.model_features = 128
    config.num_heads = 8
    config.num_decoders = 8
    config.decoder_droprate = 0.4

    config.use_graphic = True

    config.enable_optimization = True

    return config
