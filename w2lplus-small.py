# pylint: skip-file
import tensorflow as tf
from easyasr.models import Speech2Text
from easyasr.encoders import TDNNEncoder
from easyasr.decoders import FullyConnectedCTCDecoder
from easyasr.data import Speech2TextTFRecordLayer
from easyasr.losses import CTCLoss
from easyasr.optimizers.lr_policies import poly_decay

base_model = Speech2Text

base_params = {
    "random_seed": 0,
    "max_steps": 1000,
    "batch_size_per_gpu": 8,
    "save_summaries_steps": 1000,
    "print_loss_steps": 10,
    "print_samples_steps": 500,
    "save_checkpoint_steps": 5000,
    "num_checkpoints": 50,
    "logdir": "",
    "load_model_ckpt": "",
    "optimizer": "Momentum",
    "optimizer_params": {
        "momentum": 0.90,
    },
    "lr_policy": poly_decay,
    "lr_policy_params": {
        "learning_rate": 0.05,
        "power": 2.0,
    },
    "larc_params": {
        "larc_eta": 0.001,
    },
    "regularizer": tf.contrib.layers.l2_regularizer,
    "regularizer_params": {
        'scale': 0.001
    },
    "dtype": tf.float32, # "mixed",
    "summaries": ['learning_rate', 'variables', 'gradients', 'larc_summaries',
                  'variable_norm', 'gradient_norm', 'global_gradient_norm'],
    "encoder": TDNNEncoder,
    "encoder_params": {
        "convnet_layers": [
            {
                "type": "conv1d", "repeat": 1,
                "kernel_size": [5], "stride": [2],
                "num_channels": 256, "padding": "SAME",
                "dilation":[1], "dropout_keep_prob": 0.8,
            },
            {
                "type": "conv1d", "repeat": 1,
                "kernel_size": [5], "stride": [1],
                "num_channels": 256, "padding": "SAME",
                "dilation":[1], "dropout_keep_prob": 0.8,
                "residual": True, "residual_dense": True
            },
            {
                "type": "conv1d", "repeat": 1,
                "kernel_size": [7], "stride": [1],
                "num_channels": 384, "padding": "SAME",
                "dilation":[1], "dropout_keep_prob": 0.8,
                "residual": True, "residual_dense": True
            },
            {
                "type": "conv1d", "repeat": 1,
                "kernel_size": [9], "stride": [1],
                "num_channels": 512, "padding": "SAME",
                "dilation":[1], "dropout_keep_prob": 0.8,
                "residual": True, "residual_dense": True
            },
            {
                "type": "conv1d", "repeat": 1,
                "kernel_size": [11], "stride": [1],
                "num_channels": 640, "padding": "SAME",
                "dilation":[1], "dropout_keep_prob": 0.7,
                "residual": True, "residual_dense": True
            },
            {
                "type": "conv1d", "repeat": 1,
                "kernel_size": [13], "stride": [1],
                "num_channels": 768, "padding": "SAME",
                "dilation":[1], "dropout_keep_prob": 0.7,
                "residual": True, "residual_dense": True
            },
            {
                "type": "conv1d", "repeat": 1,
                "kernel_size": [15], "stride": [1],
                "num_channels": 896, "padding": "SAME",
                "dilation":[2], "dropout_keep_prob": 0.6,
            },
            {
                "type": "conv1d", "repeat": 1,
                "kernel_size": [1], "stride": [1],
                "num_channels": 1024, "padding": "SAME",
                "dilation":[1], "dropout_keep_prob": 0.6,
            }
        ],

        "dropout_keep_prob": 0.7,
        "initializer": tf.contrib.layers.xavier_initializer,
        "initializer_params": {
            'uniform': False,
        },
        "normalization": "batch_norm",
        "activation_fn": lambda x: tf.minimum(tf.nn.relu(x), 20.0),
        "data_format": "channels_last",
    },

    "decoder": FullyConnectedCTCDecoder,
    "decoder_params": {
        "initializer": tf.contrib.layers.xavier_initializer,
    },
    "data_layer": Speech2TextTFRecordLayer,
    "data_layer_params": {
        "num_audio_features": 64,
        "input_type": "logfbank",
        "vocab_file": "",
        "dataset_files": [
            "",
        ],
        "shuffle": False,
    },
    "loss": CTCLoss,
    "loss_params": {},
}

train_params = {
    "data_layer": Speech2TextTFRecordLayer,
    "data_layer_params": {
        "num_audio_features": 64,
        "input_type": "logfbank",
        "vocab_file": "",
        "dataset_files": [
            "",
        ],
        "max_duration": 15.0,
        "shuffle": True,
        "num_readers": 64,
        "sample_shuffle_buf_size": 1024,
        "tfrecord_buf_size": 1024 * 1024 * 64,
        "sample_prefetch_size": 1024,
    },
}

eval_params = {
    "data_layer": Speech2TextTFRecordLayer,
    "data_layer_params": {
        "num_audio_features": 64,
        "input_type": "logfbank",
        "vocab_file": "",
        "dataset_files": [
            "",
        ],
        "shuffle": False,
    },
}