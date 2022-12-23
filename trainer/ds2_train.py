# -*- coding: utf-8 -*-
#  Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ==============================================================================
"""Main entry to train and evaluate DeepSpeech model"""

import os
import datetime
import json

# Disable all GPUs. This prevents errors caused by all workers trying to use the same GPU. 
# In a real-world application, each worker would be on a different machine.
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# Reset the 'TF_CONFIG' environment variable
#os.environ.pop('TF_CONFIG', None)

# pylint: disable=g-bad-import-order
from absl import app as absl_app
from absl import flags
from absl import logging
logging.set_verbosity(logging.ERROR)
import tensorflow as tf
#import tensorflow_cloud as tfc
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# pylint: enable=g-bad-import-order

# Local modules
import data.dataset as dataset
import decoder
from model.keras_model import ds2_model, SUPPORTED_RNNS # type: ignore

# Modules from tf-models-official
from official.utils.flags import core as flags_core
from official.common import distribute_utils as distribution_utils
#from official.utils.misc import model_helpers

# Evaluation metrics
_WER_KEY = "WER"
_CER_KEY = "CER"

# Simple debug output (print)
DEBUG_SHAPES = True

# Simulate multiple CPUs with virtual devices
N_VIRTUAL_DEVICES = 1

def generate_dataset(dataset_csv):
    """Generate a speech dataset."""
    audio_conf = dataset.AudioConfig(sample_rate=flags_obj.sample_rate,
                                    window_ms=flags_obj.window_ms,
                                    stride_ms=flags_obj.stride_ms,
                                    num_feature_bins=flags_obj.num_feature_bins,
                                    normalize=True)
    
    logging.info(f"Dataset CSV: {os.path.join(flags_obj.data_dir, dataset_csv)}")
    logging.info(f"Speech data: {os.path.join(flags_obj.data_dir, flags_obj.speech_dir)}")
    logging.info(f"Vocabulary data: {os.path.join(flags_obj.data_dir, flags_obj.vocabulary_file)}")

    train_data_conf = dataset.DatasetConfig(
        audio_conf,
        os.path.join(flags_obj.data_dir, dataset_csv),
        os.path.join(flags_obj.data_dir, flags_obj.vocabulary_file),
        os.path.join(flags_obj.data_dir, flags_obj.speech_dir),
        flags_obj.sortagrad
    )
    speech_dataset = dataset.DeepSpeechDataset(train_data_conf)
    return speech_dataset

def per_device_batch_size(batch_size, num_gpus):
    """For multi-gpu, batch-size must be a multiple of the number of GPUs.


    Note that distribution strategy handles this automatically when used with
    Keras. For using with Estimator, we need to get per GPU batch.

    Args:
        batch_size: Global batch size to be divided among devices. This should be
        equal to num_gpus times the single-GPU batch_size for multi-gpu training.
        num_gpus: How many GPUs are used with DistributionStrategies.

    Returns:
        Batch size per device.

    Raises:
        ValueError: if batch_size is not divisible by number of devices
    """
    if num_gpus <= 1:
        return batch_size

    remainder = batch_size % num_gpus
    if remainder:
        err = ('When running with multiple GPUs, batch size '
            'must be a multiple of the number of available GPUs. Found {} '
            'GPUs with a batch size of {}; try --batch_size={} instead.'
            ).format(num_gpus, batch_size, batch_size - remainder)
        raise ValueError(err)
    return int(batch_size / num_gpus)

def evaluate_model(model):
    """Evaluate the model performance using WER anc CER as metrics.

    The evaluation dataset indicated by flags_obj.eval_data_csv is used.

    Args:
        model: Keras trained model to evaluate.

    Returns:
        Dictionary with evaluation results:
            'WER': Word Error Rate
            'CER': Character Error Rate
    """
    # Evaluation
    logging.info("Starting to evaluate...")
   
    # Input dataset
    eval_speech_dataset = generate_dataset(flags_obj.eval_data_csv)
    speech_labels       = eval_speech_dataset.speech_labels
    entries             = eval_speech_dataset.entries

    # Input dataset (generator function)
    input_dataset_eval = dataset.input_fn(flags_obj.batch_size, eval_speech_dataset)

    # Evaluate
    probs = model.predict(
        x=input_dataset_eval,
        steps=1,
        verbose=2,
    )

    num_of_examples = len(probs)
    targets = [entry[2] for entry in entries]  # The ground truth transcript

    total_wer, total_cer = 0, 0
    greedy_decoder = decoder.DeepSpeechDecoder(speech_labels, blank_index=len(speech_labels))
    for i in range(num_of_examples):
        # Decode string.
        decoded_str = greedy_decoder.decode(probs[i])
        # Compute CER.
        total_cer += greedy_decoder.cer(decoded_str, targets[i]) / float(
            len(targets[i]))
        # Compute WER.
        total_wer += greedy_decoder.wer(decoded_str, targets[i]) / float(
            len(targets[i].split()))

        # Output the transcripts
        logging.info(f"Target: {targets[i][:-1]}")
        logging.info(f"Predicted: {decoded_str}")
        logging.info(f"---")

    # Get mean value
    total_cer /= num_of_examples
    total_wer /= num_of_examples

    eval_results = {
        _WER_KEY: total_wer,
        _CER_KEY: total_cer,
    }

    logging.info(f"==> Evaluation results: WER = {eval_results[_WER_KEY]:.2f}, CER = {eval_results[_CER_KEY]:.2f}")

    return eval_results


def train_model(_):
    """Run DeepSpeech2 training loop (TF2/Keras).
    
    Args:
        flags.FLAGS from absl 

    Returns:
        Keras trained model.   
    """
    
    # Initialise random seed 
    tf.random.set_seed(flags_obj.seed)

    # Number of GPUs to use
    num_gpus = flags_core.get_num_gpus(flags_obj)   

    # Simulate multiple CPUs with virtual devices
    if num_gpus == 0 and N_VIRTUAL_DEVICES > 1:
        physical_devices = tf.config.list_physical_devices("CPU")
        tf.config.set_logical_device_configuration(
            physical_devices[0], [tf.config.LogicalDeviceConfiguration() for _ in range(N_VIRTUAL_DEVICES)])

    # Available devices (CPU+GPU/TPU)
    print("Available devices:")
    for i, device in enumerate(tf.config.list_logical_devices()):
        print("%d) %s" % (i, device))

    # Data preprocessing
    logging.info("Data preprocessing...")
    train_speech_dataset = generate_dataset(flags_obj.train_data_csv)
    test_speech_dataset = generate_dataset(flags_obj.test_data_csv)
    
    # Number of label classes. Label string is generated from the --vocabulary_file file
    num_classes = len(train_speech_dataset.speech_labels)

    # Set distributionn strategy 
    # Uses MirroredStrategy as default, distribution_strategy="mirrored"
    # MirroredStrategy trains your model on multiple GPUs on a single machine/worker. 
    # For synchronous training on many GPUs on multiple workers, use distribution_strategy="multi_worker_mirrored"
    distribution_strategy = distribution_utils.get_distribution_strategy(num_gpus=num_gpus) 
    # if num_gpus == 0:
    #   devices = ["device:CPU:0"]
    # else:
    #   devices = ["device:GPU:%d" % i for i in range(num_gpus)]
    # distribution_strategy = tf.distribute.MirroredStrategy(devices=devices)   

    # Input datasets (generator functions)
    # The input batch of data specified in flags_obj.batch_size (called global batch) is automatically split 
    # into num_replicas_in_sync different sub-batches (called local batches)
    per_replica_batch_size = per_device_batch_size(flags_obj.batch_size, distribution_strategy.num_replicas_in_sync)
    input_dataset_train = dataset.input_fn(per_replica_batch_size * distribution_strategy.num_replicas_in_sync, train_speech_dataset)
    input_dataset_test  = dataset.input_fn(per_replica_batch_size * distribution_strategy.num_replicas_in_sync, test_speech_dataset)

    # These are iterators over DataSet
    #train_dist_dataset = distribution_strategy.experimental_distribute_dataset(input_dataset_train)
    #test_dist_dataset  = distribution_strategy.experimental_distribute_dataset(input_dataset_test)

    # Get one element from the input dataset (= tuple of (features_dict, labels))
    # and print some info about it
    if DEBUG_SHAPES:
        features_dict = list(input_dataset_train.take(1).as_numpy_iterator())[0][0]
        labels        = list(input_dataset_train.take(1).as_numpy_iterator())[0][1]
        features      = features_dict["features"]
        input_length  = features_dict["input_length"]
        labels_length = features_dict["labels_length"]
        print(f"input_length_shape = {input_length.shape}\nlabels_shape = {labels.shape}\nlabels_length_shape = {labels_length.shape}\nfeatures_shape = {features.shape}")
        #print(f"input_length = {input_length}\nlabels = {labels}\nlabels_length = {labels_length}\n")

    # Use distribution strategy for multi-gpu on single worker training (when available)
    logging.info("Model generation and distribution...")

    # tf.distribute calls the input function on the CPU device of each of the workers
    # See: https://www.tensorflow.org/tutorials/distribute/keras
    # https://www.tensorflow.org/tutorials/distribute/input
    with distribution_strategy.scope():

        # Model
        model = ds2_model(
            flags_obj.num_feature_bins,
            num_classes, 
            flags_obj.rnn_hidden_layers, flags_obj.rnn_type,
            flags_obj.is_bidirectional, flags_obj.rnn_hidden_size,
            flags_obj.use_bias
        )

        # Optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=flags_obj.learning_rate)

        # Compile the model
        model.compile(
            optimizer=optimizer, 
            loss=None
        )

    # Plot summary of the model
    if flags_obj.plot_model:
        logging.info("Plot model summary...")

        model.summary(line_length=110)

        # tf.keras.utils.plot_model(
        #     model, 
        #     to_file=os.path.join(flags_obj.model_dir, "ds2_model.png"), 
        #     show_shapes=True,
        #     show_dtype=True,
        #     show_layer_names=True,
        # )


    # Set output paths and callbacks for training
    callbacks = []

    # TensorBoard will store logs for each epoch and graph performance
    # tensorboard_path = os.path.join(  # Timestamp included to enable timeseries graphs
    #     flags_obj.model_dir, "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # )
    # logging.info(f"TB logs: {tensorboard_path}")
    # callbacks.append(
    #     tf.keras.callbacks.TensorBoard(
    #         log_dir=tensorboard_path, 
    #         histogram_freq=1)
    # )

    # 'EarlyStopping' to stop training when the model is not enhancing anymore
    callbacks.append(
        tf.keras.callbacks.EarlyStopping(
            monitor="loss",
            patience=5, 
            verbose=1,
            restore_best_weights=True)
    )
    # 'ModelCheckPoint' to always keep the model from each epoch
    checkpoint_path = os.path.join(flags_obj.model_dir, "save_at_{epoch}")
    logging.info(f"Checkpoints: {checkpoint_path}")
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="loss", 
            verbose=1,
            save_best_only=False)
    )

    # Train/fit
    logging.info("Starting to train...")

    model.fit(
        x=input_dataset_train,
        validation_data=input_dataset_test,
        epochs=flags_obj.train_epochs,
        callbacks=callbacks,
        verbose=2, # type: ignore
    )

    # Save the model (creates a SavedModel folder)
    save_path = os.path.join(flags_obj.model_dir,"ds2_final")
    logging.info(f"Saving trained model to {save_path}...")
    model.save(save_path)
    # It can be used to reconstruct the model identically
    #loaded_model = tf.keras.models.load_model(save_path)

    return model

def define_deep_speech_flags():
    """Add flags for train_model."""
    # Add common flags
    flags_core.define_base(
        data_dir=False,  # we use train_data_csv and test_data_csv instead
        train_epochs=True,
        hooks=True,
        num_gpu=True,
        epochs_between_evals=True
    )
    flags_core.define_performance(
        num_parallel_calls=False,
        inter_op=False,
        intra_op=False,
        synthetic_data=False,
        max_train_steps=False,
        dtype=False
    )
    flags_core.define_benchmark()
    flags.adopt_module_key_flags(flags_core)

    #
    # Deafult parameter values configurable from JSON or args
    #
    # Default dataset path
    _DATA_DIR = "./data/cv-corpus-8.0-2022-01-19/hu"

    # Default vocabulary file under _DATA_DIR 
    _VOCABULARY_FILE = "vocabulary-hu.txt"

    # Default WAV files path under _DATA_DIR
    _SPEECH_DIR = "clips-wav"

    # Default csv file names under _DATA_DIR
    _TRAIN_CSV = "train.csv"
    _TEST_CSV  = "test.csv"
    _DEV_CSV   = "dev.csv"

    # Default output model path
    _MODEL_DIR  = "model_v0"
    # Default model (folder) to evaluate under _MODEL_DIR
    _MODEL_EVAL = "ds2_final"

    # Default training hyper-parameters
    _EPOCHS = 10
    _BATCH = 64

    # Load parameter values from JSON file, if it exist
    # Overwrites the above defaults!
    if tf.io.gfile.exists("trainer/config.json"):
        with open("trainer/config.json", 'r') as f:
            params = json.load(f)

        logging.info("Default parameters: trainer/config.json")
        _DATA_DIR   = params["data_dir"]
        _SPEECH_DIR = params["speech_subdir"]
        _VOCABULARY_FILE = params["vocabulary_txt"]
        _TRAIN_CSV  = params["train_csv"]
        _TEST_CSV   = params["test_csv"]
        _DEV_CSV    = params["dev_csv"]
        _MODEL_DIR  = params["model_dir"]
        _MODEL_EVAL = params["model_eval"]

        _EPOCHS = params["train_epochs"]
        _BATCH  = params["batch_size"]

    #
    # Parameters configurable from JSON or as args
    #
    # Training parameters
    flags_core.set_defaults(
        model_dir=_MODEL_DIR,
        train_epochs=_EPOCHS,
        batch_size=_BATCH,
        hooks="")

    # Input data flags
    flags.DEFINE_string(
        name="data_dir",
        default=_DATA_DIR,
        help=flags_core.help_wrap("The file path of the datasets."))

    flags.DEFINE_string(
        name="train_data_csv",
        default=_TRAIN_CSV,
        help=flags_core.help_wrap("The csv file path of the train dataset, under data_dir."))

    flags.DEFINE_string(
        name="test_data_csv",
        default=_TEST_CSV,
        help=flags_core.help_wrap("The csv file path of the test dataset, under data_dir."))

    flags.DEFINE_string(
        name="eval_data_csv",
        default=_DEV_CSV,
        help=flags_core.help_wrap("The csv file path of the evaluation dataset, under data_dir."))

    flags.DEFINE_string(
        name= "speech_dir",
        default=_SPEECH_DIR,
        help=flags_core.help_wrap("The speech files path, under under data_dir"))

    flags.DEFINE_string(
        name="vocabulary_file", 
        default=_VOCABULARY_FILE,
        help=flags_core.help_wrap("The vocabulary file path, under data_dir."))

    flags.DEFINE_string(
        name="model_eval",
        default=_MODEL_EVAL,
        help=flags_core.help_wrap("The model (folder) to evaluate, under model_dir."))

    #
    # Parameters configurable only as args
    #
    # Deep speech flags
    flags.DEFINE_integer(
        name="seed", 
        default=1,
        help=flags_core.help_wrap("The random seed."))

    flags.DEFINE_bool(
        name="sortagrad", 
        default=True,
        help=flags_core.help_wrap(
            "If true, sort examples by audio length and perform no "
            "batch_wise shuffling for the first epoch."))

    flags.DEFINE_integer(
        name="sample_rate", 
        default=16000,
        help=flags_core.help_wrap("The sample rate for audio."))

    flags.DEFINE_integer(
        name="window_ms", 
        default=20,
        help=flags_core.help_wrap("The frame length for spectrogram."))

    flags.DEFINE_integer(
        name="stride_ms", 
        default=10,
        help=flags_core.help_wrap("The frame step."))

    flags.DEFINE_integer(
        name="num_feature_bins", 
        default=161,
        help=flags_core.help_wrap("The size of the spectrogram."))

    # RNN related flags
    flags.DEFINE_integer(
        name="rnn_hidden_size", 
        default=800,
        help=flags_core.help_wrap("The hidden size of RNNs."))

    flags.DEFINE_integer(
        name="rnn_hidden_layers", 
        default=4,
        help=flags_core.help_wrap("The number of RNN layers."))

    flags.DEFINE_bool(
        name="use_bias", 
        default=True,
        help=flags_core.help_wrap("Use bias in the last fully-connected layer"))

    flags.DEFINE_bool(
        name="is_bidirectional", 
        default=True,
        help=flags_core.help_wrap("If rnn unit is bidirectional"))

    flags.DEFINE_enum(
        name="rnn_type", 
        default="gru",
        enum_values=SUPPORTED_RNNS.keys(),
        case_sensitive=False,
        help=flags_core.help_wrap("Type of RNN cell."))

    # Training related flags
    flags.DEFINE_float(
        name="learning_rate", 
        default=5e-4,
        help=flags_core.help_wrap("The initial learning rate."))

    flags.DEFINE_bool(
        name = "plot_model", 
        default=True,
        help=flags_core.help_wrap("If model is to be shown, ploted and saved to ds2_model.png"))

    # Evaluation metrics threshold
    flags.DEFINE_float(
        name="wer_threshold", 
        default=None,
        help=flags_core.help_wrap(
            "If passed, training will stop when the evaluation metric WER is "
            "greater than or equal to wer_threshold. For libri speech dataset "
            "the desired wer_threshold is 0.23 which is the result achieved by "
            "MLPerf implementation."))


def main(_):
    """Entry point when running locally from absl.app"""

    # Train model
    if flags_obj.train_epochs > 0:
        model = train_model(flags_obj)

    # Load model and evaluate
    load_path = os.path.join(flags_obj.model_dir, flags_obj.model_eval)
    loaded_model = tf.keras.models.load_model(load_path)
    evaluate_model(loaded_model)

if __name__ == "__main__":
    logging.set_verbosity(logging.DEBUG)
    define_deep_speech_flags()
    flags_obj = flags.FLAGS
    absl_app.run(main)
