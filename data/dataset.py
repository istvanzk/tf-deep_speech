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
"""Generate tf.data.Dataset object for deep speech training/evaluation."""

import os
import math
import random
# pylint: disable=g-bad-import-order
import numpy as np
#from six.moves import xrange  # pylint: disable=redefined-builtin
#import soundfile
import tensorflow as tf
from absl import logging
# pylint: enable=g-bad-import-order

import data.featurizer as featurizer  # pylint: disable=g-bad-import-order


class AudioConfig(object):
    """Configs for spectrogram extraction from audio."""

    def __init__(self,
               sample_rate,
               window_ms,
               stride_ms,
               num_feature_bins,
               normalize=False):
        """Initialize the AudioConfig class.
        Args:
            sample_rate: an integer denoting the sample rate of the input waveform.
            window_ms: an integer for the length of a spectrogram frame, in ms.
            stride_ms: an integer for the frame stride, in ms.
            num_feature_bins: an integer for the length of the spectrogram, in bins.
            normalize: a boolean for whether apply normalization on the audio feature.
        """

        self.sample_rate = sample_rate
        self.window_ms = window_ms
        self.stride_ms = stride_ms
        self.num_feature_bins = num_feature_bins
        self.normalize = normalize


class DatasetConfig(object):
    """Config class for generating the DeepSpeechDataset."""

    def __init__(self, audio_config, data_path, vocab_file_path, speech_files_path, sortagrad):
        """Initialize the configs for deep speech dataset.

        Args:
            audio_config: AudioConfig object specifying the audio-related configs.
            data_path: a string denoting the full path of a manifest file.
            vocab_file_path: a string specifying the vocabulary file path.
            sortagrad: a boolean, if set to true, audio sequences will be fed by
                increasing length in the first training epoch, which will
                expedite network convergence.

        Raises:
            RuntimeError: file path not exist.
        """

        self.audio_config = audio_config
        assert tf.io.gfile.exists(data_path)
        assert tf.io.gfile.exists(vocab_file_path)
        self.data_path = data_path
        self.vocab_file_path = vocab_file_path
        self.speech_files_path = speech_files_path
        self.sortagrad = sortagrad


def _preprocess_audio(audio_file_path, audio_featurizer, normalize):
    """Load the audio file and compute spectrogram feature."""
    # Read wav file
    file = tf.io.read_file(audio_file_path)
    # Decode the wav file
    data, _ = tf.audio.decode_wav(file)
    data = tf.squeeze(data, axis=-1)
    # Change type to float
    data = tf.cast(data, tf.float32)
    # The spectogram (features)
    spectrogram = featurizer.compute_spectrogram_feature(
        data, audio_featurizer.sample_rate, audio_featurizer.stride_ms,
        audio_featurizer.window_ms, None, audio_featurizer.fft_length)
 
    # Normalisation
    # Perform mean and variance normalization on the spectrogram feature
    if normalize:
        means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
        stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
        spectrogram = (spectrogram - means) / (stddevs + 1e-10)

    # Adding Channel dimension for conv2D input.
    # The spectrogram is used
    # as image-like input data with convolution layers (which expect
    # shape (`batch_size`, `height`, `width`, `channels`).
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram

    # data, _ = soundfile.read(audio_file_path)
    # feature = featurizer.compute_spectrogram_feature(
    #     data, audio_featurizer.sample_rate, audio_featurizer.stride_ms,
    #     audio_featurizer.window_ms)
    # # Feature normalization
    # if normalize:
    #     feature = _normalize_audio_feature(feature)

    # # Adding Channel dimension for conv2D input.
    # feature = np.expand_dims(feature, axis=2)
    # return feature


def _preprocess_data(file_path, speech_files_path):
    """Generate a list of tuples (wav_filename, wav_filesize, transcript).

    Each dataset file contains three columns: "wav_filename", "wav_filesize",
    and "transcript". This function parses the csv file and stores each example
    by the increasing order of audio length (indicated by wav_filesize).
    AS the waveforms are ordered in increasing length, audio samples in a
    mini-batch have similar length.

    Args:
        file_path: a string specifying the csv file path for a dataset.
        speech_files_path: a string specifying the path for the speech files.

    Returns:
        A list of tuples (wav_filename, wav_filesize, transcript) sorted by
        file_size.
    """
    logging.info("Loading data set {}".format(file_path))
    # import csv
    # with open(file_path, newline='', encoding="utf8") as csvfile:
    #     # The metadata file is tab separated.
    #     csvreader = csv.reader(csvfile, delimiter="\t")
    #     lines =[]
    #     for row in csvreader:
    #         # Skip the csv header 
    #         if row[0] == "wav_filename":
    #             continue
    #         else:
    #             # Add the full path to the speech data files
    #             row[0] = os.path.join(speech_files_path, row[0])
    #             lines.append(row)       
        
    with tf.io.gfile.GFile(file_path, "r") as f:
        all_rows = f.readlines()
        lines=[]
        for row in all_rows:
            # The metadata file is tab separated.
            row = row.split("\t", 2)

            # Skip the csv header
            if row[0] == "wav_filename":
                continue
            else:
                # Add the full path to the speech data files
                row[0] = os.path.join(speech_files_path, row[0])
                lines.append(row)       
  
    # import unicodedata
    # lines = [unicodedata.normalize("NFKD",line).encode("utf-8").decode("utf-8").split("\t", 2) for line in lines]  

    # Sort input data by the length of audio sequence.
    lines.sort(key=lambda item: int(item[1]))

    return [tuple(line) for line in lines]


class DeepSpeechDataset(object):
    """Dataset class for training/validation/evaluation of DeepSpeech2 model."""

    def __init__(self, dataset_config):
        """Initialize the DeepSpeechDataset class.

        Args:
            dataset_config: DatasetConfig object.
        """
        self.config = dataset_config
        # Instantiate audio feature extractor.
        self.audio_featurizer = featurizer.AudioFeaturizer(
            sample_rate=self.config.audio_config.sample_rate,
            window_ms=self.config.audio_config.window_ms,
            stride_ms=self.config.audio_config.stride_ms,
            num_feature_bins=self.config.audio_config.num_feature_bins)
        
        # Instantiate text feature extractor.
        self.text_featurizer = featurizer.TextFeaturizer(
            vocab_file=self.config.vocab_file_path)

        # The entries in the data set, tuples (wav_filename, wav_filesize, transcript)
        self.entries = _preprocess_data(self.config.data_path, self.config.speech_files_path)

        # The labels
        self.speech_labels = self.text_featurizer.speech_labels

        # The generated spectrogram size/bins
        self.num_feature_bins = self.config.audio_config.num_feature_bins


def batch_wise_dataset_shuffle(entries, epoch_index, sortagrad, batch_size):
    """Batch-wise shuffling of the data entries.

    Each data entry is in the format of (audio_file, file_size, transcript).
    If epoch_index is 0 and sortagrad is true, we don't perform shuffling and
    return entries in sorted file_size order. Otherwise, do batch_wise shuffling.

    Args:
        entries: a list of data entries.
        epoch_index: an integer of epoch index
        sortagrad: a boolean to control whether sorting the audio in the first
        training epoch.
        batch_size: an integer for the batch size.

    Returns:
        The shuffled data entries.
    """
    shuffled_entries = []
    if epoch_index == 0 and sortagrad:
        # No need to shuffle.
        shuffled_entries = entries
    else:
        # Shuffle entries batch-wise.
        max_buckets = int(math.floor(len(entries) / batch_size))
        total_buckets = [i for i in range(max_buckets)]
        random.shuffle(total_buckets)
        shuffled_entries = []
        for i in total_buckets:
            shuffled_entries.extend(entries[i * batch_size : (i + 1) * batch_size])
        # If the last batch doesn't contain enough batch_size examples,
        # just append it to the shuffled_entries.
        shuffled_entries.extend(entries[max_buckets * batch_size:])

    return shuffled_entries


def input_fn(batch_size, deep_speech_dataset, repeat=1):
    """Input function for model training and evaluation.

    Args:
        batch_size: an integer denoting the size of a (global) batch.
        deep_speech_dataset: DeepSpeechDataset object.
        repeat: an integer for how many times to repeat the dataset.

    Returns:
        a tf.data.Dataset object for model to consume.

    NOTE: The input batch of data (called global batch) is automatically split 
    into stratewgy.num_replicas_in_sync different sub-batches (called local batches)

    For MirroredStrategy: https://keras.io/guides/distributed_training/
    For distributed strategy:
    https://www.tensorflow.org/tutorials/distribute/input
    https://www.tensorflow.org/api_docs/python/tf/data/experimental/DistributeOptions
    """
    # Dataset properties
    data_entries = deep_speech_dataset.entries
    num_feature_bins = deep_speech_dataset.num_feature_bins
    audio_featurizer = deep_speech_dataset.audio_featurizer
    feature_normalize = deep_speech_dataset.config.audio_config.normalize
    text_featurizer = deep_speech_dataset.text_featurizer

    def _gen_data():
        """Dataset generator function."""
        for audio_file, _, transcript in data_entries:
            features = _preprocess_audio(
                audio_file, audio_featurizer, feature_normalize)
            if text_featurizer.dc_labels:
                labels = featurizer.compute_label_feature_dc(
                    transcript, text_featurizer.token_to_index)
            else:    
                labels = featurizer.compute_label_feature(
                    transcript, text_featurizer.token_to_index)
            input_length = [features.shape[0]]
            label_length = [len(labels)]
            # Yield a tuple of (features, labels) where features is a dict containing
            # all info about the actual data features.
            yield (
                {
                    "features": features,
                    "input_length": input_length,
                    "labels_length": label_length
                },
                labels)

    # Updated to use output_signature argument:
    # https://www.tensorflow.org/versions/r2.8/api_docs/python/tf/data/Dataset#from_generator
    dataset = tf.data.Dataset.from_generator(
        _gen_data,
        output_signature=(
            {
                "features": tf.TensorSpec(shape=(None, num_feature_bins, 1), dtype=tf.float32),  # type: ignore
                "input_length": tf.TensorSpec(shape=(1), dtype=tf.int32),  # type: ignore
                "labels_length": tf.TensorSpec(shape=(1), dtype=tf.int32)  # type: ignore
            },
            tf.TensorSpec(shape=(None), dtype=tf.int32)  # type: ignore
        )
    )
    
    # Using an Options instance to enable auto-sharding with tf.distribute (multiple devices on a single worker)
    # https://stackoverflow.com/questions/65917500/tensorflow-keras-generator-turn-off-auto-sharding-or-switch-auto-shard-policiy
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    dataset = dataset.with_options(options)

    # NOTE: Using
    # https://www.tensorflow.org/tutorials/distribute/input#tfdistributestrategydistribute_datasets_from_function
    # solution does not work directly with Keras API .fit() as the returned ditributed dataset is an iterator not a DataSet!
 
    # Repeat and batch the dataset
    dataset = dataset.repeat(repeat)

    # Data is batched with the global batch size to increase distributed performance
    # Padding the features to its max length dimensions
    dataset = dataset.padded_batch(
        batch_size=batch_size,
        padded_shapes=(
            {
                "features": tf.TensorShape([None, num_feature_bins, 1]),
                "input_length": tf.TensorShape([1]),
                "labels_length": tf.TensorShape([1])
            },
            tf.TensorShape([None])),
        drop_remainder=True,
    )

    # Prefetch to improve speed of input pipeline
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


# Based on: https://keras.io/examples/audio/ctc_asr/
def plot_spectrogram(deep_speech_dataset, entry_idx):
    """Visualize an example in our dataset, including the audio clip, the spectrogram and the corresponding label.
    
    Args:
        deep_speech_dataset: a DeepSpeechDataset data set
        entry_idx: the entry index in deep_speech_dataset.entries to plot

    Returns:
        Figure/plot
    
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Dataset properties
    data_entries = deep_speech_dataset.entries
    audio_featurizer = deep_speech_dataset.audio_featurizer
    feature_normalize = deep_speech_dataset.config.audio_config.normalize

    fig = plt.figure(figsize=(8, 5))
    for audio_file, _, transcript in data_entries[entry_idx]:
        spectrogram = _preprocess_audio(
                    audio_file, audio_featurizer, feature_normalize)

        # Spectrogram
        ax = plt.subplot(2, 1, 1)
        ax.imshow(spectrogram, vmax=1)
        ax.set_title(transcript)
        ax.axis("off")
        
        # Wav
        audio, _ = tf.audio.decode_wav(audio_file)
        audio = audio.numpy()
        ax = plt.subplot(2, 1, 2)
        plt.plot(audio)
        ax.set_title("Signal Wave")
        ax.set_xlim(0, len(audio))
        #display.display(display.Audio(np.transpose(audio), rate=16000))

    plt.show()