# -*- coding: utf-8 -*-
# Copyright 2022 Istvan Z. Kovacs. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Network structure for DeepSpeech2 model."""
#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

import tensorflow as tf

# Supported rnn cells.
SUPPORTED_RNNS = {
    "lstm": tf.keras.layers.LSTMCell,
    "rnn": tf.keras.layers.SimpleRNNCell,
    "gru": tf.keras.layers.GRUCell,
}

# Parameters for batch normalization.
_BATCH_NORM_EPSILON = 1e-5
_BATCH_NORM_DECAY = 0.997

# Filters of convolution layer
_CONV_FILTERS = 32

def CTCLoss(labels, logits, features_length, input_length):
    """Compute CTC loss """

    def compute_length_after_conv(max_time_steps, ctc_time_steps, input_length):
        """Computes the time_steps/ctc_input_length after convolution.

        Suppose that the original feature contains two parts:
        1) Real spectrogram signals, spanning input_length steps.
        2) Padded part with all 0s.
        The total length of those two parts is denoted as max_time_steps, which is
        the padded length of the current batch. After convolution layers, the time
        steps of a spectrogram feature will be decreased. As we know the percentage
        of its original length within the entire length, we can compute the time steps
        for the signal after conv as follows (using ctc_input_length to denote):
        ctc_input_length = (input_length / max_time_steps) * output_length_of_conv.
        This length is then fed into ctc loss function to compute loss.

        Args:
            max_time_steps: max_time_steps for the batch, after padding.
            ctc_time_steps: number of timesteps after convolution.
            input_length: actual length of the original spectrogram, without padding.

        Returns:
            the ctc_input_length after convolution layer.
        """
        ctc_input_length = tf.cast(tf.multiply(
            input_length, ctc_time_steps), dtype=tf.float32)
        return tf.cast(tf.math.floordiv(
            ctc_input_length, tf.cast(max_time_steps, dtype=tf.float32)), dtype=tf.int32)


    ctc_time_steps = tf.cast(tf.shape(logits)[1], dtype=tf.int32)
    label_length = tf.cast(tf.shape(labels)[1], dtype=tf.int32)

    ctc_input_length = compute_length_after_conv(
        features_length, ctc_time_steps, tf.cast(input_length, dtype=tf.int32))

    #batch_len = tf.cast(tf.shape(labels)[0], dtype="int64")
    #ctc_input_length = ctc_input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    #label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")        

    return tf.reduce_mean(tf.keras.backend.ctc_batch_cost(
        labels, logits, ctc_input_length, label_length))


# def WER(labels, logits):
#     """Compute WER metric"""

#     num_of_examples = len(logits)
#     total_wer = 0
#     greedy_decoder = decoder.DeepSpeechDecoder(speech_labels, blank_index=28)
#     for i in range(num_of_examples):
#         # Decode string.
#         decoded_str = greedy_decoder.decode(logits[i])
#         # Compute WER.
#         total_wer += greedy_decoder.wer(decoded_str, labels[i]) / float(
#             len(labels[i].split()))

#     # Get mean value
#     total_wer /= num_of_examples

#     return total_wer


def ds2_model(input_dim, num_classes, num_rnn_layers, rnn_type, is_bidirectional,
                rnn_hidden_size, use_bias):
    """Define DeepSpeech2 model using Keras Functional API.

    Args:
        input_dim: the dimension of the input features tensor
        num_classes: an integer, the number of output classes/labels.
        num_rnn_layers: an integer, the number of rnn layers. By default, it's 5.
        rnn_type: a string, one of the supported rnn cells: gru, rnn and lstm.
        is_bidirectional: a boolean to indicate if the rnn layer is bidirectional.
        rnn_hidden_size: an integer for the number of hidden states in each unit.
        use_bias: a boolean specifying whether to use bias in the last fc layer.

    Returns:
        A Keras model.   
    """
    padding_conv_1 = (20, 5)
    padding_conv_2 = (10, 5)

    # Input layers
    input_    = tf.keras.layers.Input(shape=(None, input_dim, 1), name="features")
    inputlng_ = tf.keras.layers.Input(shape=(1), name="input_length")
    labels_   = tf.keras.layers.Input(shape=(1), name="labels")

    # Padding layer
    # Perform symmetric padding on the feature dimension of time_step
    # This step is required to avoid issues when RNN output sequence is shorter than the label length.
    x = tf.keras.layers.ZeroPadding2D(padding=padding_conv_1)(input_)

    # 2-D CNN layer
    x = tf.keras.layers.Conv2D(
        filters=_CONV_FILTERS, kernel_size=[41, 11], strides=[2, 1],
        padding="valid", use_bias=False, activation=tf.nn.relu6,
        name="conv_1")(x)

    # Batch normalisation
    # During inference (i.e. when using evaluate() or predict() or when calling the layer/model with the argument training=False)
    # During training (i.e. when using fit() or when calling the layer/model with the argument training=True)
    x = tf.keras.layers.BatchNormalization(
        momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON)(x)

    # Padding layer
    # Perform symmetric padding on the feature dimension of time_step
    # This step is required to avoid issues when RNN output sequence is shorter than the label length.   
    x = tf.keras.layers.ZeroPadding2D(padding=padding_conv_2)(x)

    # 2-D CNN layer
    x = tf.keras.layers.Conv2D(
        filters=_CONV_FILTERS, kernel_size=[21, 11], strides=[2, 1],
        padding="valid", use_bias=False, activation=tf.nn.relu6,
        name="conv_2")(x)

    # Batch normalisation
    x = tf.keras.layers.BatchNormalization(
        momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON)(x)

    # Output of 2nd conv layer with the shape of
    # [batch_size (N), times (T), features (F), channels (C)].
    # Convert the conv output to rnn input.
    batch_size = tf.shape(x)[0]
    feat_size = x.get_shape().as_list()[2]
    x = tf.reshape(
        x,
        [batch_size, -1, feat_size * _CONV_FILTERS])

    # RNN layers
    rnn_cell = SUPPORTED_RNNS[rnn_type]
    for layer_counter in range(num_rnn_layers):
        # No batch normalization on the first layer.
        if (layer_counter != 0):
            x = tf.keras.layers.BatchNormalization(
                momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON)(x)

        if is_bidirectional:
            x = tf.keras.layers.Bidirectional(
                tf.keras.layers.RNN(rnn_cell(rnn_hidden_size),
                                return_sequences=True,
                                name=f"rnn_{layer_counter}"))(x)
        else:
            x = tf.keras.layers.RNN(
                rnn_cell(rnn_hidden_size), 
                    return_sequences=True,
                    name=f"rnn_{layer_counter}")(x)

    # Output layer: FC layer with batch norm
    x = tf.keras.layers.BatchNormalization(
            momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON)(x)
   
    output_ = tf.keras.layers.Dense(
        num_classes+1, use_bias=use_bias, activation="softmax")(x)

    # The model
    model = tf.keras.Model(
        inputs=[input_, inputlng_, labels_], 
        outputs=output_, 
        name="DeepSpeech2_KerasModel")

    # Add custom CTC loss function
    model.add_loss( CTCLoss(labels_, output_, tf.shape(input_)[1], inputlng_) )

    return model