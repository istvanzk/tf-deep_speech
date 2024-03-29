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

#import sys
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


# Create Metric instances to track our loss
loss_tracker = tf.keras.metrics.Mean(name="loss")
val_loss_tracker = tf.keras.metrics.Mean(name="val_loss")
#mae_metric = tf.keras.metrics.MeanAbsoluteError(name="mae")


class CustomModelCTCLoss(tf.keras.Model):
    """Custom Model class with CTC loss"""

    def _init_(self):
        super().__init__()

    def compute_length_after_conv(self, max_time_steps, ctc_time_steps, input_length):
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

    def train_step(self, data):
        """Custom trainig step function

        See: https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit
        https://www.tensorflow.org/tutorials/distribute/custom_training

        The data input will be what gets yielded by dataset at each batch, 
        a tuple of (features_dict, labels)
        """
        features_dict, labels = data

        with tf.GradientTape() as tape:
            # Forward pass
            logits = self(features_dict['features'], training=True)

            # print(f"features_shape = {tf.shape(features_dict['features'])}\nlogits_shape = {tf.shape(logits)}")
            # print(f"features = {features_dict['features']}\nlabels = {labels}")
            # print(f"input_length = {features_dict['input_length']}\nlabel_length = {features_dict['labels_length']}")
            # print(f"logits = {logits}")

            #tf.print("logits: ",  logits, output_stream=sys.stdout)
            #tf.print("labels: ",  labels, output_stream=sys.stdout)

            # CTC input length after convolution
            ctc_input_length = tf.cast(
                self.compute_length_after_conv(
                    tf.shape(features_dict['features'])[1],
                    tf.shape(logits)[1],
                    features_dict['input_length']),
                dtype=tf.int32)

            # print(f"ctc_input_length = {ctc_input_length}")
            #tf.print("ctc_input_length: ",  ctc_input_length, output_stream=sys.stdout)

            # Compute CTC loss
            loss = tf.keras.backend.ctc_batch_cost(
                labels, logits, ctc_input_length, features_dict['labels_length'])
            # loss = tf.nn.compute_average_loss(loss, global_batch_size=GLOBAL_BATCH_SIZE)

            # loss = tf.nn.ctc_loss(
            #     labels=labels,
            #     logits=logits,
            #     label_length=features_dict['labels_length'],
            #     logit_length=ctc_input_length,
            #     logits_time_major=False)
            # loss = tf.reduce_mean(loss)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics
        loss_tracker.update_state(loss)

        return {'loss': loss_tracker.result()}

    def test_step(self, data):
        """Custom testing step function

        See: https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit
        https://www.tensorflow.org/tutorials/distribute/custom_training

        The data input will be what gets yielded by dataset at each batch,
        a tuple of (features_dict, labels)
        """
        features_dict, labels = data

        # Compute predictions
        logits = self(features_dict['features'], training=False)

        # CTC input length after convolution
        ctc_input_length = tf.cast(
            self.compute_length_after_conv(
                tf.shape(features_dict['features'])[1],
                tf.shape(logits)[1],
                features_dict['input_length']),
            dtype=tf.int32)
        # Update the loss
        loss = tf.keras.backend.ctc_batch_cost(
            labels, logits, ctc_input_length, features_dict['labels_length'])

        # Compute our own metrics
        val_loss_tracker.update_state(loss)

        return {'loss': val_loss_tracker.result()}

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [loss_tracker, val_loss_tracker]


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
    #pylint: disable=invalid-name
    padding_conv_1 = (20, 5)
    padding_conv_2 = (10, 5)

    # Input layers
    input_ = tf.keras.layers.Input(shape=(None, input_dim, 1), name="features")
    #inputlng_   = tf.keras.layers.Input(shape=(1), name="input_length")
    #labels_     = tf.keras.layers.Input(shape=(1), name="labels")
    #labelslng_  = tf.keras.layers.Input(shape=(1), name="labels_length")

    # Padding layer
    # Perform symmetric padding on the feature dimension of time_step
    # This step is required to avoid issues when RNN output sequence
    # is shorter than the label length.
    x = tf.keras.layers.ZeroPadding2D(padding=padding_conv_1)(input_)

    # 2-D CNN layer
    # activation=tf.nn.relu6 is applied after BN!
    x = tf.keras.layers.Conv2D(
        filters=_CONV_FILTERS, kernel_size=[41, 11], strides=[2, 2],
        padding="valid", use_bias=False,
        kernel_regularizer=tf.keras.regularizers.l2(0.0005),
        name="conv_1")(x)

    # Batch normalisation
    # During inference (i.e. when using evaluate() or predict() or
    # # when calling the layer/model with the argument training=False)
    # During training (i.e. when using fit() or when calling
    # the layer/model with the argument training=True)
    x = tf.keras.layers.BatchNormalization(
        momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON)(x)

    # Relu6 activation, min(max(features, 0), 6)
    x= tf.keras.layers.ReLU(
        max_value=6, negative_slope=0.0, threshold=0.0)(x)

    x = tf.keras.layers.Dropout(rate=0.5)(x)

    # Padding layer
    # Perform symmetric padding on the feature dimension of time_step
    # This step is required to avoid issues when RNN output sequence
    # is shorter than the label length.
    x = tf.keras.layers.ZeroPadding2D(padding=padding_conv_2)(x)

    # 2-D CNN layer
    x = tf.keras.layers.Conv2D(
        filters=_CONV_FILTERS, kernel_size=[21, 11], strides=[2, 1],
        padding="valid", use_bias=False,
        kernel_regularizer=tf.keras.regularizers.l2(0.0005),
        name="conv_2")(x)

    # Batch normalisation
    x = tf.keras.layers.BatchNormalization(
        momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON)(x)
    
    # Relu6 activation
    x= tf.keras.layers.ReLU(
        max_value=6, negative_slope=0.0, threshold=0.0)(x)

    x = tf.keras.layers.Dropout(rate=0.5)(x)

    # Output of 2nd conv layer with the shape of
    # [batch_size (N), times (T), features (F), channels (C)].
    # Convert the conv output to rnn input.
    x = tf.keras.layers.Reshape(
        (-1, x.shape[-2] * x.shape[-1]))(x) # type: ignore

    # RNN layers w/ batch normalisation
    rnn_cell = SUPPORTED_RNNS[rnn_type]
    for layer_counter in range(num_rnn_layers):

        if is_bidirectional:
            x = tf.keras.layers.Bidirectional(
                tf.keras.layers.RNN(rnn_cell(rnn_hidden_size),
                                    return_sequences=True),
                name=f"bidirectional_{layer_counter}")(x)
        else:
            x = tf.keras.layers.RNN(
                rnn_cell(rnn_hidden_size),
                return_sequences=True,
                name=f"rnn_{layer_counter}")(x)

        # Batch normalisation
        x = tf.keras.layers.BatchNormalization(
            momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON)(x)

    # Output layer: FC layer with batch norm
    x = tf.keras.layers.Dense(
        units=num_classes+1,
        use_bias=use_bias,
        kernel_regularizer=tf.keras.regularizers.l2(0.0005),
        bias_regularizer=tf.keras.regularizers.l2(0.0005))(x)

    # Batch normalisation
    #x = tf.keras.layers.BatchNormalization(
    #    momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON)(x)

    # Softmax activation
    output_ = tf.keras.layers.Softmax()(x)

    # The model
    #inputs=[input_, inputlng_, labels_, labelslng_]
    # model = tf.keras.Model(
    model = CustomModelCTCLoss(
        inputs=input_,
        outputs=output_,
        name="DeepSpeech2_KerasModel")

    return model
    #pylint: enable=invalid-name
