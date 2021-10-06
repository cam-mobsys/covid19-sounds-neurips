# coding: utf-8
# author: T.XIA

"""Defines the 'audio' model used to classify the VGGish features."""

from __future__ import print_function

import sys

import model_params as params
import tensorflow as tf
import tf_slim as slim

sys.path.append("../vggish")
import vggish_slim  # noqa: E402


def define_audio_slim(
    modality=params.MODALITY,
    reg_l2=params.L2,
    rnn_units=32,
    num_units=params.NUM_UNITS,
    modfuse=params.MODFUSE,
    train_vgg=False,
):
    """Defines the audio TensorFlow model.

    All ops are created in the current default graph, under the scope 'audio/'.

    The input is a placeholder named 'audio/vggish_input' of type float32 and
    shape [batch_size, feature_size] where batch_size is variable and
    feature_size is constant, and feature_size represents a VGGish output feature.
    The output is an op named 'audio/prediction' which produces the activations of
    a NUM_CLASSES layer.

    Args:
        training: If true, all parameters are marked trainable.

    Returns:
        The op 'mymodel/logits'.
    """

    embeddings = vggish_slim.define_vggish_slim(train_vgg)  # (? x 128) vggish is the pre-trained model
    print("model summary:", train_vgg)

    with slim.arg_scope(
        [slim.fully_connected],
        weights_initializer=tf.truncated_normal_initializer(
            stddev=params.INIT_STDDEV, seed=0
        ),  # 1 is the best for old data
        biases_initializer=tf.zeros_initializer(),
        weights_regularizer=slim.l2_regularizer(reg_l2),
    ), tf.variable_scope("mymodel"):

        index = tf.placeholder(dtype=tf.int32, shape=(1, 1), name="index")  # split B C V
        index2 = tf.placeholder(dtype=tf.int32, shape=(1, 1), name="index2")
        dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_rate")

        if "B" in modality:
            with tf.name_scope("Breath"):
                # breath branch
                fc_vgg_breath = embeddings[0 : index[0, 0], :]  # (len, 128)
                fc1_b = tf.reduce_mean(fc_vgg_breath, axis=0)
                fc2_b = tf.reshape(fc1_b, (-1, 128), name="vgg_b")

        if "C" in modality:
            with tf.name_scope("Cough"):
                # cough branch
                fc_vgg_cough = embeddings[index[0, 0] : index2[0, 0], :]
                fc1_c = tf.reduce_mean(fc_vgg_cough, axis=0)
                fc2_c = tf.reshape(fc1_c, (-1, 128), name="vgg_c")

        if "V" in modality:
            with tf.name_scope("Voice"):
                # voice branch
                fc_vgg_voice = embeddings[index2[0, 0] :, :]
                fc1_v = tf.reduce_mean(fc_vgg_voice, axis=0)
                fc2_v = tf.reshape(fc1_v, (-1, 128), name="vgg_v")

        with tf.name_scope("Output"):
            # fusing and classifier
            if modality == "BCV":  # combination of three modalities
                if modfuse == "concat":
                    fc3 = tf.concat((fc2_b, fc2_c, fc2_v), axis=1, name="vgg_comb")
                if modfuse == "add":
                    fc3 = tf.add(fc2_b, fc2_c, fc2_v, name="vgg_comb")
            if modality == "B":
                fc3 = fc2_b
            if modality == "C":
                fc3 = fc2_c
            if modality == "V":
                fc3 = fc2_v

            # classification
            fc3_dp = tf.nn.dropout(fc3, dropout_keep_prob[0, 0], seed=0)
            fc4 = slim.fully_connected(fc3_dp, num_units)
            fc4_dp = tf.nn.dropout(fc4, dropout_keep_prob[0, 0], seed=0)
            logits = slim.fully_connected(fc4_dp, params.NUM_CLASSES, activation_fn=None, scope="logits")
            tf.nn.softmax(logits, name="prediction")

        with tf.name_scope("symptom"):
            fc5_dp = tf.nn.dropout(fc3, dropout_keep_prob[0, 0], seed=0)
            fc5 = slim.fully_connected(fc5_dp, num_units)
            fc6_dp = tf.nn.dropout(fc5, dropout_keep_prob[0, 0], seed=0)
            logits_sym = slim.fully_connected(fc6_dp, params.NUM_SYMPTOMS, activation_fn=None, scope="logits_sym")
            tf.nn.sigmoid(logits_sym, name="prediction_sym")
        return logits, logits_sym


def load_audio_slim_checkpoint(session, checkpoint_path):
    """Loads a pre-trained audio-compatible checkpoint.

    This function can be used as an initialization function (referred to as
    init_fn in TensorFlow documentation) which is called in a Session after
    initializating all variables. When used as an init_fn, this will load
    a pre-trained checkpoint that is compatible with the audio model
    definition. Only variables defined by audio will be loaded.

    Args:
        session: an active TensorFlow session.
        checkpoint_path: path to a file containing a checkpoint that is
          compatible with the audio model definition.
    """

    # Get the list of names of all audio variables that exist in
    # the checkpoint (i.e., all inference-mode audio variables).
    with tf.Graph().as_default():
        define_audio_slim(training=False)
        audio_var_names = [v.name for v in tf.global_variables()]

    # Get list of variables from exist graph which passed by session
    with session.graph.as_default():
        global_variables = tf.global_variables()

    # Get the list of all currently existing variables that match
    # the list of variable names we just computed.
    audio_vars = [v for v in global_variables if v.name in audio_var_names]

    # Use a Saver to restore just the variables selected above.
    saver = tf.train.Saver(audio_vars, name="audio_load_pretrained", write_version=1)
    saver.restore(session, checkpoint_path)
