# -*- coding: utf-8 -*-
"""
Created on Wed OCt 9 17:18:28 2020

@author: XT
"""
from __future__ import print_function

import argparse
import os
import random

import nni
import numpy as np
import tensorflow as tf
import tf_slim as slim
from tfdeterminism import patch

patch()
SEED = 123
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
tf.set_random_seed(SEED)

import sys  # noqa: E402

import model_params as params  # noqa: E402
import model_util as util  # noqa: E402
from model_network import define_audio_slim  # noqa: E402

sys.path.append("../vggish")
import warnings  # noqa: E402

from vggish_slim import load_vggish_slim_checkpoint  # noqa: E402

warnings.filterwarnings("ignore")
tf.logging.set_verbosity(tf.logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument("--train_name", type=str, default=params.AUDIO_TRAIN_NAME, help="Name of this programe.")
parser.add_argument("--task", type=str, default=params.TASK, help="Name of the task.")
parser.add_argument("--data_name", type=str, default=params.DATA_NAME, help="Original data path.")
parser.add_argument("--is_aug", type=bool, default=False, help="Add data augmentation.")
parser.add_argument("--restore_if_possible", type=bool, default=True, help="Restore variables.")
parser.add_argument("--modality", type=str, default=params.MODALITY, help="Breath, cough, or voice.")
parser.add_argument("--reg_l2", type=float, default=params.L2, help="L2 regulation.")
parser.add_argument("--lr_decay", type=float, default=params.LEARNING_RATE_DECAY, help="learning rate decay rate.")
parser.add_argument("--dropout_rate", type=float, default=params.DROPOUT_RATE, help="Dropout rate.")
parser.add_argument(
    "--early_stop", type=str, default=params.EARLY_STOP, help="The indicator on validation set to stop training."
)
parser.add_argument("--modfuse", type=str, default=params.MODFUSE, help="The method to fusing modalities.")
parser.add_argument("--is_diff", type=bool, default=False, help="Whether to use differential learing rate.")
parser.add_argument("--train_vgg", type=bool, default=False, help="Fine tuning Vgg")
parser.add_argument("--rnn_units", type=int, default=32, help="The numer of unit in rnn.")
parser.add_argument("--loss_weight", type=float, default=1, help="loss weight for symptom prediction.")
parser.add_argument("--is_sym", type=bool, default=False, help="Use symptom prediction.")

parser.add_argument("--lr1", type=float, default=5e-5, help="learning rate for Vgg layers.")
parser.add_argument("--lr2", type=float, default=1e-4, help="learning rate for top layers.")
parser.add_argument("--num_units", type=int, default=64, help="The numer of unit in network.")
parser.add_argument("--trained_layers", type=int, default=12, help="The number Vgg layers to be fine tuned.")

FLAGS, _ = parser.parse_known_args()
# FLAGS = flags.FLAGS
tuner_params = nni.get_next_parameter()
FLAGS = vars(FLAGS)
FLAGS.update(tuner_params)

data_dir = os.path.join(params.TF_DATA_DIR, FLAGS["data_name"])  # ./data
tensorboard_dir = os.path.join(params.TENSORBOARD_DIR, FLAGS["train_name"])  # ./data/tensorbord/
audio_ckpt_dir = os.path.join(
    params.AUDIO_CHECKPOINT_DIR, FLAGS["train_name"]
)  # ./data/train/ name_modality: name, with/out feature, modality: B, C, V, BCV
name_pre = (
    FLAGS["modality"]
    + "_"
    + "Dp"
    + str(FLAGS["dropout_rate"])
    + "_"
    + "U"
    + str(FLAGS["num_units"])
    + "_"
    + "R"
    + str(FLAGS["rnn_units"])
)
name_mid = "DC" + str(FLAGS["lr_decay"]) + "_" + "LR" + str(FLAGS["lr1"]) + "_" + str(FLAGS["lr2"])
name_pos = "MF" + str(FLAGS["modfuse"]) + "_" + "Aug" + str(FLAGS["is_aug"])
name_all = name_pre + "__" + name_mid + "__" + name_pos + "__"
print("save:", name_all)

util.maybe_create_directory(tensorboard_dir)
util.maybe_create_directory(audio_ckpt_dir)


def model_summary():
    """Print model to log."""
    print("\n")
    print("=" * 30 + "Model Structure" + "=" * 30)
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)
    print("=" * 60 + "\n")


def _create_data():
    """Create audio `train`, `test` and `val` records file."""
    tf.logging.info("Create records..")
    _check_vggish_ckpt_exists()
    test = util.load_test_data(data_dir)
    tf.logging.info("Dataset size: Test-{} ".format(len(test)))
    return test


def _add_triaining_graph():
    """Define the TensorFlow Graph."""
    with tf.Graph().as_default() as graph:
        logits, logits_sym = define_audio_slim(
            modality=FLAGS["modality"],
            reg_l2=FLAGS["reg_l2"],
            rnn_units=FLAGS["rnn_units"],
            num_units=FLAGS["num_units"],
            modfuse=FLAGS["modfuse"],
            train_vgg=FLAGS["train_vgg"],
        )
        tf.summary.histogram("logits", logits)
        # define training subgraph
        with tf.variable_scope("train"):
            labels = tf.placeholder(tf.float32, shape=[None, params.NUM_CLASSES], name="labels")
            symptoms = tf.placeholder(tf.float32, shape=[None, params.NUM_SYMPTOMS], name="symptoms")
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels, name="cross_entropy")
            symptom_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=symptoms, logits=logits_sym)
            sym_loss = tf.reduce_mean(symptom_entropy, name="sym_loss")
            cla_loss = tf.reduce_mean(cross_entropy, name="cla_loss")
            reg_loss2 = tf.add_n(
                [tf.nn.l2_loss(v) * FLAGS["reg_l2"] for v in tf.trainable_variables() if "bias" not in v.name],
                name="reg_loss2",
            )
            if FLAGS["is_sym"]:
                loss = tf.add(tf.add(cla_loss, reg_loss2), FLAGS["loss_weight"] * sym_loss, name="loss_op")
                # loss = tf.add(reg_loss2, sym_loss, name='loss_op')
            else:
                loss = tf.add(reg_loss2, cla_loss, name="loss_op")

            tf.summary.scalar("loss", loss)
            # training
            global_step = tf.Variable(
                0,
                name="global_step",
                trainable=False,
                collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP],
            )

            # Use Learning Rate Decaying for top layers
            number_decay_steps = 3000 if FLAGS["is_aug"] else 1000  # approciately an epoch
            base_of = FLAGS["lr_decay"]
            lr1 = tf.train.exponential_decay(
                FLAGS["lr1"], global_step, number_decay_steps, base_of, staircase=True, name="train_lr1"
            )
            lr2 = tf.train.exponential_decay(
                FLAGS["lr2"], global_step, number_decay_steps, base_of, staircase=True, name="train_lr2"
            )

            if FLAGS["is_diff"]:  # use different learning rate for vgg and others
                print("--------------learning rate control-----------------")
                var1 = tf.trainable_variables()[0 : FLAGS["trained_layers"]]  # Vggish
                var2 = tf.trainable_variables()[18:]  # FCNs
                train_op1 = tf.train.AdamOptimizer(learning_rate=lr1, epsilon=params.ADAM_EPSILON).minimize(
                    loss, var_list=var1, global_step=global_step, name="train_op1"
                )
                train_op2 = tf.train.AdamOptimizer(learning_rate=lr2, epsilon=params.ADAM_EPSILON).minimize(
                    loss, var_list=var2, global_step=global_step, name="train_op2"
                )  # fixed 'var1'
                train_op = tf.group(train_op1, train_op2, name="train_op")  # noqa: F841
            else:
                optimizer = tf.train.AdamOptimizer(learning_rate=lr2, epsilon=params.ADAM_EPSILON)
                optimizer.minimize(loss, global_step=global_step, name="train_op")
        return graph


def _check_vggish_ckpt_exists():
    """check VGGish checkpoint exists or not."""
    util.maybe_create_directory(params.VGGISH_CHECKPOINT_DIR)
    if not util.is_exists(params.VGGISH_CHECKPOINT):
        url = "https://storage.googleapis.com/audioset/vggish_model.ckpt"
        util.maybe_download(url, params.VGGISH_CHECKPOINT_DIR)


def main(_):
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True

    with tf.Session(graph=_add_triaining_graph(), config=sess_config) as sess:

        # op and tensors
        # dropout_on = np.array([True], dtype=np.bool)
        dropout_off = np.array([False], dtype=np.bool)
        vgg_tensor = sess.graph.get_tensor_by_name(params.VGGISH_INPUT_TENSOR_NAME)
        index_tensor = sess.graph.get_tensor_by_name("mymodel/index:0")
        index2_tensor = sess.graph.get_tensor_by_name("mymodel/index2:0")
        dropout_tensor = sess.graph.get_tensor_by_name("mymodel/dropout_rate:0")
        logit_tensor = sess.graph.get_tensor_by_name("mymodel/Output/prediction:0")
        logitsym_tensor = sess.graph.get_tensor_by_name("mymodel/symptom/prediction_sym:0")
        symptom_tensor = sess.graph.get_tensor_by_name("train/symptoms:0")
        labels_tensor = sess.graph.get_tensor_by_name("train/labels:0")
        # global_step_tensor = sess.graph.get_tensor_by_name("train/global_step:0")
        # lr1_tensor = sess.graph.get_tensor_by_name("train/train_lr1:0")
        # lr2_tensor = sess.graph.get_tensor_by_name("train/train_lr2:0")
        loss_tensor = sess.graph.get_tensor_by_name("train/loss_op:0")
        cla_loss_tensor = sess.graph.get_tensor_by_name("train/cla_loss:0")
        reg_loss_tensor = sess.graph.get_tensor_by_name("train/reg_loss2:0")
        # sym_loss_tensor = sess.graph.get_tensor_by_name("train/sym_loss:0")
        # train_op = sess.graph.get_operation_by_name("train/train_op")

        # summary_op = tf.summary.merge_all()
        # summary_writer = tf.summary.FileWriter(tensorboard_dir, graph=sess.graph)
        saver = tf.train.Saver()

        init = tf.global_variables_initializer()
        sess.run(init)
        load_vggish_slim_checkpoint(sess, params.VGGISH_CHECKPOINT)

        model_summary()

        checkpoint_path = os.path.join(audio_ckpt_dir, name_all + params.AUDIO_CHECKPOINT_NAME)
        if util.is_exists(checkpoint_path + ".meta") and FLAGS["restore_if_possible"]:
            saver.restore(sess, checkpoint_path)

        test_data = _create_data()

        # test loop
        test_batch_losses = []
        probs_all = []
        label_all = []
        for sample in test_data:
            vggcomb, index, index2, labels, symptom = util.get_input(sample)
            [logits, logitsym, loss, clal, regl] = sess.run(
                [logit_tensor, logitsym_tensor, loss_tensor, cla_loss_tensor, reg_loss_tensor],
                feed_dict={
                    vgg_tensor: vggcomb,
                    index_tensor: index,
                    index2_tensor: index2,
                    dropout_tensor: [[1.0]],
                    symptom_tensor: symptom,
                    labels_tensor: [labels],
                },
            )
            test_batch_losses.append(loss)
            probs_all.append(logits)
            label_all.append(labels[1])

        AUC, TPR, TNR, TPR_TNR_9 = util.get_metrics(probs_all, label_all)
        data = [[probs_all[i][0, 1], label_all[i]] for i in range(len(label_all))]
        util.get_CI(data, AUC, TPR, TNR)


if __name__ == "__main__":
    tf.app.run()
