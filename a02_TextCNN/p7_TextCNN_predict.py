import tensorflow as tf
import numpy as np
import pickle as pkl
from a02_TextCNN.p7_TextCNN_model import TextCNN
# from a02_TextCNN.data_util import create_vocabulary,load_data_multilabel
import os
# import word2vec

#configuration
FLAGS=tf.flags.FLAGS

tf.flags.DEFINE_string("traning_data_path", "../data/sample_multiple_label.txt","path of traning data.") #sample_multiple_label.txt-->train_label_single100_merge
tf.flags.DEFINE_integer("vocab_size", 323069, "maximum vocab size.")

tf.flags.DEFINE_float("learning_rate", 0.0003, "learning rate")
tf.flags.DEFINE_integer("batch_size", 256, "Batch size for training/evaluating.")  # 批处理的大小 32-->128
tf.flags.DEFINE_integer("decay_steps", 1000, "how many steps before decay learning rate.")  # 6000批处理的大小 32-->128
tf.flags.DEFINE_float("decay_rate", 1.0, "Rate of decay for learning rate.")  # 0.65一次衰减多少
tf.flags.DEFINE_string("ckpt_dir", "text_cnn_title_desc_checkpoint/", "checkpoint location for the model")
tf.flags.DEFINE_integer("sentence_len", 2000, "max sentence length")
tf.flags.DEFINE_integer("embed_size", 128, "embedding size")
tf.flags.DEFINE_boolean("is_training", False, "is traning.true:tranining,false:testing/inference")
tf.flags.DEFINE_integer("num_epochs", 10, "number of epochs to run.")
tf.flags.DEFINE_integer("validate_every", 1, "Validate every validate_every epochs.")  # 每10轮做一次验证
# tf.flags.DEFINE_boolean("use_embedding", False, "whether to use embedding or not.")
tf.flags.DEFINE_integer("num_filters", 128, "number of filters")  # 256--->512
# tf.flags.DEFINE_string("word2vec_model_path", "word2vec-title-desc.bin","word2vec's vocabulary and vectors")
tf.flags.DEFINE_string("name_scope", "cnn", "name scope value.")
tf.flags.DEFINE_boolean("multi_label_flag", True, "use multi label or single label.")
filter_sizes = [6, 7, 8]


def main(_):
    with open('../data_preprocessed/data_train_keras.pkl', 'rb') as fw_train, \
            open('../data_preprocessed/data_test_keras.pkl', 'rb') as fw_test:
        trainX, trainY = pkl.load(fw_train)
        testX = pkl.load(fw_test)
    # trainY = trainY.argmax(axis=-1)

    print("length of training data:", len(trainX), ";length of validation data:",len(testX))
    print("trainX[0]:", trainX[0])
    print("trainY[0]:", trainY[0])

    # 2.create session.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # Instantiate Model
        textCNN = TextCNN(filter_sizes, FLAGS.num_filters, 19,
                          FLAGS.learning_rate, FLAGS.batch_size,
                          FLAGS.decay_steps, FLAGS.decay_rate, FLAGS.sentence_len,
                          323069,
                          FLAGS.embed_size, FLAGS.is_training,
                          multi_label_flag=FLAGS.multi_label_flag)
        # Initialize Save
        saver = tf.train.Saver(max_to_keep=10)
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir))
        # 3.feed data & training
        number_of_training_data = len(trainX)
        batch_size = FLAGS.batch_size
        iteration = 0

        logits = []
        y_pre = []
        for start, end in zip(range(0, number_of_training_data, batch_size),
                              range(batch_size, number_of_training_data+batch_size, batch_size)):
            iteration = iteration + 1
            feed_dict = {textCNN.input_x: testX[start:end],
                         textCNN.dropout_keep_prob: 1, textCNN.iter: iteration,
                         textCNN.tst: not FLAGS.is_training}
            logits_batch, y_pre_batch = sess.run(
                [textCNN.logits, textCNN.possibility], feed_dict)
            logits.append(logits_batch)
            y_pre.append(y_pre_batch)

        logits = np.concatenate(logits, axis=0)
        y_pre = np.concatenate(y_pre, axis=0)

        predictions = y_pre.argmax(axis=1)


if __name__ == "__main__":
    tf.app.run()
